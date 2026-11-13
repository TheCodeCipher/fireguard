"""Role 1 data-integrity tests for the WSTS dataset interface (P3-R1-02/05).

Synthetic GeoTIFF fixtures with the real 23-band WSTS layout, plus an optional real-data check
(set FIREGUARD_WSTS_ROOT to the folder holding 2018..2021).
"""

from __future__ import annotations

import csv
import os
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("fireguard_ml.training.temporal_data", reason="needs Role 3's sample contract")
from rasterio.transform import Affine  # noqa: E402

from fireguard_ml.data.wsts_dataset import (  # noqa: E402
    ACTIVE_FIRE_BAND,
    PROCESSED_CHANNELS,
    RAW_BANDS,
    RAW_DESCRIPTIONS,
    WSTSAlignmentError,
    WSTSFileError,
    WSTSTemporalSource,
    channel_indices,
    contract_hash,
    padding_metadata,
    process_raw,
    target_from_band,
)
from fireguard_ml.training.temporal_data import (  # noqa: E402
    TemporalSampleSource,
    pad_samples,
    unpad,
)

STRICT = ("viirs_m11", "viirs_i2", "viirs_i1", "slope", "aspect", "elevation",
          "binary_active_fire_mask")  # fmt: skip
EXCLUDED = (
    "ndvi", "evi2", "total_precipitation", "wind_speed", "wind_direction", "minimum_temperature",
    "maximum_temperature", "energy_release_component", "specific_humidity",
    "palmer_drought_severity_index", "gfs_total_precipitation", "gfs_wind_speed",
    "gfs_wind_direction", "gfs_temperature", "gfs_specific_humidity", "active_fire_time",
)  # fmt: skip


@pytest.fixture(scope="module")
def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ml").is_dir() and (parent / "docs").is_dir():
            return parent
    raise RuntimeError("repository root not found")


T0 = Affine(375.0, 0.0, 600000.0, 0.0, -375.0, 4200000.0)


def write_tif(path: Path, data: np.ndarray, transform=T0, crs="EPSG:32610", desc=RAW_DESCRIPTIONS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="float32",
        transform=transform,
        crs=crs,
    ) as d:
        d.write(data.astype(np.float32))
        for i, t in enumerate(desc, 1):
            d.set_band_description(i, t)  # fmt: skip


def raw_day(h, w, day: int, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed + day)
    r = rng.uniform(100, 5000, size=(23, h, w)).astype(np.float32)
    r[13] = rng.uniform(0, 360, size=(h, w))  # aspect, degrees
    r[16] = rng.integers(1, 18, size=(h, w))  # land cover class
    af = np.full((h, w), np.nan, np.float32)
    af[2 + day % 3, 3 : 3 + 2] = 1234.0  # detected at 12:34
    r[ACTIVE_FIRE_BAND] = af
    return r


def make_event(root: Path, ev="fire_1", year=2020, n=8, h=21, w=27, first=date(2020, 8, 1)):
    days = [first + timedelta(days=i) for i in range(n)]
    for i, d in enumerate(days):
        write_tif(root / str(year) / ev / f"{d.isoformat()}.tif", raw_day(h, w, i))
    return days


def samples_csv(path: Path, ev: str, year: int, days, first_index=5) -> Path:
    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["event_id", "event_year", "target_index", "last_input_date", "target_date"])
        for k in range(first_index, len(days)):
            wr.writerow([ev, year, k, days[k - 1], days[k]])
    return path


@pytest.fixture
def world(tmp_path):
    days = make_event(tmp_path / "data")
    return tmp_path / "data", samples_csv(tmp_path / "s.csv", "fire_1", 2020, days), days


# ----------------------------------------------------------------------------- schema


def test_processed_schema_matches_the_pinned_loader_layout():
    assert len(RAW_BANDS) == 23 and len(PROCESSED_CHANNELS) == 40
    assert PROCESSED_CHANNELS[:16] == RAW_BANDS[:16]
    assert PROCESSED_CHANNELS[33:38] == RAW_BANDS[17:22]  # GFS at 33-37
    assert PROCESSED_CHANNELS[38:] == ("active_fire_time", "binary_active_fire_mask")
    assert channel_indices(STRICT) == [0, 1, 2, 12, 13, 14, 39]


def test_strict_indices_agree_with_the_leakage_register(repo_root):
    pol = pytest.importorskip("fireguard_ml.data.channel_policy")
    reg = pol.load_register(repo_root / "docs" / "leakage_audit.csv")
    strict = tuple(pol.strict_channels(reg))
    assert strict == STRICT
    with (repo_root / "docs" / "leakage_audit.csv").open(encoding="utf-8-sig") as f:
        rows = {r["channel_name"]: r for r in csv.DictReader(f)}
    for name, idx in zip(strict, channel_indices(strict), strict=True):
        assert int(rows[name]["processed_channel_index"]) == idx
    assert not set(EXCLUDED) & set(strict)


@pytest.mark.parametrize("bad", EXCLUDED + ("land_cover_water",))
def test_every_excluded_channel_is_refused_by_the_register(repo_root, bad):
    pol = pytest.importorskip("fireguard_ml.data.channel_policy")
    reg = pol.load_register(repo_root / "docs" / "leakage_audit.csv")
    with pytest.raises(ValueError):
        pol.assert_strict_channels([*STRICT, bad], reg)


# ----------------------------------------------------------------------------- transforms


def test_active_fire_time_and_binary_mask_follow_the_loader():
    raw = np.zeros((23, 2, 2), np.float32)
    raw[ACTIVE_FIRE_BAND] = [[np.nan, 1234.0], [59.0, 2359.0]]
    p = process_raw(raw)
    assert p[38].tolist() == [[0.0, 12.0], [0.0, 23.0]]  # HHMM -> hour, NaN -> 0
    assert p[39].tolist() == [[0.0, 1.0], [0.0, 1.0]]  # 00:59 -> hour 0 -> not a fire
    assert target_from_band(raw[ACTIVE_FIRE_BAND]).tolist() == [[0, 1], [1, 1]]


def test_hhmm_floor_quirk_is_reproduced_and_recorded():
    """A detection at 00:00-00:59 UTC becomes hour 0 and is LOST from the binary input mask
    (loader computes the mask after floor-division), but it IS a positive target. Reproduced on
    purpose for parity with the pinned loader; documented in docs/data_dictionary.md 12."""
    raw = np.zeros((23, 1, 1), np.float32)
    raw[ACTIVE_FIRE_BAND] = 30.0
    assert process_raw(raw)[39, 0, 0] == 0.0 and target_from_band(raw[22])[0, 0] == 1


@pytest.mark.parametrize("deg, expected", [(0, 0.0), (90, 1.0), (180, 0.0), (270, -1.0),
                                           (360, 0.0)])  # fmt: skip
def test_aspect_cardinal_directions_under_the_verified_sin_convention(deg, expected):
    raw = np.zeros((23, 1, 1), np.float32)
    raw[13] = deg
    assert process_raw(raw)[13, 0, 0] == pytest.approx(expected, abs=1e-6)


def test_aspect_wraparound_and_nodata():
    raw = np.zeros((23, 1, 3), np.float32)
    raw[13] = [[359.9, 0.1, np.nan]]
    a = process_raw(raw)[13, 0]
    assert abs(a[0] - a[1]) < 0.01  # 359.9 and 0.1 degrees are close after sin
    assert np.isnan(a[2])  # missing aspect stays missing; the fold normaliser imputes it


def test_sin_only_aspect_cannot_tell_east_of_north_from_east_of_south():
    """The known limitation behind the (pending) sin/cos proposal: 30 and 150 degrees collide."""
    raw = np.zeros((23, 1, 2), np.float32)
    raw[13] = [[30.0, 150.0]]
    a = process_raw(raw)[13, 0]
    assert a[0] == pytest.approx(a[1], abs=1e-6)


def test_measured_nan_is_preserved_for_the_fold_normaliser():
    raw = np.ones((23, 1, 2), np.float32)
    raw[0, 0, 0] = np.nan
    assert np.isnan(process_raw(raw)[0, 0, 0])


# ----------------------------------------------------------------------------- the source


def test_source_satisfies_the_contract_and_is_deterministic(world):
    root, s, days = world
    src = WSTSTemporalSource(root, STRICT, 5, s)
    assert isinstance(src, TemporalSampleSource) and src.is_mock is False
    keys = src.keys()
    assert keys == sorted(keys) and keys[0] == ("fire_1", days[5])
    a, b = src.load(keys[0]), WSTSTemporalSource(root, STRICT, 5, s).load(keys[0])
    assert np.array_equal(a.x, b.x, equal_nan=True) and np.array_equal(a.y, b.y)
    assert a.x.shape == (5, 7, 21, 27) and a.input_dates == tuple(days[:5])
    assert a.target_date == days[5] and a.valid.all()
    assert a.transform == tuple(T0)[:6] and a.crs == "EPSG:32610"


def test_inputs_equal_process_raw_of_the_input_files(world):
    root, s, days = world
    src = WSTSTemporalSource(root, STRICT, 3, s)
    smp = src.load(("fire_1", days[6]))
    for t, d in enumerate(days[3:6]):
        with rasterio.open(root / "2020" / "fire_1" / f"{d}.tif") as f:
            want = process_raw(f.read())[channel_indices(STRICT)]
        assert np.array_equal(smp.x[t], want, equal_nan=True)


def test_target_day_contributes_only_the_target_band(world, monkeypatch):
    root, s, days = world
    import fireguard_ml.data.wsts_dataset as w

    tgt = root / "2020" / "fire_1" / f"{days[5]}.tif"
    poisoned = raw_day(21, 27, 5)
    poisoned[:22] = 9e9  # every non-target band of the target day is poison
    write_tif(tgt, poisoned)
    seen = []
    real = w._read
    monkeypatch.setattr(w, "_read", lambda p, b: (seen.append((Path(p).name, tuple(b))),
                                                  real(p, b))[1])  # fmt: skip
    smp = WSTSTemporalSource(root, STRICT, 5, s).load(("fire_1", days[5]))
    assert (f"{days[5]}.tif", (23,)) in seen
    assert all(b == (23,) for name, b in seen if name == f"{days[5]}.tif")
    assert np.nanmax(smp.x) < 9e9  # no target-day value reached the inputs
    assert smp.y.sum() == 2


def test_misaligned_frames_are_rejected(world):
    root, s, days = world
    write_tif(root / "2020" / "fire_1" / f"{days[2]}.tif", raw_day(21, 27, 2),
              transform=Affine(375.0, 0.0, 600001.0, 0.0, -375.0, 4200000.0))  # fmt: skip
    with pytest.raises(WSTSAlignmentError):
        WSTSTemporalSource(root, STRICT, 5, s).load(("fire_1", days[5]))


def test_a_frame_of_a_different_size_is_rejected(world):
    root, s, days = world
    write_tif(root / "2020" / "fire_1" / f"{days[1]}.tif", raw_day(22, 27, 1))
    with pytest.raises(WSTSAlignmentError):
        WSTSTemporalSource(root, STRICT, 5, s).load(("fire_1", days[5]))


def test_missing_and_malformed_files_are_rejected(world):
    root, s, days = world
    (root / "2020" / "fire_1" / f"{days[3]}.tif").unlink()
    with pytest.raises(WSTSFileError, match="missing"):
        WSTSTemporalSource(root, STRICT, 5, s).load(("fire_1", days[5]))
    write_tif(root / "2020" / "fire_1" / f"{days[3]}.tif", raw_day(21, 27, 3),
              desc=RAW_DESCRIPTIONS[::-1])  # fmt: skip
    with pytest.raises(WSTSFileError, match="descriptions"):
        WSTSTemporalSource(root, STRICT, 5, s).load(("fire_1", days[5]))
    (root / "2020" / "fire_1" / f"{days[3]}.tif").write_bytes(b"not a tiff")
    with pytest.raises(WSTSFileError, match="unreadable"):
        WSTSTemporalSource(root, STRICT, 5, s).load(("fire_1", days[5]))


def test_sources_with_shorter_context_score_the_same_samples(world):
    root, s, _ = world
    assert WSTSTemporalSource(root, STRICT, 3, s).keys() == WSTSTemporalSource(
        root, STRICT, 5, s).keys()  # fmt: skip


# ----------------------------------------------------------------------------- padding, masks


def test_padding_keeps_grid_and_marks_padding_invalid(world):
    root, s, days = world
    smp = WSTSTemporalSource(root, STRICT, 5, s).load(("fire_1", days[5]))
    meta = padding_metadata(smp.hw)
    assert meta == {"orig_h": 21, "orig_w": 27, "padded_h": 32, "padded_w": 32,
                    "pad_bottom": 11, "pad_right": 5, "pad_top": 0, "pad_left": 0,
                    "transform_changed": False}  # fmt: skip
    clean = replace(smp, x=np.nan_to_num(smp.x))
    b = pad_samples([clean], 32)
    assert b.x.shape[-2:] == (32, 32) and int(b.valid.sum()) == 21 * 27
    assert not b.valid[0, 0, 21:, :].any() and not b.valid[0, 0, :, 27:].any()
    np.testing.assert_array_equal(unpad(b.y[0, 0], smp.hw).numpy(), smp.y)
    # native cell (r, c) keeps its map coordinates: padding is bottom/right only
    a, b, c, d, e, f = smp.transform
    assert (c, f) == (600000.0, 4200000.0) and (a, e) == (375.0, -375.0) and b == d == 0


def test_evaluation_only_data_cannot_be_offered_as_input(world):
    obs = pytest.importorskip("fireguard_ml.data.observability")
    root, s, days = world
    smp = WSTSTemporalSource(root, STRICT, 5, s).load(("fire_1", days[5]))
    obs.check_input_keys(dict(zip(smp.channels, smp.x[-1], strict=True)))  # strict names pass
    assert "observability" not in {f for f in smp.__dataclass_fields__}
    with pytest.raises(obs.ObservabilityLeakError):
        obs.check_input_keys({"next_day_observability": smp.x[-1, 0]})


def test_both_gfs_precipitation_band_names_are_accepted_and_nothing_else():
    from fireguard_ml.data.wsts_dataset import descriptions_ok

    last = list(RAW_DESCRIPTIONS)
    last[17] = "total_precipitation_surface_last"  # files after the 2019-11-07 regime change
    assert descriptions_ok(RAW_DESCRIPTIONS) and descriptions_ok(last)
    other = list(RAW_DESCRIPTIONS)
    other[13] = "aspect_deg"
    assert not descriptions_ok(other) and not descriptions_ok(RAW_DESCRIPTIONS[:-1])


def test_contract_hash_changes_with_the_contract():
    assert contract_hash(STRICT, 5) != contract_hash(STRICT, 3)
    assert contract_hash(STRICT, 5) == contract_hash(list(STRICT), 5)


# ----------------------------------------------------------------------------- real data


@pytest.mark.skipif(not os.environ.get("FIREGUARD_WSTS_ROOT"), reason="set FIREGUARD_WSTS_ROOT")
def test_real_wsts_sample(repo_root):
    root = Path(os.environ["FIREGUARD_WSTS_ROOT"])
    csv_path = repo_root / "data" / "manifests" / "wsts_eval_samples.csv"
    src = WSTSTemporalSource(root, STRICT, 5, csv_path)
    assert src.keys(), "no scored sample found under FIREGUARD_WSTS_ROOT"
    smp = src.load(src.keys()[0])
    assert smp.crs == "EPSG:32610" and smp.transform[0] == 375.0 and smp.transform[4] == -375.0
    assert smp.x.shape[:2] == (5, 7) and set(np.unique(smp.y)) <= {0, 1}
    assert set(np.unique(smp.x[:, -1])) <= {0.0, 1.0}


def test_strict_input_config_matches_the_code(repo_root):
    yaml = pytest.importorskip("yaml")
    from fireguard_ml.data.wsts_dataset import DATA_CONTRACT_VERSION, PINNED_LOADER_COMMIT

    cfg = yaml.safe_load((repo_root / "configs" / "data" / "strict_input_v1.yaml").read_text())
    assert cfg["version"] == DATA_CONTRACT_VERSION
    assert cfg["pinned_loader"]["commit"] == PINNED_LOADER_COMMIT
    assert tuple(c["name"] for c in cfg["channels"]) == STRICT
    assert [c["processed"] for c in cfg["channels"]] == channel_indices(STRICT)
    assert cfg["aspect_sin_cos"] == "PROPOSED_NOT_APPROVED"
