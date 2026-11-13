"""Role 3 Phase 3/4 pipeline tests on clearly labelled synthetic (MOCK) data.

They prove the protocol mechanics: contract checks, padding, train-only normalisation, M3/M4
training with validation-only selection, the frozen-split guards and leakage-safe OOF export.
They say nothing about WSTS model performance.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("fireguard_ml.evaluation.metrics", reason="needs Role 4's evaluation package")

from fireguard_ml.data.channel_policy import load_register, strict_channels  # noqa: E402
from fireguard_ml.data.event_splits import (  # noqa: E402
    build_p1_manifest,
    read_manifest,
    sha256_file,
    write_manifest,
)
from fireguard_ml.models.checkpointing import load_checkpoint  # noqa: E402
from fireguard_ml.training.normalization import (  # noqa: E402
    NormalizationLeakError,
    fit_normalizer,
)
from fireguard_ml.training.temporal_data import (  # noqa: E402
    SyntheticTemporalSource,
    TemporalContractError,
    TemporalSampleSource,
    pad_samples,
    unpad,
    validate_sample,
)
from fireguard_ml.training.temporal_trainer import (  # noqa: E402
    ProtocolFiles,
    TemporalRunConfig,
    _role3_masked_bce_dice,
    train_fold,
)

T = 3
YEARS = (2018, 2019, 2020, 2021)


@pytest.fixture(scope="module")
def register_path(repo_root):
    return repo_root / "docs" / "leakage_audit.csv"


@pytest.fixture(scope="module")
def channels(register_path):
    return tuple(strict_channels(load_register(register_path)))


def make_world(tmp: Path, channels, register_path, status="FROZEN"):
    """Synthetic events, groups, a FROZEN-style manifest, hashes and an eval-sample list."""
    events, group_of, anchor, event_year = {}, {}, {}, {}
    for y in YEARS:
        for g in range(6):
            gid = f"G{y}_{g:04d}"
            anchor[gid] = y
            ev = f"syn_{y}_{g}"
            events[ev] = (date(y, 7, 1) + timedelta(days=g), 8)
            group_of[ev], event_year[ev] = gid, y
    src = SyntheticTemporalSource(events, channels, T)
    rows = build_p1_manifest(group_of, anchor, event_year)
    man = tmp / "p1_loyo_fold_manifest.csv"
    write_manifest(rows, man)
    hashes = tmp / "p1_split_hashes.txt"
    hashes.write_text(f"# test\n# status={status} synthetic\n{sha256_file(man)}  {man.name}\n")
    samples = tmp / "wsts_eval_samples.csv"
    with samples.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "event_year", "target_index", "last_input_date", "target_date"])
        for ev, tgt in src.keys():
            w.writerow([ev, event_year[ev], 0, tgt - timedelta(days=1), tgt])
    return src, ProtocolFiles(man, hashes, samples, register_path), rows


def tiny(model, channels, **kw):
    return TemporalRunConfig(
        model=model,
        T=T,
        channels=channels,
        base=4,
        depth=2,
        max_epochs=2,
        patience=5,
        grad_accum=2,
        amp=False,
        **kw,
    )


# ----------------------------------------------------------------------------- contract


def test_synthetic_source_satisfies_the_protocol_and_is_mock(channels):
    src = SyntheticTemporalSource({"e": (date(2020, 1, 1), 7)}, channels, T)
    assert isinstance(src, TemporalSampleSource) and src.is_mock
    s = src.load(src.keys()[0])
    assert s.x.shape[:2] == (T, len(channels)) and s.is_mock
    assert s.target_date == s.input_dates[-1] + timedelta(days=1)
    assert np.array_equal(s.x, src.load(src.keys()[0]).x)  # deterministic


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda s: replace(s, input_dates=s.input_dates[::-1]), "consecutive"),
        (lambda s: replace(s, target_date=s.input_dates[-1]), "day after"),
        (lambda s: replace(s, target_date=s.target_date + timedelta(days=1)), "day after"),
        (lambda s: replace(s, channels=s.channels[::-1]), "channel order"),
        (lambda s: replace(s, x=s.x[:-1]), "frames"),
        (lambda s: replace(s, x=np.where(s.x > 0.5, np.nan, s.x).astype(np.float32)), "mask"),
        (lambda s: replace(s, x=np.where(s.x > 1, np.inf, s.x).astype(np.float32)), "inf"),
        (lambda s: replace(s, y=s.y * 2), "binary"),
    ],
)
def test_contract_violations_are_rejected(channels, mutate, msg):
    src = SyntheticTemporalSource({"e": (date(2020, 1, 1), 7)}, channels, T)
    bad = mutate(src.load(src.keys()[0]))
    with pytest.raises(TemporalContractError, match=msg):
        validate_sample(bad, T, channels)


def test_padding_is_bottom_right_invalid_and_exactly_reversible(channels):
    src = SyntheticTemporalSource(
        {"a": (date(2020, 1, 1), 6), "bb": (date(2020, 1, 1), 6)}, channels, T
    )
    s1, s2 = src.load(src.keys()[0]), src.load(src.keys()[-1])
    b = pad_samples([s1, s2], multiple=32)
    assert b.x.shape[-2] % 32 == 0 and b.x.shape[-1] % 32 == 0
    for i, s in enumerate((s1, s2)):
        assert torch.equal(unpad(b.x[i], s.hw), torch.from_numpy(s.x))
        assert int(b.valid[i].sum()) == int(s.valid.sum())  # padding contributes nothing


# ----------------------------------------------------------------------------- normalisation


def test_normaliser_refuses_non_training_events(tmp_path, channels, register_path):
    src, proto, rows = make_world(tmp_path, channels, register_path)
    train = {r.event_id for r in rows if r.fold_test_year == 2019 and r.split == "train"}
    test_ev = next(r.event_id for r in rows if r.fold_test_year == 2019 and r.split == "test")
    keys = [k for k in src.keys() if k[0] in train]
    leaked = keys + [k for k in src.keys() if k[0] == test_ev][:1]
    with pytest.raises(NormalizationLeakError):
        fit_normalizer(src, leaked, train, 2019, "h")
    norm = fit_normalizer(src, keys, train, 2019, "h")
    assert norm.standardized[channels.index("binary_active_fire_mask")] is False
    x = src.load(keys[0])
    z = norm.apply(x.x, x.valid)
    assert np.array_equal(z[:, -1], x.x[:, -1])  # the binary mask passes through untouched


def test_missing_measurements_are_ignored_by_the_fit_and_imputed_with_the_train_mean(
    tmp_path, channels, register_path
):
    src, _, rows = make_world(tmp_path, channels, register_path)
    train = {r.event_id for r in rows if r.fold_test_year == 2019 and r.split == "train"}
    keys = [k for k in src.keys() if k[0] in train][:4]

    class Holey:
        channels, sequence_length, is_mock = src.channels, src.sequence_length, True

        def keys(self):
            return keys

        def load(self, k):
            s = src.load(k)
            x = s.x.copy()
            x[:, 0, :3, :3] = np.nan  # missing reflectance in the corner
            validate_sample(replace(s, x=x), T, channels)  # NaN is legal in measured channels
            return replace(s, x=x)

    holey = fit_normalizer(Holey(), keys, train, 2019, "h")
    clean = fit_normalizer(src, keys, train, 2019, "h")
    assert holey.mean[0] != clean.mean[0] and np.isfinite(holey.mean[0])
    z = holey.apply(Holey().load(keys[0]).x, src.load(keys[0]).valid)
    assert np.isfinite(z).all() and (z[:, 0, :3, :3] == 0).all()


# ----------------------------------------------------------------------------- loss


def test_loss_ignores_padding_and_matches_role2_when_installed():
    logits = torch.randn(1, 1, 6, 6)
    y = (torch.rand(1, 1, 6, 6) > 0.8).float()
    m = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    m[..., -2:, :] = False
    y2 = y.clone()
    y2[..., -2:, :] = 1 - y2[..., -2:, :]
    assert torch.allclose(
        _role3_masked_bce_dice(logits, y, m), _role3_masked_bce_dice(logits, y2, m)
    )
    try:
        from fireguard_ml.training.losses import masked_bce_dice_loss
    except ImportError:
        pytest.skip("Role 2's losses module not installed yet")
    assert torch.allclose(masked_bce_dice_loss(logits, y, m), _role3_masked_bce_dice(logits, y, m))


# ----------------------------------------------------------------------------- training


class RecordingSource:
    """Wraps a source and records every key loaded, to prove the test split is never opened."""

    def __init__(self, inner):
        self.inner, self.loaded = inner, set()
        self.channels, self.sequence_length, self.is_mock = (
            inner.channels,
            inner.sequence_length,
            inner.is_mock,
        )

    def keys(self):
        return self.inner.keys()

    def load(self, key):
        self.loaded.add(key[0])
        return self.inner.load(key)


@pytest.mark.parametrize("model", ["m3_stacked_unet", "m4_convlstm_unet"])
def test_train_fold_end_to_end_never_touches_test_events(tmp_path, channels, register_path, model):
    src, proto, rows = make_world(tmp_path, channels, register_path)
    rec = RecordingSource(src)
    summary = train_fold(tiny(model, channels), rec, proto, 2020, tmp_path / "out", device="cpu")
    test_ev = {r.event_id for r in rows if r.fold_test_year == 2020 and r.split == "test"}
    assert not rec.loaded & test_ev
    fdir = tmp_path / "out" / "fold_2020"
    run = summary["run"]
    model_, meta = load_checkpoint(fdir / f"{run}.pt")
    assert meta["fold"] == 2020 and meta["is_mock"] is True and model_.strict_checkpoint
    assert meta["split_manifest_hash"] == sha256_file(proto.manifest)
    sel = json.loads((fdir / f"{run}.selection.json").read_text())
    assert sel["fitted_on"] == "validation" and sel["fold"] == 2020
    log = [json.loads(x) for x in (fdir / f"{run}.log.jsonl").read_text().splitlines()]
    assert 1 <= len(log) <= 2 and all("val_auprc" in r for r in log)
    assert summary["checkpoint_sha256"] and summary["parameters"] > 0


def test_event_blocked_order_is_a_seeded_permutation_grouped_by_event():
    import random

    from fireguard_ml.training.temporal_trainer import event_blocked_order

    keys = [(f"e{i % 4}", date(2020, 1, 1) + timedelta(days=i)) for i in range(20)]
    o1 = event_blocked_order(keys, random.Random(3))
    assert sorted(o1) == sorted(keys) and o1 == event_blocked_order(keys, random.Random(3))
    runs = [k[0] for i, k in enumerate(o1) if i == 0 or o1[i - 1][0] != k[0]]
    assert len(runs) == 4  # each event appears as one contiguous block


def test_training_refuses_provisional_splits(tmp_path, channels, register_path):
    src, proto, _ = make_world(tmp_path, channels, register_path, status="PROVISIONAL")
    with pytest.raises(RuntimeError, match="FROZEN"):
        train_fold(tiny("m3_stacked_unet", channels), src, proto, 2020, tmp_path / "o", "cpu")


def test_training_refuses_a_tampered_manifest(tmp_path, channels, register_path):
    src, proto, _ = make_world(tmp_path, channels, register_path)
    proto.manifest.write_text(proto.manifest.read_text().replace(",train,", ",validation,", 1))
    with pytest.raises(RuntimeError, match="hash"):
        train_fold(tiny("m3_stacked_unet", channels), src, proto, 2020, tmp_path / "o", "cpu")


def test_training_refuses_an_excluded_channel(tmp_path, channels, register_path):
    leaky = tuple(c if c != "elevation" else "ndvi" for c in channels)
    src, proto, _ = make_world(tmp_path, leaky, register_path)
    with pytest.raises(ValueError, match="not allowed"):
        train_fold(tiny("m3_stacked_unet", leaky), src, proto, 2020, tmp_path / "o", "cpu")


# ----------------------------------------------------------------------------- OOF export


def test_oof_export_is_fold_safe_and_pre_reveal_clean(tmp_path, channels, register_path):
    rasterio = pytest.importorskip("rasterio")
    from fireguard_ml.inference.temporal_oof import (
        FORBIDDEN_PRE_REVEAL,
        PRE_REVEAL_FIELDS,
        OOFMappingError,
        export_oof,
    )

    src, proto, rows = make_world(tmp_path, channels, register_path)
    out = tmp_path / "runs_out"
    ckpts = {}
    for fold in (2019, 2021):
        s = train_fold(tiny("m3_stacked_unet", channels), src, proto, fold, out, "cpu")
        ckpts[fold] = out / f"fold_{fold}" / s["checkpoint"]

    with pytest.raises(OOFMappingError, match="fold"):
        export_oof(out, {2019: ckpts[2021]}, src, proto, tmp_path / "art", "bad")

    res = export_oof(out, ckpts, src, proto, tmp_path / "art", "mock_m3")
    run_dir = tmp_path / "art" / "runs" / "mock_m3"
    recs = [json.loads(x) for x in (run_dir / "oof_forecasts.jsonl").read_text().splitlines()]
    assert res["forecasts"] == len(recs) > 0
    manifest = read_manifest(proto.manifest)
    for r in recs:
        assert tuple(r) == PRE_REVEAL_FIELDS
        assert not any(f in r for f in FORBIDDEN_PRE_REVEAL)
        assert r["is_oof"] and r["split_name"] == "test" and r["is_mock"]
        row = next(
            m
            for m in manifest
            if m.fold_test_year == int(r["fold"]) and m.event_id == r["event_id"]
        )
        assert row.split == "test"  # mechanically held out by the fold that predicted it
        with rasterio.open(run_dir / Path(r["storage_uri"]).relative_to("runs/mock_m3")) as ds:
            assert (ds.height, ds.width) == (r["height"], r["width"])
            assert str(ds.crs) == r["crs"]
            p = ds.read(1)
            assert np.nanmin(p) >= 0 and np.nanmax(p) <= 1
    with (run_dir / "post_reveal" / "per_event_test_scores.csv").open() as f:
        scores = list(csv.DictReader(f))
    assert {s["event_id"] for s in scores} == {r["event_id"] for r in recs}
    for fold in (2019, 2021):
        z = np.load(run_dir / "post_reveal" / f"pooled_counts_fold_{fold}.npz")
        n_fold = sum(int(s["n_valid"]) for s in scores if int(s["fold_test_year"]) == fold)
        assert int(z["pos"].sum() + z["neg"].sum()) == n_fold


# ----------------------------------------------------------------------------- config


def test_temporal_config_lists_exactly_the_registers_strict_channels(repo_root, channels):
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((repo_root / "configs" / "temporal_unet.yaml").read_text())
    assert tuple(cfg["data"]["channels"]) == channels
    assert cfg["ablation"]["context_lengths"][0] == cfg["data"]["n_leading_observations"]
