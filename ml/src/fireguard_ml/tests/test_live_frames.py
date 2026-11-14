"""Raw live frames become strict channels through the training loader's own code (layer 1)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

pytest.importorskip("rasterio")

from live_test_helpers import (  # noqa: E402
    CHANNELS,
    ORIGIN,
    ArrayFire,
    ArrayReflectance,
    ArrayTerrain,
    fire_band,
    small_region,
)

from fireguard_ml.data.wsts_dataset import RAW_BANDS  # noqa: E402
from fireguard_ml.live.frames import (  # noqa: E402
    LIVE_RAW_BANDS,
    LiveInputError,
    LiveTemporalSource,
    build_raw_frame,
    strict_from_raw,
)
from fireguard_ml.live.reflectance import (  # noqa: E402
    LinearReflectanceMapping,
    MissingReflectanceProvider,
    ReflectanceUnavailable,
    fit_linear_mapping,
)
from fireguard_ml.live.terrain import (  # noqa: E402
    CachedTerrain,
    ConstantTerrain,
    TerrainMosaic,
    TerrainUnavailable,
    write_terrain_tif,
)
from fireguard_ml.training.temporal_data import (  # noqa: E402
    TemporalSampleSource,
    validate_sample,
)


def _source(T=1, region=None, fire=None, refl=None, terrain=None, **kw):
    region = region or small_region()
    return LiveTemporalSource([region], ORIGIN, CHANNELS, T, fire or ArrayFire({}),
                              refl or ArrayReflectance(), terrain or ArrayTerrain(), **kw)


def test_raw_frame_fills_exactly_the_seven_live_bands():
    r = small_region()
    raw = build_raw_frame(r, fire_band(r.hw, {(3, 3): 2030}), ArrayReflectance().fetch(r, ORIGIN),
                          ArrayTerrain().fetch(r))
    assert raw.shape == (23, *r.hw) and raw.dtype == np.float32
    assert LIVE_RAW_BANDS == (0, 1, 2, 12, 13, 14, 22)
    for b in range(len(RAW_BANDS)):
        if b in (0, 1, 2, 12, 13, 14):
            assert np.isfinite(raw[b]).all(), RAW_BANDS[b]
        elif b == 22:
            assert np.isfinite(raw[b]).sum() == 1
        else:
            assert np.isnan(raw[b]).all(), RAW_BANDS[b]


def test_strict_channels_follow_the_config_order_and_training_transforms():
    r = small_region()
    terrain = np.stack([np.full(r.hw, 12.0), np.full(r.hw, 90.0), np.full(r.hw, 1500.0)]).astype(
        np.float32)
    refl = np.stack([np.full(r.hw, v) for v in (1111.0, 2222.0, 3333.0)]).astype(np.float32)
    raw = build_raw_frame(r, fire_band(r.hw, {(3, 3): 2030, (4, 4): 30}), refl, terrain)
    x = strict_from_raw(raw, CHANNELS)
    assert CHANNELS == ("viirs_m11", "viirs_i2", "viirs_i1", "slope", "aspect", "elevation",
                        "binary_active_fire_mask")
    assert x.shape == (7, *r.hw) and x.dtype == np.float32
    assert (x[0] == 1111.0).all() and (x[1] == 2222.0).all() and (x[2] == 3333.0).all()
    assert (x[3] == 12.0).all() and (x[5] == 1500.0).all()
    np.testing.assert_allclose(x[4], 1.0, atol=1e-6)  # sin(90 deg), applied exactly once
    assert x[6, 3, 3] == 1.0 and x[6, 4, 4] == 0.0 and x[6].sum() == 1.0


def test_source_satisfies_the_temporal_sample_contract():
    for T in (1, 3):
        src = _source(T=T, fire=ArrayFire({ORIGIN: fire_band((40, 52), {(5, 5): 2030})}))
        assert isinstance(src, TemporalSampleSource)
        assert src.is_mock is False and src.sequence_length == T and src.channels == CHANNELS
        (key,) = src.keys()
        assert key == ("r_test", ORIGIN + timedelta(days=1))
        s = src.load(key)
        validate_sample(s, T, CHANNELS)
        assert s.input_dates == tuple(ORIGIN - timedelta(days=T - 1 - i) for i in range(T))
        assert s.origin_date == ORIGIN and s.target_date == ORIGIN + timedelta(days=1)
        assert s.x.shape == (T, 7, 40, 52) and s.valid.all() and not s.y.any()
        assert s.transform == small_region().transform and s.crs == "EPSG:32610"
        assert s.x[-1, 6, 5, 5] == 1.0
    with pytest.raises(KeyError):
        src.load(("r_test", ORIGIN))
    with pytest.raises(KeyError):
        src.load(("r_other", ORIGIN + timedelta(days=1)))


def test_missing_reflectance_is_nan_reported_and_falls_back():
    class Unavailable:
        name = "down"

        def fetch(self, region, day):
            raise ReflectanceUnavailable("no granule")

    for provider in (MissingReflectanceProvider(), Unavailable()):
        src = _source(refl=provider)
        s = src.load(src.keys()[0])
        assert np.isnan(s.x[0, :3]).all() and np.isfinite(s.x[0, 3:]).all()
        prov = src.provenance("r_test")
        assert prov["reflectance_source"] == "missing"
        assert [prov["nan_fraction"][c] for c in CHANNELS] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        assert prov["input_dates"] == [ORIGIN.isoformat()]


def test_bad_inputs_are_refused():
    r = small_region()
    good_r, good_t = ArrayReflectance().fetch(r, ORIGIN), ArrayTerrain().fetch(r)
    with pytest.raises(LiveInputError):
        build_raw_frame(r, np.zeros((3, 3), np.float32), good_r, good_t)
    with pytest.raises(LiveInputError):
        build_raw_frame(r, fire_band(r.hw, {}), good_r[:2], good_t)
    holed = good_t.copy()
    holed[2, 0, 0] = np.nan  # one masked cell (open water) is allowed, as in WSTS frames
    assert np.isnan(build_raw_frame(r, fire_band(r.hw, {}), good_r, holed)[14, 0, 0])
    holed[2] = np.nan  # a band with no data at all is a failed fetch
    with pytest.raises(LiveInputError, match="terrain"):
        build_raw_frame(r, fire_band(r.hw, {}), good_r, holed)
    with pytest.raises(LiveInputError, match="no live source"):
        LiveTemporalSource([r], ORIGIN, (*CHANNELS, "ndvi"), 1, ArrayFire({}), ArrayReflectance(),
                           ArrayTerrain())
    with pytest.raises(LiveInputError, match="duplicate"):
        LiveTemporalSource([r, r], ORIGIN, CHANNELS, 1, ArrayFire({}), ArrayReflectance(),
                           ArrayTerrain())


def test_terrain_mosaic_and_cache(tmp_path):
    r = small_region()
    big = small_region(h=60, w=80, rid="mosaic")
    arr = ArrayTerrain(seed=3).fetch(big)
    write_terrain_tif(tmp_path / "mosaic.tif", arr, big)
    got = TerrainMosaic(tmp_path / "mosaic.tif").fetch(r)
    np.testing.assert_array_equal(got, arr[:, :40, :52])
    far = small_region(h=40, w=52)
    far = type(far)("far", 0, 0, (375.0, 0.0, 697500.0 + 375 * 70, 0.0, -375.0, 4136625.0), 40, 52)
    with pytest.raises(TerrainUnavailable, match="cover"):
        TerrainMosaic(tmp_path / "mosaic.tif").fetch(far)
    off = type(far)("off", 0, 0, (375.0, 0.0, 697500.0 + 100, 0.0, -375.0, 4136625.0), 40, 52)
    with pytest.raises(TerrainUnavailable, match="aligned"):
        TerrainMosaic(tmp_path / "mosaic.tif").fetch(off)
    with pytest.raises(TerrainUnavailable):
        TerrainMosaic(tmp_path / "absent.tif")

    class Counting(ArrayTerrain):
        calls = 0

        def fetch(self, region):
            Counting.calls += 1
            return super().fetch(region)

    cached = CachedTerrain(Counting(), tmp_path / "cache")
    a, b = cached.fetch(r), cached.fetch(r)
    np.testing.assert_array_equal(a, b)
    assert Counting.calls == 1 and (tmp_path / "cache" / "r_test.tif").is_file()
    assert ConstantTerrain().fetch(r).shape == (3, 40, 52)


def test_linear_reflectance_mapping_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 6000, size=(4, 3, 20, 20)).astype(np.float32)
    y = x * np.array([1.02, 0.97, 1.1], np.float32)[None, :, None, None] + np.array(
        [15.0, -30.0, 5.0], np.float32)[None, :, None, None]
    x[0, :, :3, :3] = np.nan
    m = fit_linear_mapping(x, y, "unit-test")
    np.testing.assert_allclose(m.scale, (1.02, 0.97, 1.1), rtol=1e-3)
    np.testing.assert_allclose(m.offset, (15.0, -30.0, 5.0), atol=0.5)
    m.to_json(tmp_path / "m.json")
    back = LinearReflectanceMapping.from_json(tmp_path / "m.json")
    assert back == m
    np.testing.assert_allclose(back.apply(x[1]), y[1], rtol=1e-3, atol=1.0)
    assert LinearReflectanceMapping().apply(x[1]).tolist() == x[1].tolist()
