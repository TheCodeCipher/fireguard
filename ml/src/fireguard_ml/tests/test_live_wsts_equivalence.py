"""The live path produces, from the same raw data, exactly what the training loader produces.

This is the central guarantee of live forecasting: a frame assembled by ``LiveTemporalSource`` is
bit-identical to the frame ``WSTSTemporalSource`` builds from a WSTS GeoTIFF holding the same raw
values, so the model sees the same transforms live as in training (layer 1).
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

from live_test_helpers import (  # noqa: E402
    CHANNELS,
    REPO,
    ArrayTerrain,
    RawFrameFire,
    RawFrameReflectance,
    make_wsts_event,
)

from fireguard_ml.data.wsts_dataset import WSTSTemporalSource, _read  # noqa: E402
from fireguard_ml.live.frames import LiveTemporalSource  # noqa: E402
from fireguard_ml.live.grid import region_from_grid  # noqa: E402


@pytest.mark.parametrize("T", [1, 3, 5])
def test_live_frames_equal_the_training_loader_on_wsts_format_files(tmp_path, T):
    region, samples, frames = make_wsts_event(tmp_path, n_days=8)
    wsts = WSTSTemporalSource(tmp_path, CHANNELS, T, samples)
    keys = [k for k in wsts.keys() if (k[1] - min(frames)).days >= T]
    assert keys
    for event, target in keys:
        expected = wsts.load((event, target))
        origin = target - timedelta(days=1)
        live = LiveTemporalSource(
            [region], origin, CHANNELS, T, RawFrameFire(frames), RawFrameReflectance(frames),
            ArrayTerrain(frames[origin][12:15]))
        got = live.load((region.region_id, target))
        assert got.x.dtype == expected.x.dtype == np.float32
        assert got.x.tobytes() == expected.x.tobytes()  # bit-identical, NaNs included
        assert got.input_dates == expected.input_dates
        assert got.target_date == expected.target_date
        assert got.channels == expected.channels
        assert got.transform == expected.transform and got.crs == expected.crs
        np.testing.assert_array_equal(got.valid, expected.valid)
        assert np.isnan(got.x[:, :3]).any()  # the cloud gap survived as NaN in both
        assert got.x[-1, 6, 20, 20 + (origin - min(frames)).days] == 0.0  # 00:30 UTC quirk


@pytest.mark.skipif(not os.environ.get("FIREGUARD_WSTS_ROOT"), reason="set FIREGUARD_WSTS_ROOT")
def test_live_frames_equal_the_training_loader_on_real_wsts_events():
    root = Path(os.environ["FIREGUARD_WSTS_ROOT"])
    samples = REPO / "data" / "manifests" / "wsts_eval_samples.csv"
    wsts = WSTSTemporalSource(root, CHANNELS, 1, samples)
    seen: set[str] = set()
    for event, target in wsts.keys():
        if event in seen:
            continue
        seen.add(event)
        expected = wsts.load((event, target))
        origin = target - timedelta(days=1)
        year_dir = next(p for p in root.iterdir() if (p / event).is_dir())
        raw, transform, crs = _read(year_dir / event / f"{origin.isoformat()}.tif",
                                    list(range(1, 24)))
        region = region_from_grid(event, transform, raw.shape[1], raw.shape[2], crs)
        frames = {origin: raw}
        live = LiveTemporalSource([region], origin, CHANNELS, 1, RawFrameFire(frames),
                                  RawFrameReflectance(frames), ArrayTerrain(raw[12:15]))
        got = live.load((event, target))
        assert got.x.tobytes() == expected.x.tobytes(), event
        if len(seen) == 5:
            break
    assert len(seen) == 5
