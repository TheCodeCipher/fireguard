"""Live prediction equals the OOF export's prediction; the published layer is the fold mean."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
import torch

pytest.importorskip("rasterio")

from live_test_helpers import (  # noqa: E402
    CHANNELS,
    RUN_ID,
    ArrayTerrain,
    RawFrameFire,
    RawFrameReflectance,
    make_repo_root,
    make_trained_run,
    make_wsts_event,
)

from fireguard_ml.data.wsts_dataset import WSTSTemporalSource  # noqa: E402
from fireguard_ml.live.forecast import predict_fold, predict_sample  # noqa: E402
from fireguard_ml.live.frames import LiveTemporalSource  # noqa: E402
from fireguard_ml.live.readiness import check_live_readiness  # noqa: E402
from fireguard_ml.training.temporal_data import pad_samples  # noqa: E402


def _bundles(tmp_path, T=1):
    run_dir, run, artifact_root = make_trained_run(tmp_path, (2020, 2021), T=T)
    _, bundles = check_live_readiness(run_dir=run_dir, run=run, folds=(2020, 2021),
                                      run_id=RUN_ID, artifact_root=artifact_root,
                                      repo_root=make_repo_root(tmp_path))
    return bundles


def _export_path_probability(sample, bundle):
    """The three lines of inference/temporal_oof.export_oof, verbatim."""
    model, norm, s = bundle.model, bundle.norm, sample
    b = pad_samples([replace(s, x=norm.apply(s.x, s.valid))], 2**model.depth)
    prob = model.predict(b.x.to(torch.device("cpu")), orig_hw=s.hw)[0, 0].cpu().numpy()
    return np.where(s.valid, prob, np.nan).astype(np.float32)


@pytest.mark.parametrize("T", [1, 3])
def test_live_prediction_equals_export_prediction_on_the_same_raw_data(tmp_path, T):
    region, samples, frames = make_wsts_event(tmp_path / "wsts", n_days=6)
    bundles = _bundles(tmp_path, T)
    wsts = WSTSTemporalSource(tmp_path / "wsts", CHANNELS, T, samples)
    event, target = [k for k in wsts.keys() if (k[1] - min(frames)).days >= T][-1]
    origin = target - timedelta(days=1)
    live = LiveTemporalSource([region], origin, CHANNELS, T, RawFrameFire(frames),
                              RawFrameReflectance(frames), ArrayTerrain(frames[origin][12:15]))
    wsts_sample, live_sample = wsts.load((event, target)), live.load((event, target))
    per_fold = []
    for b in bundles:
        expected = _export_path_probability(wsts_sample, b)  # training loader -> export code
        got = predict_fold(live_sample, b, torch.device("cpu"))  # live builder -> live code
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)
        assert got.shape == region.hw and got.dtype == np.float32
        assert np.isfinite(got).all() and got.min() >= 0.0 and got.max() <= 1.0
        per_fold.append(expected)
    assert not np.allclose(per_fold[0], per_fold[1])  # the folds are different models

    out = predict_sample(live_sample, bundles)
    np.testing.assert_allclose(out["mean"], np.mean(per_fold, axis=0), atol=1e-6)
    np.testing.assert_allclose(out["spread"], np.abs(per_fold[0] - per_fold[1]), atol=1e-6)
    assert set(out["per_fold"]) == {2020, 2021}
    assert out["mean"].dtype == np.float32 and (out["spread"] >= 0).all()


def test_invalid_cells_are_nan_and_only_those(tmp_path):
    region, samples, frames = make_wsts_event(tmp_path / "wsts", n_days=3)
    bundles = _bundles(tmp_path)
    origin = max(frames) - timedelta(days=1)
    live = LiveTemporalSource([region], origin, CHANNELS, 1, RawFrameFire(frames),
                              RawFrameReflectance(frames), ArrayTerrain(frames[origin][12:15]))
    s = live.load(live.keys()[0])
    valid = s.valid.copy()
    valid[:4, :7] = False
    out = predict_sample(replace(s, valid=valid), bundles)
    assert np.isnan(out["mean"][:4, :7]).all() and np.isfinite(out["mean"][valid]).all()
    assert np.isnan(out["mean"]).sum() == 28


def test_calibration_is_each_folds_own_platt_map(tmp_path):
    region, samples, frames = make_wsts_event(tmp_path / "wsts", n_days=3)
    bundles = _bundles(tmp_path)
    origin = max(frames) - timedelta(days=1)
    live = LiveTemporalSource([region], origin, CHANNELS, 1, RawFrameFire(frames),
                              RawFrameReflectance(frames), ArrayTerrain(frames[origin][12:15]))
    s = live.load(live.keys()[0])
    b = bundles[0]
    raw = predict_fold(s, b, torch.device("cpu")).astype(np.float64)
    cal = predict_fold(s, b, torch.device("cpu"), calibrated=True)
    a, off = b.selection["calibrator"]["a"], b.selection["calibrator"]["b"]
    p = np.clip(raw, 1e-12, 1 - 1e-12)
    expected = 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + off)))
    np.testing.assert_allclose(cal, expected, atol=1e-5)


def test_no_fold_models_is_an_error(tmp_path):
    region, samples, frames = make_wsts_event(tmp_path / "wsts", n_days=3)
    origin = max(frames) - timedelta(days=1)
    live = LiveTemporalSource([region], origin, CHANNELS, 1, RawFrameFire(frames),
                              RawFrameReflectance(frames), ArrayTerrain(frames[origin][12:15]))
    with pytest.raises(ValueError):
        predict_sample(live.load(live.keys()[0]), [])
