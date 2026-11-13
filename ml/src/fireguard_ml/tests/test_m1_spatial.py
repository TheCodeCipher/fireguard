"""M1 (spatial U-Net, T=1) added by Integration for Gate 2 (Prototype, 2026-11-13).

Checks that M1 is a one-day model, that it round-trips through the shared checkpoint code (so
the OOF export can rebuild it), and that it differs from M3 only in T.
"""

from __future__ import annotations

import pytest
import torch

from fireguard_ml.models.checkpointing import load_checkpoint, save_checkpoint
from fireguard_ml.models.temporal_unet import MODEL_REGISTRY, SpatialUNet, StackedDaysUNet


def test_m1_is_registered_and_one_day():
    assert MODEL_REGISTRY["m1_spatial_unet"] is SpatialUNet
    m = SpatialUNet(c_strict=7, base=8, depth=5)
    y = m(torch.zeros(1, 1, 7, 64, 64))
    assert y.shape == (1, 1, 64, 64)
    with pytest.raises(ValueError):
        SpatialUNet(c_strict=7, T=5)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 5, 7, 64, 64))  # five days into a one-day model is refused


def test_m1_has_same_architecture_as_m3_at_t1():
    m1 = SpatialUNet(c_strict=7, base=8, depth=5)
    m3 = StackedDaysUNet(c_strict=7, T=1, base=8, depth=5)
    assert [p.shape for p in m1.parameters()] == [p.shape for p in m3.parameters()]


def test_m1_checkpoint_round_trip(tmp_path):
    channels = [
        "viirs_m11", "viirs_i2", "viirs_i1", "slope", "aspect", "elevation",
        "binary_active_fire_mask",
    ]  # fmt: skip
    m = SpatialUNet(c_strict=7, base=8, depth=5)
    path = tmp_path / "m1.pt"
    save_checkpoint(
        m,
        path,
        {
            "channels": channels,
            "T": 1,
            "fold": 2020,
            "preprocessing_version": "test",
            "config_hash": "x",
            "split_manifest_hash": "y",
        },
    )
    back, meta = load_checkpoint(path)
    assert type(back) is SpatialUNet and back.T == 1
    assert meta["fold"] == 2020
