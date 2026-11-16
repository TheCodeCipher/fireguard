"""M1 (spatial U-Net, T=1) added by Integration for Gate 2 (Prototype, 2026-11-13).

Checks that M1 is a one-day model, that it round-trips through the shared checkpoint code (so
the OOF export can rebuild it), and that it differs from M3 only in T.
"""

from __future__ import annotations

import pytest
import torch

from fireguard_ml.models.architectures import MODEL_REGISTRY, SpatialUNet, StackedDaysUNet
from fireguard_ml.models.checkpointing import load_checkpoint, save_checkpoint


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


# ---------------------------------------------------------------------------- M2


def test_m2_is_registered_one_day_and_close_to_m1_in_size():
    from fireguard_ml.models.architectures import AttentionResUNet

    m2 = MODEL_REGISTRY["m2_attention_res_unet"](c_strict=7, T=1, base=8, depth=3)
    assert isinstance(m2, AttentionResUNet)
    out = m2(torch.randn(2, 1, 7, 64, 32))
    assert out.shape == (2, 1, 64, 32)
    with pytest.raises(ValueError, match="T=1"):
        AttentionResUNet(c_strict=7, T=5)
    n1 = sum(p.numel() for p in SpatialUNet(7).parameters())
    n2 = sum(p.numel() for p in AttentionResUNet(7).parameters())
    assert 1.0 < n2 / n1 < 1.15  # same widths and depth; residual + gate weights add ~6 %


def test_m2_attention_gates_weight_every_skip():
    from fireguard_ml.models.architectures import AttentionGate, AttentionResUNet

    m2 = AttentionResUNet(c_strict=7, base=8, depth=3)
    gates = [g for g in m2.dec.gates if isinstance(g, AttentionGate)]
    assert len(gates) == 3
    g = gates[0]
    w = g.psi(g.wg(torch.randn(1, 32, 8, 8)) + g.wx(torch.randn(1, 32, 8, 8)))
    assert w.shape == (1, 1, 8, 8) and float(w.min()) >= 0 and float(w.max()) <= 1


@pytest.mark.parametrize("model_type,T", [("m1_spatial_unet", 1), ("m2_attention_res_unet", 1),
                                          ("m3_stacked_unet", 3), ("m4_convlstm_unet", 3)])
def test_every_model_round_trips_with_dropout(tmp_path, model_type, T):
    from fireguard_ml.models.checkpointing import load_checkpoint, save_checkpoint

    m = MODEL_REGISTRY[model_type](c_strict=7, T=T, base=8, depth=3, dropout=0.2).eval()
    meta = {"channels": [f"c{i}" for i in range(7)], "T": T, "preprocessing_version": "v",
            "config_hash": "h", "fold": 2018, "split_manifest_hash": "s"}
    save_checkpoint(m, tmp_path / "m.pt", meta)
    m2, _ = load_checkpoint(tmp_path / "m.pt")
    assert m2.dropout == 0.2
    x = torch.randn(1, T, 7, 32, 32)
    assert torch.allclose(m.predict(x), m2.predict(x))
