from pathlib import Path

import pytest
import torch

from fireguard.models.checkpointing import REQUIRED_META, load_checkpoint, save_checkpoint
from fireguard.models.temporal_unet import ConvLSTMUNet, StackedDaysUNet

T, C, BASE, DEPTH = 5, 6, 8, 5
CHANNELS = [f"ch{i}" for i in range(C)]
META = dict(channels=CHANNELS, T=T, preprocessing_version="v0", config_hash="abc", fold=2019,
            split_manifest_hash="def")


def make(cls):
    torch.manual_seed(0)
    return cls(C, T, BASE, DEPTH).eval()


@pytest.mark.parametrize("cls", [StackedDaysUNet, ConvLSTMUNet])
def test_output_shape_matches_padded_input(cls):
    x = torch.randn(1, T, C, 64, 96)
    assert make(cls)(x).shape == (1, 1, 64, 96)


@pytest.mark.parametrize("cls", [StackedDaysUNet, ConvLSTMUNet])
def test_representative_frame_size(cls):
    x = torch.randn(1, T, C, 320, 352)  # padded from about 308x356 / 304x207 style frames
    assert make(cls)(x).shape == (1, 1, 320, 352)


@pytest.mark.parametrize("cls", [StackedDaysUNet, ConvLSTMUNet])
def test_rejects_unpadded_or_wrong_channels(cls):
    m = make(cls)
    with pytest.raises(ValueError):
        m(torch.randn(1, T, C, 70, 64))
    with pytest.raises(ValueError):
        m(torch.randn(1, T, C + 1, 64, 64))
    with pytest.raises(ValueError):
        m(torch.randn(1, T - 1, C, 64, 64))


def test_m3_input_channels_are_T_times_C():
    assert make(StackedDaysUNet).enc.stem[0].in_channels == T * C


def test_m4_encoder_is_shared_across_days():
    assert make(ConvLSTMUNet).enc.stem[0].in_channels == C  # one encoder, per-frame input


def test_m4_is_sensitive_to_frame_order():
    m = make(ConvLSTMUNet)
    x = torch.randn(1, T, C, 64, 64)
    assert not torch.allclose(m(x), m(x.flip(1)), atol=1e-6)


@pytest.mark.parametrize("cls", [StackedDaysUNet, ConvLSTMUNet])
def test_predict_range_unpad_and_mode_restored(cls):
    m = cls(C, T, BASE, DEPTH).train()
    p = m.predict(torch.randn(1, T, C, 64, 96), orig_hw=(50, 90))
    assert p.shape == (1, 1, 50, 90)
    assert float(p.min()) >= 0 and float(p.max()) <= 1
    assert not p.requires_grad
    assert m.training


@pytest.mark.parametrize("cls", [StackedDaysUNet, ConvLSTMUNet])
def test_checkpoint_roundtrip_gives_identical_predictions(cls, tmp_path):
    m = make(cls)
    x = torch.randn(1, T, C, 64, 64)
    before = m.predict(x)
    path = tmp_path / "m.pt"
    save_checkpoint(m, path, META)
    m2, meta = load_checkpoint(path)
    assert type(m2) is cls and meta["fold"] == 2019
    assert torch.equal(before, m2.predict(x))


@pytest.mark.parametrize("bad_name", ["gfs_wind_speed", "GFS_wind_direction", "forecast wind speed",
                                      "Forecast total precipitation"])
def test_checkpoint_rejects_forecast_channels(tmp_path, bad_name):
    m = make(StackedDaysUNet)
    with pytest.raises(ValueError):
        save_checkpoint(m, tmp_path / "a.pt", {**META, "channels": CHANNELS[:-1] + [bad_name]})


def test_non_strict_checkpoint_is_labelled(tmp_path):
    # the benchmark-assumption branch may store forecast channels, but only with strict=False, and the flag is saved
    import torch as _t
    m = make(StackedDaysUNet)
    path = tmp_path / "bench.pt"
    save_checkpoint(m, path, {**META, "channels": CHANNELS[:-1] + ["gfs_wind_speed"]}, strict=False)
    assert _t.load(path, weights_only=True)["strict"] is False


def test_checkpoint_rejects_missing_meta(tmp_path):
    m = make(StackedDaysUNet)
    for key in REQUIRED_META:
        with pytest.raises(ValueError):
            save_checkpoint(m, tmp_path / "b.pt", {k: v for k, v in META.items() if k != key})


def test_checkpoint_checks_channel_count_and_T(tmp_path):
    m = make(StackedDaysUNet)
    with pytest.raises(ValueError):
        save_checkpoint(m, tmp_path / "c.pt", {**META, "channels": CHANNELS[:-1]})
    with pytest.raises(ValueError):
        save_checkpoint(m, tmp_path / "d.pt", {**META, "T": T + 1})


CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "temporal_unet.yaml"


def test_strict_config_has_no_gfs():
    lines = [ln.split("#")[0].lower() for ln in CFG_PATH.read_text(encoding="utf8").splitlines()]
    assert not [ln for ln in lines if "gfs" in ln or "forecast" in ln]


def test_strict_config_is_consistent_with_the_models():
    yaml = pytest.importorskip("yaml")
    from fireguard.models.temporal_unet import MODEL_REGISTRY

    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf8"))
    assert cfg["strict"] is True
    assert cfg["data"]["geometric_augmentation"] is False
    assert cfg["data"]["frame_order"] == "oldest_to_newest"
    # processed loader indices of the five forecast channels (docs/wsts_data_dictionary.md)
    assert cfg["data"]["excluded_processed_indices"] == [33, 34, 35, 36, 37]
    assert cfg["training"]["enabled"] is False  # Phase 2 delivers interfaces, not training runs
    t = cfg["data"]["n_leading_observations"]
    for m in cfg["models"].values():
        kwargs = {k: v for k, v in m.items() if k != "model_type"}
        assert 2 ** kwargs["depth"] == cfg["data"]["pad_multiple"]
        model = MODEL_REGISTRY[m["model_type"]](c_strict=C, T=t, **kwargs)
        assert model(torch.randn(1, t, C, 32, 64)).shape == (1, 1, 32, 64)
