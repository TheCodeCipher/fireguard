import pytest
import torch

from fireguard_ml.data.padding import make_input_valid_mask, pad_to_multiple, pad_with_valid_mask, unpad
from fireguard_ml.data.channel_schema import PROCESSED_WSTS_CHANNELS, select_channels
from fireguard_ml.data.sample_adapter import adapt_sample, restore_frame
from fireguard_ml.models import UNet
from fireguard_ml.training.losses import masked_bce_dice_loss


def test_pad_to_32_and_exact_unpad():
    x = torch.randn(2, 3, 305, 207)
    padded, info = pad_to_multiple(x, 32)
    assert padded.shape[-2:] == (320, 224)
    assert torch.equal(unpad(padded, info), x)


def test_padding_is_invalid_in_mask():
    x = torch.ones(1, 2, 31, 33)
    mask = torch.ones(1, 1, 31, 33, dtype=torch.bool)
    xp, mp, info = pad_with_valid_mask(x, mask)
    assert xp.shape[-2:] == (32, 64)
    assert mp.sum().item() == 31 * 33
    assert not mp[..., 31:, :].any()
    assert not mp[..., :, 33:].any()
    assert info.original_height == 31 and info.original_width == 33


def test_unet_logits_match_padded_target():
    model = UNet(in_channels=3, base_channels=4)
    x, info = pad_to_multiple(torch.randn(2, 3, 65, 97))
    logits = model(x)
    assert logits.shape == (2, 1, info.padded_height, info.padded_width)


def test_masked_loss_ignores_invalid_target_changes():
    logits = torch.zeros(1, 1, 4, 4)
    targets = torch.zeros_like(logits)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[..., -1, -1] = False
    a = masked_bce_dice_loss(logits, targets, mask)
    changed = targets.clone()
    changed[..., -1, -1] = 1.0
    b = masked_bce_dice_loss(logits, changed, mask)
    assert torch.allclose(a, b)


def test_masked_loss_rejects_empty_mask():
    x = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match="no valid pixels"):
        masked_bce_dice_loss(x, x, torch.zeros_like(x, dtype=torch.bool))


def test_documented_40_channel_sample_adapter_pads_without_cropping():
    x = torch.randn(1, 40, 304, 247)
    y = torch.zeros(1, 304, 247, dtype=torch.long)
    y[..., 12, 20] = 1
    sample = adapt_sample(x, y)
    assert len(PROCESSED_WSTS_CHANNELS) == 40
    assert sample.inputs.shape == (1, 40, 320, 256)
    assert sample.target.shape == (1, 1, 320, 256)
    assert sample.valid_mask.shape == (1, 1, 320, 256)
    assert sample.valid_mask.sum().item() == 304 * 247
    assert restore_frame(sample.target, sample).shape == (1, 1, 304, 247)
    assert sample.channel_selection.selection_status == "BASELINE_SCHEMA_EXPLORATORY_ONLY"


def test_channel_list_can_be_swapped_by_name():
    x = torch.arange(40.0).reshape(1, 40, 1, 1)
    selected, metadata = select_channels(x, selected_names=("active_fire_binary", "elevation"))
    assert selected[:, 0].item() == 39
    assert selected[:, 1].item() == 14
    assert metadata.names == ("active_fire_binary", "elevation")


def test_strict_mode_requires_explicit_subset():
    x = torch.zeros(1, 40, 2, 2)
    with pytest.raises(ValueError, match="cannot be assumed strict-approved"):
        select_channels(x, strict_approved=True)


def test_nonfinite_channels_mark_pixel_invalid():
    x = torch.zeros(1, 2, 3, 3)
    x[..., 1, 2] = float("nan")
    mask = make_input_valid_mask(x)
    assert mask.shape == (1, 1, 3, 3)
    assert not mask[..., 1, 2].item()
