from __future__ import annotations

import hashlib

import pytest
import torch

from fireguard_ml.data.channel_schema import PROCESSED_WSTS_CHANNELS, select_channels
from fireguard_ml.data.padding import pad_to_multiple, unpad
from fireguard_ml.data.sample_adapter import adapt_sample, restore_frame
from fireguard_ml.models import UNet
from fireguard_ml.training.config import (
    configure_reproducibility,
    load_config,
    require_training_handoffs,
)
from fireguard_ml.training.losses import masked_bce_dice_loss


def test_pad_unpad_identity_odd_shape() -> None:
    x = torch.randn(2, 3, 305, 207)
    padded, info = pad_to_multiple(x)
    assert padded.shape[-2:] == (320, 224)
    assert torch.equal(unpad(padded, info), x)


def test_odd_dimensions_run_through_model_after_pad() -> None:
    x = torch.randn(1, 3, 305, 207)
    padded, info = pad_to_multiple(x)
    logits = UNet(3, base_channels=4)(padded)
    assert logits.shape[-2:] == padded.shape[-2:]
    assert unpad(logits, info).shape[-2:] == (305, 207)


def test_unet_returns_logits_and_rejects_temporal_ambiguity() -> None:
    model = UNet(3, base_channels=4)
    assert model(torch.randn(2, 3, 64, 96)).shape == (2, 1, 64, 96)
    with pytest.raises(ValueError, match=r"\[B,C,H,W\]"):
        model(torch.randn(1, 2, 3, 64, 96))


def test_masked_nan_and_inf_values_do_not_poison_loss() -> None:
    logits = torch.tensor([[[[0.0, float("nan"), float("inf")]]]], requires_grad=True)
    targets = torch.tensor([[[[1.0, float("nan"), float("inf")]]]])
    mask = torch.tensor([[[[True, False, False]]]])
    loss = masked_bce_dice_loss(logits, targets, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad[..., 0]).all()


@pytest.mark.parametrize(
    "logit,target",
    [
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0, 0.5),
    ],
)
def test_loss_rejects_invalid_values_inside_valid_pixels(logit: float, target: float) -> None:
    with pytest.raises(ValueError):
        masked_bce_dice_loss(
            torch.tensor([[[[logit]]]]),
            torch.tensor([[[[target]]]]),
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
        )


def test_loss_ignores_changes_at_masked_positions() -> None:
    logits = torch.zeros(1, 1, 1, 3)
    targets = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    mask = torch.tensor([[[[True, False, True]]]])
    baseline = masked_bce_dice_loss(logits, targets, mask)
    targets[..., 1] = float("nan")
    logits[..., 1] = float("inf")
    assert torch.equal(baseline, masked_bce_dice_loss(logits, targets, mask))


def test_empty_valid_mask_rejected() -> None:
    tensor = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match="no valid pixels"):
        masked_bce_dice_loss(tensor, tensor, torch.zeros_like(tensor, dtype=torch.bool))


def test_source_mask_excludes_finite_nodata_and_keeps_valid_zero() -> None:
    x = torch.zeros(1, 40, 3, 4)
    y = torch.zeros(3, 4, dtype=torch.int64)
    source_input_valid = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    source_target_valid = torch.ones_like(source_input_valid)
    source_input_valid[..., 0, 0] = False  # finite nodata sentinel is invalid by source mask
    x[..., 0, 0] = 9999.0
    sample = adapt_sample(
        x,
        y,
        source_input_valid_mask=source_input_valid,
        source_target_valid_mask=source_target_valid,
    )
    assert not sample.input_valid_mask[..., 0, 0].item()
    assert sample.input_valid_mask[..., 0, 1].item()  # valid zero remains valid
    assert sample.validity_status == "SOURCE_MASKED"


def test_target_nodata_excluded_and_padded_cells_invalid() -> None:
    x = torch.zeros(1, 40, 3, 5)
    y = torch.zeros(3, 5, dtype=torch.float32)
    y[0, 0] = -9999.0
    iv = torch.ones(1, 1, 3, 5, dtype=torch.bool)
    tv = torch.ones_like(iv)
    sample = adapt_sample(
        x,
        y,
        source_input_valid_mask=iv,
        source_target_valid_mask=tv,
        target_nodata_value=-9999.0,
    )
    assert not sample.target_valid_mask[..., 0, 0].item()
    assert not sample.loss_valid_mask[..., 0, 0].item()
    assert sample.inputs.shape[-2:] == (32, 32)
    assert not sample.loss_valid_mask[..., 3:, :].any()
    assert not sample.loss_valid_mask[..., :, 5:].any()
    assert restore_frame(sample.target, sample).shape[-2:] == (3, 5)


def test_real_data_adapter_fails_closed_without_source_masks() -> None:
    x = torch.zeros(1, 40, 2, 2)
    y = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="Source-aware"):
        adapt_sample(x, y)


def test_provisional_finite_fallback_is_explicit() -> None:
    x = torch.zeros(1, 40, 2, 2)
    y = torch.zeros(2, 2)
    sample = adapt_sample(x, y, allow_finite_mask_fallback=True)
    assert sample.validity_status == "PROVISIONAL_FINITE_ONLY"


def test_masks_must_align_and_ambiguous_sequence_shape_rejected() -> None:
    x = torch.zeros(1, 40, 4, 4)
    y = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="source_input_valid_mask"):
        adapt_sample(
            x,
            y,
            source_input_valid_mask=torch.ones(2, 2, dtype=torch.bool),
            source_target_valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="temporal tensors"):
        adapt_sample(torch.zeros(1, 2, 40, 4, 4), y, allow_finite_mask_fallback=True)


def test_channel_order_and_hash_stable() -> None:
    x = torch.arange(40.0).reshape(1, 40, 1, 1)
    result, meta = select_channels(x, selected_names=("binary_active_fire_mask", "elevation"))
    assert result[:, 0].item() == 39 and result[:, 1].item() == 14
    assert len(meta.sha256) == 64
    with pytest.raises(ValueError, match="duplicates"):
        select_channels(x, selected_names=("elevation", "elevation"))
    with pytest.raises(ValueError, match="Unknown"):
        select_channels(x, selected_names=("not_a_channel",))


def test_strict_channels_require_review_reference() -> None:
    x = torch.zeros(1, 40, 2, 2)
    with pytest.raises(ValueError, match="reviewed subset"):
        select_channels(x, strict_approved=True)
    subset = PROCESSED_WSTS_CHANNELS[:2]
    _, meta = select_channels(
        x, selected_names=subset, strict_approved=True, approval_reference="role4-review"
    )
    assert meta.selection_status == "ROLE4_APPROVED"


def test_training_guard_requires_masks_strict_channels_and_split_hash() -> None:
    config = {
        "channel_schema": {"strict_approved": False, "approval_reference": None},
        "data": {"split_manifest_path": None, "split_manifest_sha256": None},
    }
    with pytest.raises(RuntimeError, match="Role 4 strict channel approval"):
        require_training_handoffs(config, validity_status="PROVISIONAL_FINITE_ONLY")


def test_split_manifest_hash_is_checked(tmp_path) -> None:
    from fireguard_ml.training.config import verify_split_manifest

    manifest = tmp_path / "splits.json"
    manifest.write_text('{"fold":"p1"}', encoding="utf-8")
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config = {"data": {"split_manifest_path": str(manifest), "split_manifest_sha256": expected}}
    assert verify_split_manifest(config, project_root=tmp_path) == expected
    config["data"]["split_manifest_sha256"] = "bad-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        verify_split_manifest(config, project_root=tmp_path)


def test_source_masked_training_gate_passes_only_with_exact_handoffs() -> None:
    from fireguard_ml.data.channel_schema import channel_names_sha256

    names = PROCESSED_WSTS_CHANNELS[:2]
    digest = channel_names_sha256(names)
    config = {
        "channel_schema": {
            "strict_approved": True,
            "approval_reference": "role4-approved-record",
            "approved_channels_sha256": digest,
        },
        "data": {"split_manifest_path": "splits.json", "split_manifest_sha256": "abc123"},
    }
    require_training_handoffs(
        config,
        validity_status="SOURCE_MASKED",
        split_manifest_sha256="abc123",
        selected_channel_sha256=digest,
    )


def test_canonical_yaml_config_is_loaded_and_hashed() -> None:
    config, digest = load_config("../configs/unet.yaml")
    assert config["model"]["base_channels"] == 4
    assert config["channel_schema"]["strict_approved"] is False
    assert len(digest) == 64


def test_reproducibility_seed_repeats_synthetic_model_output() -> None:
    configure_reproducibility(1234, True)
    first_model = UNet(2, base_channels=4)
    inputs = torch.randn(1, 2, 32, 32)
    first = first_model(inputs)
    configure_reproducibility(1234, True)
    second_model = UNet(2, base_channels=4)
    inputs_again = torch.randn(1, 2, 32, 32)
    second = second_model(inputs_again)
    assert torch.equal(inputs, inputs_again)
    assert torch.equal(first, second)
