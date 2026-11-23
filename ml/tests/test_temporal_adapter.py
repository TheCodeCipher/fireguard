import pytest
import torch

from fireguard_ml.data.padding import unpad
from fireguard_ml.data.temporal_adapter import adapt_temporal_sample


def test_full_frame_pad_unpad_keeps_temporal_inputs_and_masks_aligned():
    x = torch.ones(1, 5, 2, 304, 247)
    y = torch.zeros(1, 1, 304, 247)
    imask = torch.ones(1, 5, 1, 304, 247, dtype=torch.bool)
    tmask = torch.ones(1, 1, 304, 247, dtype=torch.bool)

    sample = adapt_temporal_sample(
        x, y, source_input_valid_mask=imask, source_target_valid_mask=tmask
    )

    assert sample.inputs.shape == (1, 5, 2, 320, 256)
    assert sample.target.shape == (1, 1, 320, 256)
    assert sample.loss_valid_mask[:, :, 304:, :].sum() == 0
    assert sample.loss_valid_mask[:, :, :, 247:].sum() == 0
    assert unpad(sample.inputs, sample.pad_info).shape == x.shape
    assert unpad(sample.target, sample.pad_info).shape == y.shape


def test_all_input_days_and_target_must_be_valid_for_loss():
    x = torch.ones(1, 2, 1, 3, 4)
    y = torch.zeros(1, 1, 3, 4)
    imask = torch.ones(1, 2, 1, 3, 4, dtype=torch.bool)
    tmask = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    imask[0, 0, 0, 1, 2] = False
    tmask[0, 0, 2, 1] = False

    sample = adapt_temporal_sample(
        x,
        y,
        source_input_valid_mask=imask,
        source_target_valid_mask=tmask,
        pad_multiple=1,
    )

    assert not sample.loss_valid_mask[0, 0, 1, 2]
    assert not sample.loss_valid_mask[0, 0, 2, 1]
    assert sample.inputs[0, 0, 0, 1, 2] == 0


def test_finite_nodata_target_is_excluded_and_valid_zero_is_kept():
    x = torch.ones(1, 2, 1, 2, 2)
    y = torch.tensor([[[[0.0, -1.0], [1.0, 0.0]]]])
    imask = torch.ones(1, 2, 1, 2, 2, dtype=torch.bool)
    tmask = torch.ones(1, 1, 2, 2, dtype=torch.bool)

    sample = adapt_temporal_sample(
        x,
        y,
        source_input_valid_mask=imask,
        source_target_valid_mask=tmask,
        target_nodata_value=-1,
        pad_multiple=1,
    )

    assert sample.loss_valid_mask.sum() == 3
    assert sample.loss_valid_mask[0, 0, 0, 0]
    assert not sample.loss_valid_mask[0, 0, 0, 1]


def test_temporal_mask_is_required_unless_finite_fallback_is_explicit():
    x = torch.ones(1, 2, 1, 2, 2)
    y = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match="Source-aware"):
        adapt_temporal_sample(x, y)
    sample = adapt_temporal_sample(x, y, allow_finite_mask_fallback=True)
    assert sample.validity_status == "PROVISIONAL_FINITE_ONLY"


def test_masked_nan_in_temporal_input_is_zeroed_before_model_arithmetic():
    x = torch.ones(1, 2, 1, 2, 2)
    x[0, 1, 0, 1, 1] = float("nan")
    y = torch.zeros(1, 1, 2, 2)
    imask = torch.ones(1, 2, 1, 2, 2, dtype=torch.bool)
    tmask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    sample = adapt_temporal_sample(
        x,
        y,
        source_input_valid_mask=imask,
        source_target_valid_mask=tmask,
        pad_multiple=1,
    )
    assert sample.inputs[0, 1, 0, 1, 1] == 0
    assert not sample.loss_valid_mask[0, 0, 1, 1]
