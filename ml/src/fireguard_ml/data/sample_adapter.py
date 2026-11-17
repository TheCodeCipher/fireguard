"""Model-facing spatial sample contract with fail-closed validity handling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .channel_schema import PROCESSED_WSTS_CHANNELS, ChannelSelection, select_channels
from .padding import PadInfo, pad_to_multiple, unpad


@dataclass
class ModelSample:
    inputs: torch.Tensor  # [B,C,H_pad,W_pad]
    target: torch.Tensor  # [B,1,H_pad,W_pad]
    input_valid_mask: torch.Tensor  # [B,1,H_pad,W_pad], authoritative only if source supplied
    target_valid_mask: torch.Tensor  # [B,1,H_pad,W_pad], authoritative only if source supplied
    loss_valid_mask: torch.Tensor  # intersection of input, target, and padding validity
    pad_info: PadInfo
    channel_selection: ChannelSelection
    validity_status: str
    event_id: str | None = None
    origin: str | None = None
    target_day: str | None = None


def _normalize_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 2:  # [H,W]
        return target.unsqueeze(0).unsqueeze(0)
    if target.ndim == 3:  # [B,H,W]
        return target.unsqueeze(1)
    if target.ndim == 4 and target.shape[1] == 1:  # [B,1,H,W]
        return target
    raise ValueError("Target must be [H,W], [B,H,W], or [B,1,H,W]")


def _normalize_pixel_mask(
    mask: torch.Tensor, target_shape: torch.Size, *, name: str
) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape != target_shape:
        raise ValueError(
            f"{name} must normalize to [B,1,H,W] matching target {tuple(target_shape)}"
        )
    return mask.bool()


def adapt_sample(
    inputs: torch.Tensor,
    target: torch.Tensor,
    *,
    channel_names: tuple[str, ...] = PROCESSED_WSTS_CHANNELS,
    selected_channels: tuple[str, ...] | None = None,
    strict_approved: bool = False,
    approval_reference: str | None = None,
    source_input_valid_mask: torch.Tensor | None = None,
    source_target_valid_mask: torch.Tensor | None = None,
    allow_finite_mask_fallback: bool = False,
    target_nodata_value: float | int | None = None,
    pad_multiple: int = 32,
    event_id: str | None = None,
    origin: str | None = None,
    target_day: str | None = None,
) -> ModelSample:
    """Adapt a spatial sample and apply authoritative source masks.

    Input accepts [C,H,W] or [B,C,H,W]; the leading dimension of 4D input is
    always batch. Temporal sequences require a distinct [B,T,C,H,W] contract.
    Real-data callers must provide separate input and target validity masks.
    The finite-only fallback is opt-in for synthetic/debug/interface checks.
    """
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)
    if inputs.ndim != 4:
        raise ValueError("Input must be [C,H,W] or [B,C,H,W]; temporal tensors need [B,T,C,H,W]")
    target = _normalize_target(target).to(device=inputs.device)
    if inputs.shape[0] != target.shape[0] or inputs.shape[-2:] != target.shape[-2:]:
        raise ValueError("Input and target batch/spatial dimensions must match before padding")

    selected, selection = select_channels(
        inputs,
        names=channel_names,
        selected_names=selected_channels,
        strict_approved=strict_approved,
        approval_reference=approval_reference,
    )
    source_masks_present = (
        source_input_valid_mask is not None and source_target_valid_mask is not None
    )
    if not source_masks_present and not allow_finite_mask_fallback:
        raise ValueError(
            "Source-aware input and target validity masks are required for real-data adaptation"
        )

    batch, channels, height, width = selected.shape
    input_finite = torch.isfinite(selected).all(dim=1, keepdim=True)
    if source_input_valid_mask is None:
        input_valid = input_finite
    else:
        imask = source_input_valid_mask.to(device=selected.device, dtype=torch.bool)
        if imask.ndim == 2:
            imask = imask.unsqueeze(0).unsqueeze(0)
        elif imask.ndim == 3:
            imask = imask.unsqueeze(1)
        if imask.ndim != 4 or imask.shape[0] != batch or imask.shape[-2:] != (height, width):
            raise ValueError("source_input_valid_mask must be [B,1,H,W] or [B,C,H,W]")
        if imask.shape[1] == len(channel_names):
            chosen_indices = [channel_names.index(name) for name in selection.names]
            imask = imask.index_select(1, torch.tensor(chosen_indices, device=imask.device))
        if imask.shape[1] not in (1, channels):
            raise ValueError("Input validity mask channel count must be 1 or match selected inputs")
        input_valid = imask.all(dim=1, keepdim=True) & input_finite

    target_finite = torch.isfinite(target)
    if source_target_valid_mask is None:
        target_valid = target_finite
    else:
        target_valid = (
            _normalize_pixel_mask(
                source_target_valid_mask.to(device=target.device),
                target.shape,
                name="source_target_valid_mask",
            )
            & target_finite
        )
    if target_nodata_value is not None:
        target_valid &= target != target_nodata_value

    loss_valid = input_valid & target_valid
    if not loss_valid.any():
        raise ValueError("No valid pixels remain after applying input and target masks")
    valid_targets = target[loss_valid]
    if not torch.isfinite(valid_targets).all():
        raise ValueError("Target contains NaN/Inf in source-valid cells")
    if not torch.logical_or(valid_targets == 0, valid_targets == 1).all():
        raise ValueError("Valid target values must be binary (0/1); convert labels upstream")

    # Values outside validity are replaced to keep model arithmetic finite;
    # masks ensure they do not contribute to the loss or metrics.
    selected = torch.where(
        input_valid.expand(-1, channels, -1, -1), selected, torch.zeros_like(selected)
    )
    selected = torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0).float()
    target = torch.where(loss_valid, target, torch.zeros_like(target)).float()

    x_pad, info = pad_to_multiple(selected, pad_multiple, value=0.0)
    y_pad, y_info = pad_to_multiple(target, pad_multiple, value=0.0)
    input_mask_pad, im_info = pad_to_multiple(input_valid, pad_multiple, value=0)
    target_mask_pad, tm_info = pad_to_multiple(target_valid, pad_multiple, value=0)
    if not (info == y_info == im_info == tm_info):
        raise RuntimeError("Input, target, and validity masks produced inconsistent padding")
    loss_mask_pad = input_mask_pad & target_mask_pad
    validity_status = "SOURCE_MASKED" if source_masks_present else "PROVISIONAL_FINITE_ONLY"
    return ModelSample(
        x_pad,
        y_pad,
        input_mask_pad.bool(),
        target_mask_pad.bool(),
        loss_mask_pad.bool(),
        info,
        selection,
        validity_status,
        event_id,
        origin,
        target_day,
    )


def restore_frame(tensor: torch.Tensor, sample: ModelSample) -> torch.Tensor:
    return unpad(tensor, sample.pad_info)


@dataclass
class InferenceInput:
    inputs: torch.Tensor  # [B,C,H_pad,W_pad]
    input_valid_mask: torch.Tensor  # [B,1,H_pad,W_pad]
    pad_info: PadInfo
    channel_selection: ChannelSelection
    validity_status: str


def adapt_inputs_for_inference(
    inputs: torch.Tensor,
    *,
    channel_names: tuple[str, ...] = PROCESSED_WSTS_CHANNELS,
    selected_channels: tuple[str, ...] | None = None,
    strict_approved: bool = False,
    approval_reference: str | None = None,
    source_input_valid_mask: torch.Tensor | None = None,
    allow_finite_mask_fallback: bool = False,
    pad_multiple: int = 32,
) -> InferenceInput:
    """Select, mask, and bottom/right-pad inputs without accepting or reading targets."""
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)
    if inputs.ndim != 4:
        raise ValueError("Inference input must be [C,H,W] or [B,C,H,W]")
    selected, selection = select_channels(
        inputs,
        names=channel_names,
        selected_names=selected_channels,
        strict_approved=strict_approved,
        approval_reference=approval_reference,
    )
    masks_present = source_input_valid_mask is not None
    if not masks_present and not allow_finite_mask_fallback:
        raise ValueError("A source-aware input validity mask is required for official inference")

    batch, channels, height, width = selected.shape
    finite = torch.isfinite(selected).all(dim=1, keepdim=True)
    if source_input_valid_mask is None:
        valid = finite
    else:
        mask = source_input_valid_mask.to(device=selected.device, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        if mask.ndim != 4 or mask.shape[0] != batch or mask.shape[-2:] != (height, width):
            raise ValueError("Input mask must normalize to [B,1,H,W] or [B,C,H,W]")
        if mask.shape[1] == len(channel_names):
            indices = [channel_names.index(name) for name in selection.names]
            mask = mask.index_select(1, torch.tensor(indices, device=mask.device))
        if mask.shape[1] not in (1, channels):
            raise ValueError("Input mask channels must be 1 or match selected input channels")
        valid = mask.all(dim=1, keepdim=True) & finite

    if not valid.any():
        raise ValueError("No valid input pixels remain")
    selected = torch.where(valid.expand(-1, channels, -1, -1), selected, 0)
    selected = torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0).float()
    x_pad, info = pad_to_multiple(selected, pad_multiple, value=0.0)
    mask_pad, mask_info = pad_to_multiple(valid, pad_multiple, value=0)
    if info != mask_info:
        raise RuntimeError("Input and validity mask produced inconsistent padding")
    return InferenceInput(
        x_pad,
        mask_pad.bool(),
        info,
        selection,
        "SOURCE_MASKED" if masks_present else "PROVISIONAL_FINITE_ONLY",
    )
