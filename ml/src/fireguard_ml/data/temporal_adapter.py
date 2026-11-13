"""Full-frame adapter for M3/M4 temporal samples and validity masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .padding import PadInfo, pad_to_multiple


@dataclass
class TemporalModelSample:
    inputs: torch.Tensor  # [B,T,C,H_pad,W_pad], oldest frame first
    target: torch.Tensor  # [B,1,H_pad,W_pad]
    input_valid_mask: torch.Tensor  # [B,T,1,H_pad,W_pad]
    target_valid_mask: torch.Tensor  # [B,1,H_pad,W_pad]
    loss_valid_mask: torch.Tensor  # target-valid and valid in every input frame
    pad_info: PadInfo
    validity_status: str


def _target_shape(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 2:
        target = target[None, None]
    elif target.ndim == 3:
        target = target[:, None]
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("Target must be [H,W], [B,H,W], or [B,1,H,W]")
    return target


def _target_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape != target.shape:
        raise ValueError("Target mask must be [B,1,H,W] and match target")
    return mask.bool()


def adapt_temporal_sample(
    inputs: torch.Tensor,
    target: torch.Tensor,
    *,
    source_input_valid_mask: torch.Tensor | None = None,
    source_target_valid_mask: torch.Tensor | None = None,
    allow_finite_mask_fallback: bool = False,
    target_nodata_value: float | int | None = None,
    pad_multiple: int = 32,
) -> TemporalModelSample:
    """Mask and bottom/right-pad one `[B,T,C,H,W]` input and aligned next-day target.

    Input channels must already have been transformed/selected according to the approved
    channel manifest. By default, separate source-derived input and target masks are required.
    A loss pixel is valid only when its target is valid and every input frame is valid there.
    """
    if inputs.ndim != 5:
        raise ValueError("Input must be [B,T,C,H,W] with frames ordered oldest to newest")
    if min(inputs.shape[:3]) < 1:
        raise ValueError("Batch, time, and channel dimensions must be non-empty")
    target = _target_shape(target).to(device=inputs.device)
    b, t, c, h, w = inputs.shape
    if target.shape != (b, 1, h, w):
        raise ValueError("Input and target batch/spatial dimensions must match before padding")
    masks_present = source_input_valid_mask is not None and source_target_valid_mask is not None
    if not masks_present and not allow_finite_mask_fallback:
        raise ValueError("Source-aware input and target validity masks are required for real data")

    finite = torch.isfinite(inputs).all(dim=2, keepdim=True)
    if source_input_valid_mask is None:
        input_valid = finite
    else:
        imask = source_input_valid_mask.to(device=inputs.device, dtype=torch.bool)
        if imask.ndim == 4:
            imask = imask.unsqueeze(2)
        if imask.ndim != 5 or imask.shape[:2] != (b, t) or imask.shape[-2:] != (h, w):
            raise ValueError("Input mask must be [B,T,1,H,W] or [B,T,C,H,W]")
        if imask.shape[2] not in (1, c):
            raise ValueError("Input mask channel dimension must be 1 or match input channels")
        input_valid = imask.all(dim=2, keepdim=True) & finite

    target_finite = torch.isfinite(target)
    if source_target_valid_mask is None:
        target_valid = target_finite
    else:
        target_valid = (
            _target_mask(source_target_valid_mask.to(device=target.device), target) & target_finite
        )
    if target_nodata_value is not None:
        target_valid &= target != target_nodata_value

    loss_valid = target_valid & input_valid.all(dim=1)
    if not loss_valid.any():
        raise ValueError("No valid pixels remain after applying temporal and target masks")
    valid_targets = target[loss_valid]
    if not torch.isfinite(valid_targets).all():
        raise ValueError("Target contains NaN/Inf in source-valid cells")
    if not torch.logical_or(valid_targets == 0, valid_targets == 1).all():
        raise ValueError("Valid target values must be binary (0/1)")

    inputs = torch.where(input_valid.expand(-1, -1, c, -1, -1), inputs, 0)
    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0).float()
    target = torch.where(loss_valid, target, 0).float()
    x_pad, info = pad_to_multiple(inputs, pad_multiple, value=0.0)
    y_pad, y_info = pad_to_multiple(target, pad_multiple, value=0.0)
    im_pad, im_info = pad_to_multiple(input_valid, pad_multiple, value=0)
    tm_pad, tm_info = pad_to_multiple(target_valid, pad_multiple, value=0)
    if not (info == y_info == im_info == tm_info):
        raise RuntimeError("Input, target, and validity masks produced inconsistent padding")
    loss_pad = tm_pad & im_pad.all(dim=1)
    return TemporalModelSample(
        x_pad,
        y_pad,
        im_pad.bool(),
        tm_pad.bool(),
        loss_pad.bool(),
        info,
        "SOURCE_MASKED" if masks_present else "PROVISIONAL_FINITE_ONLY",
    )
