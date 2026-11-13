"""Canonical model-facing sample adapter for future Role 1 loader handoffs."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .channel_schema import ChannelSelection, PROCESSED_WSTS_CHANNELS, select_channels
from .padding import PadInfo, make_input_valid_mask, pad_to_multiple, unpad


@dataclass
class ModelSample:
    inputs: torch.Tensor  # [B,C,H_pad,W_pad]
    target: torch.Tensor  # [B,1,H_pad,W_pad]
    valid_mask: torch.Tensor  # [B,1,H_pad,W_pad]
    pad_info: PadInfo
    channel_selection: ChannelSelection
    event_id: str | None = None
    origin: str | None = None
    target_day: str | None = None


def adapt_sample(
    inputs: torch.Tensor,
    target: torch.Tensor,
    *,
    channel_names: tuple[str, ...] = PROCESSED_WSTS_CHANNELS,
    selected_channels: tuple[str, ...] | None = None,
    strict_approved: bool = False,
    source_valid_mask: torch.Tensor | None = None,
    pad_multiple: int = 32,
    event_id: str | None = None,
    origin: str | None = None,
    target_day: str | None = None,
) -> ModelSample:
    """Normalize documented WSTS example tensors to the model contract.

    Accepts x as [B,C,H,W] or [T,C,H,W] (the documented example uses T=1).
    Target may be [H,W], [B,H,W] or [B,1,H,W]. Uses all input channels only
    for exploratory work; strict training should pass Role 4's approved list.
    """
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)
    if inputs.ndim != 4:
        raise ValueError(f"Expected input [B,C,H,W] (or [C,H,W]), got {tuple(inputs.shape)}")
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("Target must be [H,W], [B,H,W] or [B,1,H,W]")
    if inputs.shape[0] != target.shape[0] or inputs.shape[-2:] != target.shape[-2:]:
        raise ValueError("Input and target batch/spatial dimensions must match before padding")
    selected, selection = select_channels(
        inputs, names=channel_names, selected_names=selected_channels, strict_approved=strict_approved
    )
    if source_valid_mask is None:
        valid = make_input_valid_mask(selected)
    else:
        if source_valid_mask.ndim == 2:
            source_valid_mask = source_valid_mask.unsqueeze(0).unsqueeze(0)
        elif source_valid_mask.ndim == 3:
            source_valid_mask = source_valid_mask.unsqueeze(1)
        if source_valid_mask.shape != target.shape:
            raise ValueError("source_valid_mask must match target shape")
        valid = source_valid_mask.bool() & make_input_valid_mask(selected)
    valid &= torch.isfinite(target)
    target = torch.nan_to_num(target.float(), nan=0.0).gt(0).float()
    selected = torch.nan_to_num(selected.float(), nan=0.0, posinf=0.0, neginf=0.0)
    xp, info = pad_to_multiple(selected, pad_multiple, value=0.0)
    yp, _ = pad_to_multiple(target, pad_multiple, value=0.0)
    mp, _ = pad_to_multiple(valid, pad_multiple, value=0)
    return ModelSample(xp, yp, mp.bool(), info, selection, event_id, origin, target_day)


def restore_frame(tensor: torch.Tensor, sample: ModelSample) -> torch.Tensor:
    """Remove padding from predictions/targets before full-frame evaluation."""
    return unpad(tensor, sample.pad_info)
