"""Padding helpers. Pad bottom/right only; metadata enables exact unpadding."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PadInfo:
    original_height: int
    original_width: int
    padded_height: int
    padded_width: int
    multiple: int


def pad_to_multiple(x: torch.Tensor, multiple: int = 32, value: float = 0.0) -> tuple[torch.Tensor, PadInfo]:
    """Pad [H,W] or [...,H,W] on bottom/right to a requested multiple."""
    if x.ndim < 2:
        raise ValueError("Expected at least two dimensions")
    if multiple < 1:
        raise ValueError("multiple must be positive")
    h, w = x.shape[-2:]
    ph = (-h) % multiple
    pw = (-w) % multiple
    y = F.pad(x, (0, pw, 0, ph), value=value)
    return y, PadInfo(h, w, h + ph, w + pw, multiple)


def unpad(x: torch.Tensor, info: PadInfo) -> torch.Tensor:
    """Restore original H/W by removing bottom/right padding."""
    if x.shape[-2:] != (info.padded_height, info.padded_width):
        raise ValueError(f"Tensor has spatial size {tuple(x.shape[-2:])}, expected {(info.padded_height, info.padded_width)}")
    return x[..., : info.original_height, : info.original_width]


def pad_with_valid_mask(x: torch.Tensor, valid_mask: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, torch.Tensor, PadInfo]:
    """Pad input and validity mask consistently; padded mask cells are false."""
    if valid_mask.shape[-2:] != x.shape[-2:]:
        raise ValueError("Input and valid mask must have matching H/W before padding")
    xp, info = pad_to_multiple(x, multiple)
    mp, _ = pad_to_multiple(valid_mask.to(dtype=torch.bool), multiple, value=0)
    return xp, mp, info


def make_input_valid_mask(x: torch.Tensor) -> torch.Tensor:
    """Return per-pixel validity based on all selected input channels being finite.

    Input shape is [..., C, H, W]. A pixel is valid only when every selected
    input channel is finite. The returned shape is [..., 1, H, W]. This is a
    conservative default; source-specific masks should replace it when Role 1
    provides them.
    """
    if x.ndim < 3:
        raise ValueError("Expected [..., C, H, W]")
    return torch.isfinite(x).all(dim=-3, keepdim=True)
