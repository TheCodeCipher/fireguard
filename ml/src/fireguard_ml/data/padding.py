"""Bottom/right padding helpers that preserve original frame dimensions."""

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


def pad_to_multiple(
    x: torch.Tensor, multiple: int = 32, value: float = 0.0
) -> tuple[torch.Tensor, PadInfo]:
    if x.ndim < 2:
        raise ValueError("Expected at least two dimensions")
    if multiple < 1:
        raise ValueError("multiple must be positive")
    height, width = x.shape[-2:]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    padded = F.pad(x, (0, pad_w, 0, pad_h), value=value)
    return padded, PadInfo(height, width, height + pad_h, width + pad_w, multiple)


def unpad(x: torch.Tensor, info: PadInfo) -> torch.Tensor:
    expected = (info.padded_height, info.padded_width)
    if x.shape[-2:] != expected:
        raise ValueError(f"Tensor has spatial size {tuple(x.shape[-2:])}; expected {expected}")
    return x[..., : info.original_height, : info.original_width]


def pad_valid_mask(mask: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, PadInfo]:
    """Pad a boolean validity mask, setting all newly padded cells false."""
    padded, info = pad_to_multiple(mask.to(dtype=torch.bool), multiple, value=0)
    return padded, info
