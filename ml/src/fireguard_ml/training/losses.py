"""NaN-safe masked BCE-with-logits plus soft Dice loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute BCE + soft Dice only on valid binary-target pixels.

    Invalid positions are selected out before arithmetic. This avoids IEEE
    ``NaN * 0 == NaN`` contamination. Non-finite values in valid positions are
    rejected rather than silently repaired.
    """
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, targets, and valid_mask must have identical shapes")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be floating point")
    if bce_weight < 0 or dice_weight < 0 or smooth <= 0:
        raise ValueError("Loss weights must be nonnegative and smooth must be positive")

    mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    selected_logits = logits.masked_select(mask)
    selected_targets = targets.to(device=logits.device).masked_select(mask)
    if selected_logits.numel() == 0:
        raise ValueError("valid_mask contains no valid pixels")
    if not torch.isfinite(selected_logits).all():
        raise ValueError("logits contain NaN/Inf in valid pixels")
    if not torch.isfinite(selected_targets).all():
        raise ValueError("targets contain NaN/Inf in valid pixels")
    if not torch.logical_or(selected_targets == 0, selected_targets == 1).all():
        raise ValueError("targets must be binary (0/1) in valid pixels")

    selected_targets = selected_targets.to(dtype=selected_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(selected_logits, selected_targets)
    probabilities = torch.sigmoid(selected_logits)
    dice = 1.0 - (2.0 * (probabilities * selected_targets).sum() + smooth) / (
        probabilities.sum() + selected_targets.sum() + smooth
    )
    return bce_weight * bce + dice_weight * dice
