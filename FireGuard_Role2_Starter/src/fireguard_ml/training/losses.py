"""Masked BCE + soft Dice losses for binary per-pixel fire targets."""
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
    """Compute BCE + soft Dice over valid cells only.

    Shapes must be broadcast-compatible and normally [B,1,H,W]. Invalid and
    padded cells have no effect on either term. Targets are binary floats.
    """
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, targets, and valid_mask must have identical shapes")
    if not torch.is_floating_point(targets):
        targets = targets.float()
    mask = valid_mask.to(device=logits.device, dtype=logits.dtype)
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("valid_mask contains no valid pixels")
    bce_map = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    bce = (bce_map * mask).sum() / count
    probs = torch.sigmoid(logits) * mask
    y = targets.to(logits.dtype) * mask
    dice = 1.0 - (2.0 * (probs * y).sum() + smooth) / (probs.sum() + y.sum() + smooth)
    return bce_weight * bce + dice_weight * dice
