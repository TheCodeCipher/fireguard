"""Documented WSTS processed channel names and replaceable channel selection.

The named 40-channel schema is exploratory. It is not approval for strict M1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

# INTEGRATION (Prototype, 2026-11-13): this tuple used to be a second, hand-written copy of the
# 40 processed channel names. 24 of its 40 names differed from Role 1's loader and Role 4's
# leakage register (e.g. "active_fire_binary" vs "binary_active_fire_mask", "landcover_*" vs
# "land_cover_*", "forecast_*" vs "gfs_*"), so a strict selection by name would have failed or,
# worse, needed a second mapping. The order was already identical. It now re-exports Role 1's
# tuple, which is verified against the pinned WSTS loader, so there is one source of truth.
from .wsts_dataset import PROCESSED_CHANNELS as PROCESSED_WSTS_CHANNELS  # noqa: E402


@dataclass(frozen=True)
class ChannelSelection:
    names: tuple[str, ...]
    schema_name: str
    selection_status: str
    sha256: str


def channel_names_sha256(names: tuple[str, ...]) -> str:
    payload = json.dumps(list(names), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_channels(
    x: torch.Tensor,
    *,
    names: tuple[str, ...] = PROCESSED_WSTS_CHANNELS,
    selected_names: tuple[str, ...] | None = None,
    strict_approved: bool = False,
    approval_reference: str | None = None,
) -> tuple[torch.Tensor, ChannelSelection]:
    """Select/reorder channels. Strict mode requires named approval evidence."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    if x.shape[1] != len(names):
        raise ValueError(f"Tensor has {x.shape[1]} channels but schema has {len(names)}")
    selected = names if selected_names is None else selected_names
    if not selected:
        raise ValueError("At least one channel must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_names contains duplicates")
    unknown = [name for name in selected if name not in names]
    if unknown:
        raise ValueError(f"Unknown channel names: {unknown}")
    if strict_approved and (not approval_reference or selected == names):
        raise ValueError(
            "Strict channel selection requires an explicit reviewed subset and approval reference"
        )
    status = "ROLE4_APPROVED" if strict_approved else "EXPLORATORY_NOT_STRICT_APPROVED"
    indices = torch.tensor([names.index(name) for name in selected], device=x.device)
    return x.index_select(1, indices), ChannelSelection(
        tuple(selected), "wsts_processed_v1", status, channel_names_sha256(tuple(selected))
    )
