"""Replaceable adapter for the documented processed WSTS channel schema.

This schema names the official baseline's 40-channel processed example. It is
not an approval that any channel is safe for the strict FireGuard model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


PROCESSED_WSTS_CHANNELS: tuple[str, ...] = (
    "viirs_m11", "viirs_i2", "viirs_i1", "ndvi", "evi2",
    "total_precipitation", "wind_speed", "wind_direction", "minimum_temperature",
    "maximum_temperature", "energy_release_component", "specific_humidity",
    "slope", "aspect", "elevation", "pdsi",
    "landcover_evergreen_needleleaf", "landcover_evergreen_broadleaf",
    "landcover_deciduous_needleleaf", "landcover_deciduous_broadleaf",
    "landcover_mixed_forest", "landcover_closed_shrubland", "landcover_open_shrubland",
    "landcover_woody_savanna", "landcover_savanna", "landcover_grassland",
    "landcover_permanent_wetland", "landcover_cropland", "landcover_urban",
    "landcover_cropland_natural_vegetation_mosaic", "landcover_snow_ice",
    "landcover_barren", "landcover_water",
    "forecast_total_precipitation", "forecast_wind_speed", "forecast_wind_direction",
    "forecast_temperature", "forecast_specific_humidity",
    "active_fire_detection_time", "active_fire_binary",
)


@dataclass(frozen=True)
class ChannelSelection:
    """Selection metadata carried into run logs/checkpoints."""

    names: tuple[str, ...]
    schema_name: str
    selection_status: str


def select_channels(
    x: torch.Tensor,
    *,
    names: tuple[str, ...] = PROCESSED_WSTS_CHANNELS,
    selected_names: tuple[str, ...] | None = None,
    strict_approved: bool = False,
) -> tuple[torch.Tensor, ChannelSelection]:
    """Select named channels from [B,C,H,W], preserving requested order.

    For exploratory/synthetic work, the documented 40-channel baseline schema
    may be used with strict_approved=False. Strict FireGuard training requires
    an explicit reviewed channel set passed with strict_approved=True.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    if x.shape[1] != len(names):
        raise ValueError(f"Tensor has {x.shape[1]} channels but schema has {len(names)}")
    if selected_names is None:
        selected_names = names
    if not selected_names:
        raise ValueError("At least one channel must be selected")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("selected_names contains duplicates")
    unknown = [name for name in selected_names if name not in names]
    if unknown:
        raise ValueError(f"Unknown channel names: {unknown}")
    if strict_approved and selected_names == names:
        raise ValueError("Full 40-channel schema cannot be assumed strict-approved; provide the reviewed subset")
    indices = torch.tensor([names.index(name) for name in selected_names], device=x.device)
    status = "ROLE4_APPROVED" if strict_approved else "BASELINE_SCHEMA_EXPLORATORY_ONLY"
    return x.index_select(1, indices), ChannelSelection(tuple(selected_names), "wsts_processed_v1", status)
