"""Checkpoint save/load that always carries what the offline OOF export needs (Role 3).

A checkpoint is self-describing: it rebuilds the model from its own spec, so exporting
predictions never depends on someone remembering the constructor arguments, channel order
or fold.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .temporal_unet import MODEL_REGISTRY, _TemporalBase
from .unet import UNet

REQUIRED_META = ("channels", "T", "preprocessing_version", "config_hash", "fold",
                 "split_manifest_hash")


# The dataset-creation code names these bands "forecast ..." (FirePred.py); the data
# dictionary calls them "gfs_...". Block both spellings so a rename cannot slip a forecast
# channel into a strict model.
FORBIDDEN_STRICT_SUBSTRINGS = ("gfs", "forecast")
SPATIAL_REQUIRED_META = (
    "model_version",
    "channel_names",
    "data_contract_version",
    "config_sha256",
    "split_manifest_sha256",
    "test_fold_year",
    "seed",
    "environment",
    "training_group_ids",
    "validation_group_ids",
    "strict",
)


def assert_strict_channels(channels) -> None:
    bad = [c for c in channels if any(s in str(c).lower() for s in FORBIDDEN_STRICT_SUBSTRINGS)]
    if bad:
        raise ValueError(f"GFS / forecast channels are not allowed in strict models: {bad}")


def save_checkpoint(model: _TemporalBase, path: str | Path, meta: dict[str, object],
                    strict: bool = True) -> None:
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"checkpoint meta is missing: {missing}")
    if len(meta["channels"]) != model.c_strict:
        raise ValueError(
            f"meta has {len(meta['channels'])} channels but model expects {model.c_strict}")
    if int(meta["T"]) != model.T:
        raise ValueError(f"meta T={meta['T']} but model T={model.T}")
    if strict:
        assert_strict_channels(meta["channels"])
    payload = {"spec": model.spec(), "meta": dict(meta), "strict": strict,
               "state_dict": model.state_dict()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path,
                    map_location: str = "cpu") -> tuple[_TemporalBase, dict[str, object]]:
    """Rebuild a model from its own spec.

    `model.strict_checkpoint` records whether the file was written as a strict (no-forecast-channel)
    model. Read it before putting predictions into a strict results table: a benchmark-assumption
    model loads exactly like a strict one and is otherwise indistinguishable.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    spec = dict(payload["spec"])
    cls = MODEL_REGISTRY[spec.pop("model_type")]
    model = cls(**spec)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = payload["meta"]
    strict = bool(payload["strict"])
    if strict:
        assert_strict_channels(meta["channels"])  # the file may have been written by an older guard
    model.strict_checkpoint = strict
    return model, meta


def save_spatial_checkpoint(
    model: UNet,
    path: str | Path,
    meta: dict[str, object],
) -> None:
    """Save a self-describing M1 checkpoint with fold and input provenance."""
    if not isinstance(model, UNet):
        raise TypeError("Spatial checkpoint must contain a fireguard_ml.models.UNet")
    missing = [key for key in SPATIAL_REQUIRED_META if key not in meta]
    if missing:
        raise ValueError(f"spatial checkpoint metadata is missing: {missing}")
    if len(meta["channel_names"]) != model.in_channels:
        raise ValueError("channel_names length must match the U-Net input channel count")
    if not isinstance(meta["training_group_ids"], list) or not isinstance(
        meta["validation_group_ids"], list
    ):
        raise ValueError("training_group_ids and validation_group_ids must be lists")
    if meta["strict"]:
        assert_strict_channels(meta["channel_names"])
    payload = {
        "checkpoint_version": 1,
        "model_type": "spatial_unet",
        "model_spec": {
            "in_channels": model.in_channels,
            "base_channels": model.base_channels,
            "out_channels": model.out_channels,
        },
        "meta": dict(meta),
        "state_dict": model.state_dict(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_spatial_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[UNet, dict[str, object]]:
    """Load an M1 checkpoint and re-check its strict channel metadata."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("checkpoint_version") != 1 or payload.get("model_type") != "spatial_unet":
        raise ValueError("Unsupported spatial checkpoint format")
    model = UNet(**payload["model_spec"])
    model.load_state_dict(payload["state_dict"])
    meta = payload["meta"]
    missing = [key for key in SPATIAL_REQUIRED_META if key not in meta]
    if missing:
        raise ValueError(f"spatial checkpoint metadata is missing: {missing}")
    if len(meta["channel_names"]) != model.in_channels:
        raise ValueError("checkpoint channel_names do not match model input channels")
    if meta["strict"]:
        assert_strict_channels(meta["channel_names"])
    model.strict_checkpoint = bool(meta["strict"])
    model.eval()
    return model, meta
