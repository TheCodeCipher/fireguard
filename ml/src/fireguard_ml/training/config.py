"""Executable, validated YAML model configuration and reproducibility setup."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import torch
import yaml

_SECTION_KEYS = {
    "model": {"name", "base_channels", "depth", "output"},
    "channel_schema": {
        "name",
        "status",
        "channel_names_source",
        "selected_names",
        "strict_approved",
        "approval_reference",
        "approved_channels_sha256",
    },
    "spatial": {"pad_multiple", "pad_side", "evaluation_policy", "geometric_augmentation"},
    "loss": {"name", "bce_weight", "dice_weight"},
    "reproducibility": {"seed", "deterministic"},
    "data": {
        "sample_shape_reference",
        "sample_event_reference",
        "sample_input_day_reference",
        "sample_target_day_reference",
        "split_manifest_path",
        "split_manifest_sha256",
        "target_nodata_value",
        "use_gfs",
    },
}
_TOP_LEVEL_KEYS = set(_SECTION_KEYS)


def load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    config_path = Path(path)
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a YAML mapping")
    unknown = set(config) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    for key in _TOP_LEVEL_KEYS:
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Config section {key!r} must be a mapping")
        unknown_section = set(config[key]) - _SECTION_KEYS[key]
        if unknown_section:
            raise ValueError(f"Unknown keys in config section {key!r}: {sorted(unknown_section)}")
    if config["model"].get("output", "logits") != "logits":
        raise ValueError("Only raw logits output is supported")
    if config["model"].get("depth") != 4:
        raise ValueError("Only the four-level U-Net depth is implemented")
    if config["spatial"].get("pad_side") != "bottom_right":
        raise ValueError("Only bottom_right padding is supported")
    if config["spatial"].get("pad_multiple") != 32:
        raise ValueError("Project contract requires pad_multiple=32")
    if config["spatial"].get("evaluation_policy") != "full_frame_pad_then_unpad":
        raise ValueError("Only full_frame_pad_then_unpad evaluation is supported")
    if config["spatial"].get("geometric_augmentation") is not False:
        raise ValueError(
            "Geometric augmentation must remain disabled pending directional-channel transforms"
        )
    if config["data"].get("use_gfs") is not False:
        raise ValueError("GFS is not supported in strict default M1 configuration")
    return config, hashlib.sha256(raw).hexdigest()


def require_training_handoffs(
    config: dict[str, Any],
    *,
    validity_status: str,
    split_manifest_sha256: str | None = None,
    selected_channel_sha256: str | None = None,
) -> None:
    """Fail closed until source masks, strict channels, and frozen split exist."""
    missing = []
    channel_config = config["channel_schema"]
    if (
        not channel_config.get("strict_approved")
        or not channel_config.get("approval_reference")
        or not channel_config.get("approved_channels_sha256")
    ):
        missing.append("Role 4 strict channel approval")
    elif selected_channel_sha256 != channel_config["approved_channels_sha256"]:
        missing.append("approved channel schema hash verification")
    if validity_status != "SOURCE_MASKED":
        missing.append("Role 1 authoritative source/target validity masks")
    expected_split_hash = config["data"].get("split_manifest_sha256")
    split_path = config["data"].get("split_manifest_path")
    if not split_path or not expected_split_hash:
        missing.append("Role 4 frozen split manifest and hash")
    elif split_manifest_sha256 != expected_split_hash:
        missing.append("split manifest hash verification")
    if missing:
        raise RuntimeError("Training blocked pending: " + "; ".join(missing))


def verify_split_manifest(config: dict[str, Any], *, project_root: str | Path) -> str:
    """Read the frozen Role 4 split file and verify its recorded SHA-256."""
    split_path = config["data"].get("split_manifest_path")
    expected_hash = config["data"].get("split_manifest_sha256")
    if not split_path or not expected_hash:
        raise RuntimeError("Role 4 split manifest path and SHA-256 are not configured")
    manifest = Path(split_path)
    if not manifest.is_absolute():
        manifest = Path(project_root) / manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"Frozen split manifest not found: {manifest}")
    actual_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual_hash.lower() != str(expected_hash).lower():
        raise ValueError("Frozen split manifest SHA-256 does not match the approved config hash")
    return actual_hash


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
