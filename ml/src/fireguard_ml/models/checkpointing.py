"""Checkpoint save/load that always carries what the offline OOF export needs (Role 3).

A checkpoint is self-describing: it rebuilds the model from its own spec, so exporting predictions never
depends on someone remembering the constructor arguments, channel order or fold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from .temporal_unet import MODEL_REGISTRY, _TemporalBase

REQUIRED_META = ("channels", "T", "preprocessing_version", "config_hash", "fold", "split_manifest_hash")


# The dataset-creation code names these bands "forecast ..." (FirePred.py), the data dictionary calls them
# "gfs_...". Block both spellings so a rename cannot slip a forecast channel into a strict model.
FORBIDDEN_STRICT_SUBSTRINGS = ("gfs", "forecast")


def assert_strict_channels(channels) -> None:
    bad = [c for c in channels if any(s in str(c).lower() for s in FORBIDDEN_STRICT_SUBSTRINGS)]
    if bad:
        raise ValueError(f"GFS / forecast channels are not allowed in strict models: {bad}")


def save_checkpoint(model: _TemporalBase, path: str | Path, meta: Dict[str, object], strict: bool = True) -> None:
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"checkpoint meta is missing: {missing}")
    if len(meta["channels"]) != model.c_strict:
        raise ValueError(f"meta has {len(meta['channels'])} channels but model expects {model.c_strict}")
    if int(meta["T"]) != model.T:
        raise ValueError(f"meta T={meta['T']} but model T={model.T}")
    if strict:
        assert_strict_channels(meta["channels"])
    payload = {"spec": model.spec(), "meta": dict(meta), "strict": strict, "state_dict": model.state_dict()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Tuple[_TemporalBase, Dict[str, object]]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    spec = dict(payload["spec"])
    cls = MODEL_REGISTRY[spec.pop("model_type")]
    model = cls(**spec)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["meta"]
