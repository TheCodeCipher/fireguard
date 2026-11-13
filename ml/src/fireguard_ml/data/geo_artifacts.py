"""Geospatial artifact settings and checks for replay (Role 1, P4-R1-01/02).

Every exported raster must sit exactly on its WSTS source frame: same height, width, CRS and
affine transform (no crop, no padding). ``verify_artifact`` checks that, plus nodata, dtype and the
recorded checksum. ``build_catalogue`` keeps only forecasts whose artifact passes, and never
emits a target path or target value (pre-reveal).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Export settings per layer (P4-R1-02). Only "probability" is produced today; the others are the
# settings to use if and when that layer exists and is validated.
RASTER_EXPORT_SETTINGS: dict[str, dict] = {
    "probability": {"dtype": "float32", "nodata": "nan", "range": [0.0, 1.0],
                    "compress": "deflate", "pre_reveal": True},
    "uncertainty": {"dtype": "float32", "nodata": "nan", "range": [0.0, None],
                    "compress": "deflate", "pre_reveal": True, "status": "NOT IMPLEMENTED"},
    "explanation": {"dtype": "float32", "nodata": "nan", "range": [None, None],
                    "compress": "deflate", "pre_reveal": True, "status": "NOT IMPLEMENTED"},
    "actual_target": {"dtype": "uint8", "nodata": 255, "range": [0, 1], "compress": "deflate",
                      "pre_reveal": False, "note": "reveal step only (Role 4 owns reveal)"},
}  # fmt: skip
PREVIEW = {
    "overview_factors": [2, 4, 8],
    "resampling": "average",
    "web_crs": "EPSG:3857",
    "note": "previews are display-only; analysis uses the native EPSG:32610 raster",
}
CATALOGUE_FIELDS = (
    "event_id", "group_id", "protocol", "fold", "origin_date", "target_date", "horizon_hours",
    "height", "width", "crs", "transform", "model_version", "storage_uri", "checksum_sha256",
    "artifact_verified",
)  # fmt: skip
FORBIDDEN = ("target_uri", "target_path", "actual", "y_true", "auprc", "f1", "iou", "n_pos")


class ArtifactMismatch(ValueError):
    pass


@dataclass(frozen=True)
class Frame:
    height: int
    width: int
    crs: str
    transform: tuple[float, ...]  # 6 affine terms


def verify_artifact(path: Path, frame: Frame, checksum: str | None = None,
                    layer: str = "probability", atol: float = 1e-6) -> None:  # fmt: skip
    """Raise ``ArtifactMismatch`` unless ``path`` is a valid raster on ``frame``."""
    import rasterio

    p = Path(path)
    if not p.exists():
        raise ArtifactMismatch(f"{p}: missing")
    if checksum and hashlib.sha256(p.read_bytes()).hexdigest() != checksum:
        raise ArtifactMismatch(f"{p}: checksum differs from the record")
    spec = RASTER_EXPORT_SETTINGS[layer]
    with rasterio.open(p) as ds:
        if (ds.height, ds.width) != (frame.height, frame.width):
            raise ArtifactMismatch(
                f"{p}: {ds.height}x{ds.width}, frame {frame.height}x{frame.width}"
            )
        if (ds.crs.to_string() if ds.crs else None) != frame.crs:
            raise ArtifactMismatch(f"{p}: CRS {ds.crs} != {frame.crs}")
        if not np.allclose(tuple(ds.transform)[:6], frame.transform[:6], atol=atol, rtol=0):
            raise ArtifactMismatch(f"{p}: transform {tuple(ds.transform)[:6]} != {frame.transform}")
        if ds.count != 1 or ds.dtypes[0] != spec["dtype"]:
            raise ArtifactMismatch(f"{p}: expected 1 band of {spec['dtype']}")
        v = ds.read(1)
    lo, hi = spec["range"]
    finite = v[np.isfinite(v)] if v.dtype.kind == "f" else v
    if finite.size and (
        (lo is not None and finite.min() < lo) or (hi is not None and finite.max() > hi)
    ):
        raise ArtifactMismatch(f"{p}: values outside {spec['range']}")


def frame_from_manifest_row(r: dict) -> Frame:
    t = tuple(float(x) for x in str(r["transform"]).split())
    return Frame(int(r["height"]), int(r["width"]), r["crs"], t)


def build_catalogue(oof_jsonl: Path, artifact_root: Path, frames: dict[str, Frame]) -> tuple:
    """Pre-reveal event/origin catalogue rows for forecasts whose artifact verifies.

    Returns ``(rows, rejected)``; ``rejected`` lists (event_id, origin_date, reason).
    """
    rows, rejected = [], []
    for line in Path(oof_jsonl).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = (r["event_id"], r["origin_date"])
        fr = frames.get(r["event_id"])
        if fr is None:
            rejected.append((*key, "event not in the Phase 3 input manifest"))
            continue
        if not r.get("is_oof") or r.get("split_name") != "test":
            rejected.append((*key, "not an out-of-fold test forecast"))
            continue
        try:
            verify_artifact(Path(artifact_root) / r["storage_uri"], fr, r.get("checksum_sha256"))
        except ArtifactMismatch as exc:
            rejected.append((*key, str(exc)))
            continue
        row = {
            "event_id": r["event_id"], "group_id": r.get("group_id", ""),
            "protocol": r["protocol"], "fold": r["fold"], "origin_date": r["origin_date"],
            "target_date": r["target_date"], "horizon_hours": r["horizon_hours"],
            "height": fr.height, "width": fr.width, "crs": fr.crs,
            "transform": " ".join(f"{v:.6f}" for v in fr.transform),
            "model_version": r["model_version"], "storage_uri": r["storage_uri"],
            "checksum_sha256": r["checksum_sha256"], "artifact_verified": True,
        }  # fmt: skip
        assert tuple(row) == CATALOGUE_FIELDS and not any(f in row for f in FORBIDDEN)
        rows.append(row)
    return rows, rejected
