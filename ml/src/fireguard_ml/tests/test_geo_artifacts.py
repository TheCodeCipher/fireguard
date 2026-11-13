"""Role 1 Phase 4: artifact alignment checks and the pre-reveal replay catalogue."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine  # noqa: E402

from fireguard_ml.data.geo_artifacts import (  # noqa: E402
    CATALOGUE_FIELDS,
    FORBIDDEN,
    ArtifactMismatch,
    Frame,
    build_catalogue,
    verify_artifact,
)

TR = (375.0, 0.0, 697500.0, 0.0, -375.0, 4136625.0)
FRAME = Frame(30, 20, "EPSG:32610", TR)


def write(path, arr, transform=TR, crs="EPSG:32610", dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, transform=Affine(*transform), crs=crs) as d:  # fmt: skip
        d.write(arr.astype(dtype), 1)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prob(h=30, w=20):
    return np.random.default_rng(0).uniform(0, 1, (h, w))


def test_aligned_probability_raster_passes(tmp_path):
    sha = write(tmp_path / "p.tif", prob())
    verify_artifact(tmp_path / "p.tif", FRAME, sha)


@pytest.mark.parametrize(
    "kwargs, arr, msg",
    [
        ({}, prob(32, 32), "frame"),  # padded, not unpadded
        ({}, prob(28, 20), "frame"),  # cropped
        ({"transform": (375.0, 0.0, 697875.0, 0.0, -375.0, 4136625.0)}, prob(), "transform"),
        ({"crs": "EPSG:4326"}, prob(), "CRS"),
        ({}, prob() + 1.0, "outside"),
    ],
)
def test_misaligned_or_invalid_rasters_are_rejected(tmp_path, kwargs, arr, msg):
    write(tmp_path / "p.tif", arr, **kwargs)
    with pytest.raises(ArtifactMismatch, match=msg):
        verify_artifact(tmp_path / "p.tif", FRAME)


def test_checksum_is_enforced(tmp_path):
    write(tmp_path / "p.tif", prob())
    with pytest.raises(ArtifactMismatch, match="checksum"):
        verify_artifact(tmp_path / "p.tif", FRAME, "0" * 64)


def test_catalogue_keeps_only_validated_oof_forecasts_and_no_target_fields(tmp_path):
    root = tmp_path / "art"
    recs = []
    for i, (ev, ok) in enumerate([("e1", True), ("e2", False), ("e3", True)]):
        uri = f"runs/r/{ev}/2020-08-0{i + 1}/probability.tif"
        sha = write(root / uri, prob() if ok else prob(32, 32))
        recs.append({"event_id": ev, "group_id": f"G{i}", "protocol": "P1_LOYO", "fold": "2020",
                     "split_name": "test", "is_oof": True, "origin_date": f"2020-08-0{i + 1}",
                     "target_date": f"2020-08-0{i + 2}", "horizon_hours": 24,
                     "model_version": "r", "storage_uri": uri, "checksum_sha256": sha})  # fmt: skip
    recs.append({**recs[0], "is_oof": False, "origin_date": "2020-08-09"})
    jl = tmp_path / "oof.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in recs))
    rows, rejected = build_catalogue(jl, root, {"e1": FRAME, "e2": FRAME, "e3": FRAME})
    assert [r["event_id"] for r in rows] == ["e1", "e3"]
    assert {r[0] for r in rejected} == {"e1", "e2"} and len(rejected) == 2
    for r in rows:
        assert tuple(r) == CATALOGUE_FIELDS and not any(f in r for f in FORBIDDEN)
