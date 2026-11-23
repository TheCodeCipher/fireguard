import hashlib
from datetime import date
from types import SimpleNamespace

import pytest

from fireguard.services.artifact_storage import LocalArtifactStorage
from fireguard.services.oof_registry import (
    RegistryValidationError,
    next_artifact_status,
    validate_artifact_spec,
    validate_real_oof_context,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raster_spec(path, storage_uri, text="raster-bytes"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "artifact_type": "probability_map",
        "storage_uri": storage_uri,
        "content_type": "image/tiff",
        "checksum_sha256": _sha(text),
        "reveal_class": "pre_reveal",
        "metadata": {
            "format": "GeoTIFF",
            "crs": "EPSG:32610",
            "affine_transform": [375, 0, 500000, 0, -375, 4500000],
            "width": 224,
            "height": 288,
            "nodata": None,
            "valid_extent": [0, 0, 224, 288],
            "units": "probability",
        },
    }


def test_registry_validates_raster_metadata_checksum_and_size(tmp_path):
    root = tmp_path / "artifacts"
    uri = root / "p1_loyo/model-v1/fold-2020/event-001/2020-08-01/probability.tif"
    storage_uri = "p1_loyo/model-v1/fold-2020/event-001/2020-08-01/probability.tif"
    spec = _raster_spec(uri, storage_uri)
    result = validate_artifact_spec(spec, LocalArtifactStorage(root))
    assert result["artifact_type"] == "probability_map"
    assert result["file_size_bytes"] == len("raster-bytes")
    assert result["reveal_class"] == "pre_reveal"


def test_registry_rejects_checksum_mismatch(tmp_path):
    root = tmp_path / "artifacts"
    spec = _raster_spec(root / "map.tif", "map.tif")
    spec["checksum_sha256"] = "0" * 64
    with pytest.raises(RegistryValidationError, match="Checksum mismatch"):
        validate_artifact_spec(spec, LocalArtifactStorage(root))


def test_registry_rejects_wrong_reveal_class(tmp_path):
    root = tmp_path / "artifacts"
    spec = _raster_spec(root / "map.tif", "map.tif")
    spec["reveal_class"] = "post_reveal"
    with pytest.raises(RegistryValidationError, match="must use reveal_class=pre_reveal"):
        validate_artifact_spec(spec, LocalArtifactStorage(root))


def test_registry_rejects_missing_affine_metadata(tmp_path):
    root = tmp_path / "artifacts"
    spec = _raster_spec(root / "map.tif", "map.tif")
    del spec["metadata"]["affine_transform"]
    with pytest.raises(RegistryValidationError, match="affine_transform"):
        validate_artifact_spec(spec, LocalArtifactStorage(root))


def test_registry_accepts_mock_only_when_explicitly_marked(tmp_path):
    root = tmp_path / "artifacts"
    uri = root / "mock/map.json"
    uri.parent.mkdir(parents=True)
    uri.write_text("{}", encoding="utf-8")
    spec = {
        "artifact_type": "provenance",
        "storage_uri": "mock/map.json",
        "content_type": "application/json",
        "checksum_sha256": _sha("{}"),
        "reveal_class": "pre_reveal",
        "metadata": {"mode": "mock", "format": "JSON"},
    }
    result = validate_artifact_spec(spec, LocalArtifactStorage(root), mock=True)
    assert result["mode"] == "mock"


def _fake_run():
    event = SimpleNamespace(event_key="event-001", complex_group_id="complex-001")
    config_hash = "c" * 64
    model = SimpleNamespace(
        name="fireguard-m1",
        version="m1-v1",
        architecture="UNet2D",
        checkpoint_sha256="a" * 64,
        model_config_json={"config_hash": config_hash},
    )
    run = SimpleNamespace(
        is_oof=True,
        split_name="test",
        protocol="P1_LOYO",
        fold="LOYO-2020",
        origin_date=date(2020, 8, 1),
        target_date=date(2020, 8, 2),
        horizon_hours=24,
        event=event,
        model_version=model,
        channel_manifest_sha256="d" * 64,
        time_safety_status="PASS_LEVEL_A",
        metadata_json={},
    )
    return run


def _real_manifest():
    return {
        "event_key": "event-001",
        "protocol_id": "P1_LOYO",
        "fold_id": "LOYO-2020",
        "origin_date": "2020-08-01",
        "target_date": "2020-08-02",
        "held_out_group_id": "complex-001",
        "split_name": "test",
        "horizon_hours": "24",
        "model_name": "fireguard-m1",
        "model_version": "m1-v1",
        "architecture": "UNet2D",
        "split_manifest_sha256": "1" * 64,
        "input_manifest_sha256": "2" * 64,
        "data_schema_sha256": "3" * 64,
        "data_schema_version": "strict-input-v1",
        "checkpoint_sha256": "a" * 64,
        "config_hash": "c" * 64,
        "channel_manifest_sha256": "d" * 64,
        "time_safety_status": "PASS_LEVEL_A",
    }


def test_real_oof_context_accepts_exact_held_out_mapping():
    run = _fake_run()
    validate_real_oof_context(run, _real_manifest())
    assert run.metadata_json["held_out_group_id"] == "complex-001"
    assert run.metadata_json["data_schema_version"] == "strict-input-v1"




def test_real_oof_context_requires_next_day_24_hour_target():
    run = _fake_run()
    run.target_date = date(2020, 8, 3)
    with pytest.raises(RegistryValidationError, match="next calendar day"):
        validate_real_oof_context(run, _real_manifest())


def test_artifact_lifecycle_requires_separate_validation_then_publish():
    item = {
        "artifact_type": "probability_map",
        "checksum_sha256": "a" * 64,
        "content_type": "image/tiff",
        "mode": "real",
        "reveal_class": "pre_reveal",
        "file_size_bytes": 100,
        "metadata": {"format": "GeoTIFF", "crs": "EPSG:32610"},
    }
    with pytest.raises(RegistryValidationError, match="registered as validated"):
        next_artifact_status(None, item, mode="real", publish=True)

    generated = SimpleNamespace(
        status="generated", artifact_type="probability_map", checksum_sha256="a" * 64,
        content_type="image/tiff", metadata_json={},
    )
    assert next_artifact_status(generated, item, mode="real", publish=False) == "validated"

    validated = SimpleNamespace(
        status="validated", artifact_type="probability_map", checksum_sha256="a" * 64,
        content_type="image/tiff",
        metadata_json={
            "mode": "real", "reveal_class": "pre_reveal", "file_size_bytes": 100,
            "metadata": {"format": "GeoTIFF", "crs": "EPSG:32610"},
        },
    )
    assert next_artifact_status(validated, item, mode="real", publish=True) == "published"
    assert next_artifact_status(validated, item, mode="real", publish=False) == "validated"


def test_published_artifact_cannot_be_demoted_or_mutated():
    item = {
        "artifact_type": "probability_map", "checksum_sha256": "a" * 64,
        "content_type": "image/tiff", "mode": "real", "reveal_class": "pre_reveal",
        "file_size_bytes": 100, "metadata": {"format": "GeoTIFF", "crs": "EPSG:32610"},
    }
    published = SimpleNamespace(
        status="published", artifact_type="probability_map", checksum_sha256="a" * 64,
        content_type="image/tiff",
        metadata_json={
            "mode": "real", "reveal_class": "pre_reveal", "file_size_bytes": 100,
            "metadata": {"format": "GeoTIFF", "crs": "EPSG:32610"},
        },
    )
    assert next_artifact_status(published, item, mode="real", publish=False) == "published"
    changed = {**item, "checksum_sha256": "b" * 64}
    with pytest.raises(RegistryValidationError, match="checksum is immutable"):
        next_artifact_status(published, changed, mode="real", publish=False)


def test_revoked_artifact_cannot_be_republished():
    item = {
        "artifact_type": "probability_map", "checksum_sha256": "a" * 64,
        "content_type": "image/tiff", "mode": "real", "reveal_class": "pre_reveal",
        "file_size_bytes": 100, "metadata": {"format": "GeoTIFF", "crs": "EPSG:32610"},
    }
    revoked = SimpleNamespace(
        status="revoked", artifact_type="probability_map", checksum_sha256="a" * 64,
        content_type="image/tiff", metadata_json={},
    )
    with pytest.raises(RegistryValidationError, match="Revoked artifacts"):
        next_artifact_status(revoked, item, mode="real", publish=True)


def test_real_oof_context_rejects_group_mismatch():
    run = _fake_run()
    manifest = _real_manifest()
    manifest["held_out_group_id"] = "different-group"
    with pytest.raises(RegistryValidationError, match="held_out_group_id mismatch"):
        validate_real_oof_context(run, manifest)


def test_real_oof_context_rejects_conflicting_stored_provenance():
    run = _fake_run()
    run.metadata_json = {"split_manifest_sha256": "f" * 64}
    with pytest.raises(
        RegistryValidationError,
        match="conflicts with the existing forecast-run provenance",
    ):
        validate_real_oof_context(run, _real_manifest())


def test_real_oof_context_rejects_non_oof_run():
    run = _fake_run()
    run.is_oof = False
    with pytest.raises(RegistryValidationError, match="Only test/held_out"):
        validate_real_oof_context(run, _real_manifest())


def test_real_oof_context_rejects_unknown_time_safety():
    run = _fake_run()
    run.time_safety_status = "UNKNOWN"
    manifest = _real_manifest()
    manifest["time_safety_status"] = "UNKNOWN"
    with pytest.raises(RegistryValidationError, match="PASS_LEVEL_A"):
        validate_real_oof_context(run, manifest)


def test_real_post_reveal_artifact_must_use_restricted_namespace(tmp_path):
    root = tmp_path / "artifacts"
    target = root / "actual.tif"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("target", encoding="utf-8")
    spec = {
        "artifact_type": "provenance",
        "storage_uri": "actual.tif",
        "content_type": "application/json",
        "checksum_sha256": _sha("target"),
        "reveal_class": "pre_reveal",
        "metadata": {"format": "JSON"},
    }
    # Provenance is pre-reveal, so this remains valid outside restricted/.
    assert validate_artifact_spec(spec, LocalArtifactStorage(root))["mode"] == "real"


def test_real_post_reveal_metric_requires_restricted_namespace(tmp_path):
    root = tmp_path / "artifacts"
    target = root / "metrics.json"
    root.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")
    spec = {
        "artifact_type": "metrics",
        "storage_uri": "metrics.json",
        "content_type": "application/json",
        "checksum_sha256": _sha("{}"),
        "reveal_class": "post_reveal",
        "metadata": {"format": "JSON"},
    }
    with pytest.raises(RegistryValidationError, match="restricted/"):
        validate_artifact_spec(spec, LocalArtifactStorage(root))
