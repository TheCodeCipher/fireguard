from __future__ import annotations

import math
import re
from datetime import timedelta
from typing import Any

from fireguard.core.errors import APIError
from fireguard.models import ForecastArtifact, ForecastRun
from fireguard.services.artifact_storage import LocalArtifactStorage

HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
ALLOWED_PROTOCOLS = {"P1_LOYO", "P2_GEO_HOLDOUT"}

ARTIFACT_REVEAL_CLASS = {
    "probability_map": "pre_reveal",
    "preview": "pre_reveal",
    "uncertainty_map": "pre_reveal",
    "explanation_map": "pre_reveal",
    "provenance": "pre_reveal",
    "actual_target": "post_reveal",
    "metrics": "post_reveal",
}
RASTER_ARTIFACT_TYPES = {
    "probability_map", "preview", "uncertainty_map", "explanation_map", "actual_target"
}
REGISTRY_STATUSES = {"mock", "generated", "validated", "published", "revoked"}


class RegistryValidationError(ValueError):
    """Raised when an offline OOF artifact manifest is unsafe or incomplete."""


def _require_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{key} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise RegistryValidationError(f"{field} must be a 64-character SHA-256 hex digest")
    return value.lower()


def validate_real_oof_context(run: ForecastRun, manifest: dict[str, Any]) -> None:
    """Verify that the manifest describes the existing held-out OOF run exactly."""
    if not run.is_oof or run.split_name not in {"test", "held_out"}:
        raise RegistryValidationError(
            "Only test/held_out runs explicitly marked is_oof=true can be published"
        )
    if run.protocol not in ALLOWED_PROTOCOLS:
        raise RegistryValidationError(f"Unsupported split protocol: {run.protocol}")
    if run.horizon_hours != 24 or run.target_date != run.origin_date + timedelta(days=1):
        raise RegistryValidationError(
            "WSTS replay runs must represent the next calendar day with a 24-hour horizon"
        )

    event = run.event
    model = run.model_version
    if event is None or model is None:
        raise RegistryValidationError("Forecast run must reference an event and model version")
    if not event.complex_group_id:
        raise RegistryValidationError(
            "Event has no frozen complex_group_id; OOF group mapping is unverified"
        )

    expected = {
        "event_key": event.event_key,
        "protocol_id": run.protocol,
        "fold_id": run.fold,
        "origin_date": run.origin_date.isoformat(),
        "target_date": run.target_date.isoformat(),
        "held_out_group_id": event.complex_group_id,
        "split_name": run.split_name,
        "horizon_hours": run.horizon_hours,
    }
    for key, actual in expected.items():
        # INTEGRATION FIX (Prototype): horizon_hours is an integer in the manifest contract and
        # register_oof_artifacts.resolve_real_run rejects anything else, but _require_text only
        # accepts strings, so every real manifest failed here ("horizon_hours must be a non-empty
        # string") and nothing could ever be published. Integers are now compared as text;
        # every other field is still required to be a non-empty string.
        raw = manifest.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool):
            supplied = str(raw)
        else:
            supplied = _require_text(manifest, key)
        if supplied != str(actual):
            raise RegistryValidationError(
                f"{key} mismatch: expected {actual!s}, received {supplied!s}"
            )

    split_hash = manifest.get("split_manifest_sha256")
    _require_sha256(split_hash, "split_manifest_sha256")
    _require_sha256(manifest.get("input_manifest_sha256"), "input_manifest_sha256")
    _require_sha256(manifest.get("data_schema_sha256"), "data_schema_sha256")
    _require_text(manifest, "data_schema_version")

    if model.name != _require_text(manifest, "model_name"):
        raise RegistryValidationError("model_name does not match the registered model version")
    if model.version != _require_text(manifest, "model_version"):
        raise RegistryValidationError("model_version does not match the registered model version")
    if model.architecture != _require_text(manifest, "architecture"):
        raise RegistryValidationError("architecture does not match the registered model version")

    checkpoint_hash = _require_sha256(manifest.get("checkpoint_sha256"), "checkpoint_sha256")
    if not model.checkpoint_sha256 or model.checkpoint_sha256.lower() != checkpoint_hash:
        raise RegistryValidationError(
            "checkpoint_sha256 does not match the registered model version"
        )

    config_hash = _require_sha256(manifest.get("config_hash"), "config_hash")
    stored_config_hash = (model.model_config_json or {}).get("config_hash")
    if not stored_config_hash or stored_config_hash.lower() != config_hash:
        raise RegistryValidationError(
            "config_hash does not match the registered model configuration"
        )

    supplied_channel_hash = manifest.get("channel_manifest_sha256")
    _require_sha256(supplied_channel_hash, "channel_manifest_sha256")
    if (
        run.channel_manifest_sha256
        and run.channel_manifest_sha256.lower() != supplied_channel_hash.lower()
    ):
        raise RegistryValidationError("channel_manifest_sha256 does not match the forecast run")

    time_safety = _require_text(manifest, "time_safety_status")
    if time_safety != "PASS_LEVEL_A" or run.time_safety_status != "PASS_LEVEL_A":
        raise RegistryValidationError(
            "Real replay artifacts require run and manifest time_safety_status=PASS_LEVEL_A"
        )

    run_metadata = dict(run.metadata_json or {})
    provenance_values = {
        "held_out_group_id": event.complex_group_id,
        "split_manifest_sha256": split_hash.lower(),
        "input_manifest_sha256": manifest["input_manifest_sha256"].lower(),
        "data_schema_version": manifest["data_schema_version"],
        "data_schema_sha256": manifest["data_schema_sha256"].lower(),
        "time_safety_status": time_safety,
    }
    for key, expected_value in provenance_values.items():
        stored_value = run_metadata.get(key)
        if stored_value is not None and str(stored_value).lower() != str(expected_value).lower():
            raise RegistryValidationError(
                f"{key} conflicts with the existing forecast-run provenance"
            )
    run_metadata.update(provenance_values)
    run_metadata["registry_manifest_version"] = manifest.get("manifest_version", "1.0")
    run.metadata_json = run_metadata


def next_artifact_status(
    existing: ForecastArtifact | None,
    item: dict[str, Any],
    *,
    mode: str,
    publish: bool,
) -> str:
    """Enforce immutable artifact identity and the validated-to-published transition.

    First registration validates the artifact but cannot publish it in the same
    operation. Publication is a separate, explicit action against an unchanged
    validated record. A published record is idempotent; a revoked record is terminal.
    """
    if existing is None:
        if publish:
            raise RegistryValidationError(
                "Artifacts must be registered as validated before a separate publish operation"
            )
        return "mock" if mode == "mock" else "validated"

    if existing.status == "revoked":
        raise RegistryValidationError("Revoked artifacts cannot be re-registered or republished")

    if existing.artifact_type != item["artifact_type"]:
        raise RegistryValidationError("Artifact type is immutable for an existing storage_uri")
    if (existing.checksum_sha256 or "").lower() != item["checksum_sha256"].lower():
        raise RegistryValidationError("Artifact checksum is immutable for an existing storage_uri")
    if existing.content_type != item["content_type"]:
        raise RegistryValidationError(
            "Artifact content_type is immutable for an existing storage_uri"
        )

    stored = existing.metadata_json or {}
    expected_metadata = {
        "mode": item["mode"],
        "reveal_class": item["reveal_class"],
        "file_size_bytes": item["file_size_bytes"],
        "metadata": item["metadata"],
    }
    if existing.status != "generated":
        for key, value in expected_metadata.items():
            if stored.get(key) != value:
                raise RegistryValidationError(
                    f"Artifact metadata is immutable after validation (mismatch: {key})"
                )

    if mode == "mock":
        if existing.status != "mock" or stored.get("mode") != "mock":
            raise RegistryValidationError(
                "Mock manifests cannot modify real or unvalidated artifacts"
            )
        if publish:
            raise RegistryValidationError("Mock artifacts can never be published")
        return "mock"

    if existing.status == "mock" or stored.get("mode") == "mock":
        raise RegistryValidationError("Real manifests cannot modify mock artifacts")

    if publish:
        if existing.status not in {"validated", "published"}:
            raise RegistryValidationError(
                "Only an unchanged validated artifact can transition to published"
            )
        return "published"

    if existing.status == "published":
        return "published"
    if existing.status not in {"generated", "validated"}:
        raise RegistryValidationError(f"Unsupported artifact lifecycle state: {existing.status}")
    return "validated"


def validate_artifact_spec(
    spec: dict[str, Any], storage: LocalArtifactStorage, *, mock: bool = False
) -> dict[str, Any]:
    """Validate one artifact record and return computed checksum/size metadata."""
    artifact_type = _require_text(spec, "artifact_type")
    if artifact_type not in ARTIFACT_REVEAL_CLASS:
        raise RegistryValidationError(f"Unsupported artifact_type: {artifact_type}")

    expected_reveal = ARTIFACT_REVEAL_CLASS[artifact_type]
    reveal_class = _require_text(spec, "reveal_class")
    if reveal_class != expected_reveal:
        raise RegistryValidationError(
            f"{artifact_type} must use reveal_class={expected_reveal}, not {reveal_class}"
        )

    storage_uri = _require_text(spec, "storage_uri")
    checksum = _require_sha256(spec.get("checksum_sha256"), "checksum_sha256")
    content_type = _require_text(spec, "content_type")
    try:
        inspected = storage.inspect(storage_uri)
    except APIError as exc:
        raise RegistryValidationError(f"Artifact storage validation failed: {exc.code}") from exc
    if inspected["sha256"].lower() != checksum:
        raise RegistryValidationError(f"Checksum mismatch for {storage_uri}")

    supplied_size = spec.get("file_size_bytes")
    if supplied_size is not None and supplied_size != inspected["file_size_bytes"]:
        raise RegistryValidationError(f"file_size_bytes mismatch for {storage_uri}")

    if not mock:
        if expected_reveal == "post_reveal" and not storage_uri.startswith("restricted/"):
            raise RegistryValidationError("Post-reveal artifacts must be stored under restricted/")
        if expected_reveal == "pre_reveal" and storage_uri.startswith("restricted/"):
            raise RegistryValidationError("Pre-reveal artifacts cannot be stored under restricted/")

    metadata = spec.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise RegistryValidationError("artifact metadata must be a JSON object")
    if mock:
        if metadata.get("mode") != "mock":
            raise RegistryValidationError("Mock artifacts must set metadata.mode='mock'")
    elif metadata.get("mode") == "mock":
        raise RegistryValidationError(
            "Mock-marked artifacts cannot be registered as real artifacts"
        )

    if artifact_type in RASTER_ARTIFACT_TYPES:
        required = {
            "format", "crs", "affine_transform", "width", "height", "nodata", "valid_extent"
        }
        missing = sorted(required - metadata.keys())
        if missing:
            raise RegistryValidationError(
                f"Raster metadata missing required fields: {', '.join(missing)}"
            )
        if not isinstance(metadata["width"], int) or metadata["width"] <= 0:
            raise RegistryValidationError("Raster width must be a positive integer")
        if not isinstance(metadata["height"], int) or metadata["height"] <= 0:
            raise RegistryValidationError("Raster height must be a positive integer")
        transform = metadata["affine_transform"]
        if (
            not isinstance(transform, list)
            or len(transform) != 6
            or any(
                not isinstance(item, (int, float)) or not math.isfinite(item)
                for item in transform
            )
        ):
            raise RegistryValidationError("affine_transform must contain six numeric coefficients")
        extent = metadata["valid_extent"]
        if (
            not isinstance(extent, list)
            or len(extent) != 4
            or any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in extent)
        ):
            raise RegistryValidationError("valid_extent must be [xmin, ymin, xmax, ymax]")
        if (
            extent[0] < 0
            or extent[1] < 0
            or extent[2] > metadata["width"]
            or extent[3] > metadata["height"]
        ):
            raise RegistryValidationError("valid_extent must lie within raster pixel coordinates")
        _require_text(metadata, "format")
        _require_text(metadata, "crs")
    else:
        _require_text(metadata, "format")

    return {
        "artifact_type": artifact_type,
        "storage_uri": storage_uri,
        "checksum_sha256": checksum,
        "file_size_bytes": int(inspected["file_size_bytes"]),
        "content_type": content_type,
        "reveal_class": reveal_class,
        "metadata": metadata,
        "mode": "mock" if mock else "real",
    }
