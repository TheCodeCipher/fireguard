"""Validate and register offline OOF artifacts without loading a model.

Run from backend/: python scripts/register_oof_artifacts.py --manifest path/to/manifest.json
Add --publish only after the frozen OOF and evaluation evidence has been approved.
Mock manifests are accepted only in development and can never be published.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Allow direct execution from backend/ without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from fireguard.core.config import get_settings
from fireguard.core.database import SessionLocal
from fireguard.models import ForecastArtifact, ForecastRun, ModelVersion, WildfireEvent
from fireguard.services.artifact_storage import LocalArtifactStorage
from fireguard.services.oof_registry import (
    RegistryValidationError,
    next_artifact_status,
    validate_artifact_spec,
    validate_real_oof_context,
)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryValidationError(f"Could not read JSON manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegistryValidationError("Manifest root must be a JSON object")
    if payload.get("manifest_version") != "1.0":
        raise RegistryValidationError("manifest_version must be '1.0'")
    if not isinstance(payload.get("artifacts"), list) or not payload["artifacts"]:
        raise RegistryValidationError("Manifest must contain a non-empty artifacts array")
    if payload.get("mode", "real") == "mock":
        if not isinstance(payload.get("forecast_run_id"), int) or payload["forecast_run_id"] <= 0:
            raise RegistryValidationError("Mock manifest requires a positive forecast_run_id")
    else:
        required = {
            "event_key", "model_name", "model_version", "architecture", "origin_date",
            "target_date", "horizon_hours", "split_name", "is_oof", "protocol_id", "fold_id",
            "time_safety_status", "channel_manifest_sha256", "checkpoint_sha256", "config_hash",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise RegistryValidationError(f"Real manifest missing run/model fields: {', '.join(missing)}")
        if payload.get("forecast_run_id") is not None and (
            not isinstance(payload["forecast_run_id"], int) or payload["forecast_run_id"] <= 0
        ):
            raise RegistryValidationError("forecast_run_id must be a positive integer when supplied")
    return payload


def resolve_real_run(db, manifest: dict[str, Any]) -> ForecastRun:
    """Resolve or create the event/model/run records for a real OOF manifest."""
    event = db.scalar(select(WildfireEvent).where(WildfireEvent.event_key == manifest["event_key"]))
    if event is None:
        raise RegistryValidationError("event_key is not in the imported WSTS event catalogue")
    if not event.complex_group_id:
        raise RegistryValidationError("WSTS event has no frozen complex_group_id")

    for key in ("model_name", "model_version", "architecture"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise RegistryValidationError(f"{key} must be a non-empty string")
    if not isinstance(manifest.get("horizon_hours"), int) or manifest["horizon_hours"] <= 0:
        raise RegistryValidationError("horizon_hours must be a positive integer")

    model = db.scalar(
        select(ModelVersion).where(
            ModelVersion.name == manifest["model_name"],
            ModelVersion.version == manifest["model_version"],
        )
    )
    if model is None:
        model = ModelVersion(
            name=manifest["model_name"],
            version=manifest["model_version"],
            architecture=manifest["architecture"],
            checkpoint_sha256=manifest["checkpoint_sha256"],
            model_config_json={"config_hash": manifest["config_hash"]},
        )
        db.add(model)
        db.flush()
    else:
        if model.architecture != manifest["architecture"]:
            raise RegistryValidationError("Model architecture does not match registered model version")
        if (model.checkpoint_sha256 or "").lower() != manifest["checkpoint_sha256"].lower():
            raise RegistryValidationError("Model checkpoint hash conflicts with the registered model version")
        if ((model.model_config_json or {}).get("config_hash") or "").lower() != manifest["config_hash"].lower():
            raise RegistryValidationError("Model config hash conflicts with the registered model version")

    try:
        origin = date.fromisoformat(manifest["origin_date"])
        target = date.fromisoformat(manifest["target_date"])
    except (TypeError, ValueError) as exc:
        raise RegistryValidationError("origin_date and target_date must use YYYY-MM-DD") from exc
    if manifest["is_oof"] is not True:
        raise RegistryValidationError("Real artifact manifests must set is_oof=true")
    if manifest["split_name"] not in {"test", "held_out"}:
        raise RegistryValidationError("Real artifact manifests must use split_name=test or held_out")

    if manifest.get("forecast_run_id") is not None:
        run = db.scalar(
            select(ForecastRun)
            .options(joinedload(ForecastRun.event), joinedload(ForecastRun.model_version))
            .where(ForecastRun.id == manifest["forecast_run_id"])
        )
        if run is None:
            raise RegistryValidationError("forecast_run_id does not exist")
    else:
        run = db.scalar(
            select(ForecastRun).where(
                ForecastRun.event_id == event.id,
                ForecastRun.model_version_id == model.id,
                ForecastRun.origin_date == origin,
                ForecastRun.fold == manifest["fold_id"],
            )
        )
        if run is None:
            run = ForecastRun(
                event_id=event.id,
                model_version_id=model.id,
                origin_date=origin,
                target_date=target,
                horizon_hours=manifest["horizon_hours"],
                fold=manifest["fold_id"],
                split_name=manifest["split_name"],
                is_oof=True,
                protocol=manifest["protocol_id"],
                channel_manifest_sha256=manifest["channel_manifest_sha256"],
                time_safety_status=manifest["time_safety_status"],
                metadata_json={"held_out_group_id": event.complex_group_id},
            )
            db.add(run)
            db.flush()

    if run.event_id != event.id or run.model_version_id != model.id:
        raise RegistryValidationError("Forecast run does not match the manifest event/model")
    if run.origin_date != origin or run.target_date != target:
        raise RegistryValidationError("Forecast run origin/target dates do not match the manifest")
    if run.horizon_hours != manifest["horizon_hours"]:
        raise RegistryValidationError("Forecast run horizon_hours does not match the manifest")
    if run.protocol != manifest["protocol_id"] or run.fold != manifest["fold_id"]:
        raise RegistryValidationError("Forecast run protocol/fold does not match the manifest")
    if run.split_name != manifest["split_name"] or run.is_oof is not True:
        raise RegistryValidationError("Forecast run must be an OOF test/held_out record")
    if run.channel_manifest_sha256 != manifest["channel_manifest_sha256"]:
        raise RegistryValidationError("Forecast run channel_manifest_sha256 does not match the manifest")
    if run.time_safety_status != manifest["time_safety_status"]:
        raise RegistryValidationError("Forecast run time_safety_status does not match the manifest")
    return run


def register_manifest(manifest: dict[str, Any], *, publish: bool, dry_run: bool) -> dict[str, Any]:
    settings = get_settings()
    mode = manifest.get("mode", "real")
    if mode not in {"real", "mock"}:
        raise RegistryValidationError("mode must be 'real' or 'mock'")
    if mode == "mock" and settings.app_env.lower() != "development":
        raise RegistryValidationError("Mock manifests are allowed only when FIREGUARD_APP_ENV=development")
    if mode == "mock" and publish:
        raise RegistryValidationError("Mock artifacts can never be published")

    storage = LocalArtifactStorage(settings.artifact_root)
    with SessionLocal() as db:
        if mode == "real":
            run = resolve_real_run(db, manifest)
            validate_real_oof_context(run, manifest)
        else:
            run = db.scalar(
                select(ForecastRun)
                .options(joinedload(ForecastRun.event), joinedload(ForecastRun.model_version))
                .where(ForecastRun.id == manifest["forecast_run_id"])
            )
            if run is None:
                raise RegistryValidationError("forecast_run_id does not exist")
            if run.protocol != "MOCK_ONLY" or not (run.metadata_json or {}).get("mode") == "mock":
                raise RegistryValidationError("Mock manifest must reference an explicitly MOCK_ONLY run")

        validated = [
            validate_artifact_spec(spec, storage, mock=(mode == "mock"))
            for spec in manifest["artifacts"]
        ]
        storage_uris = [item["storage_uri"] for item in validated]
        if len(storage_uris) != len(set(storage_uris)):
            raise RegistryValidationError("Manifest contains duplicate storage_uri values")
        artifact_types = {item["artifact_type"] for item in validated}
        if publish and not {"probability_map", "actual_target"}.issubset(artifact_types):
            raise RegistryValidationError("Publishing requires both probability_map and actual_target artifacts")

        existing_by_uri = {
            item["storage_uri"]: db.scalar(
                select(ForecastArtifact).where(
                    ForecastArtifact.forecast_run_id == run.id,
                    ForecastArtifact.storage_uri == item["storage_uri"],
                )
            )
            for item in validated
        }
        target_states = {
            item["storage_uri"]: next_artifact_status(
                existing_by_uri[item["storage_uri"]], item, mode=mode, publish=publish
            )
            for item in validated
        }
        if len(set(target_states.values())) > 1:
            resulting_state = "mixed"
        else:
            resulting_state = next(iter(target_states.values()))

        if dry_run:
            return {
                "forecast_run_id": run.id,
                "mode": mode,
                "validated_artifacts": len(validated),
                "resulting_state": resulting_state,
                "dry_run": True,
            }

        if mode == "real":
            db.add(run)
        registered_artifacts: list[ForecastArtifact] = []
        for item in validated:
            artifact = existing_by_uri[item["storage_uri"]]
            if artifact is None:
                artifact = ForecastArtifact(
                    forecast_run_id=run.id,
                    **{
                        key: item[key]
                        for key in ("artifact_type", "storage_uri", "checksum_sha256", "content_type")
                    },
                )
                db.add(artifact)
            artifact.status = target_states[item["storage_uri"]]
            if artifact.status != "published" or not (artifact.metadata_json or {}).get("registry_manifest_version"):
                artifact.metadata_json = {
                    "mode": item["mode"],
                    "reveal_class": item["reveal_class"],
                    "file_size_bytes": item["file_size_bytes"],
                    "metadata": item["metadata"],
                    "registry_manifest_version": manifest["manifest_version"],
                }
            else:
                # Published records are immutable; the equality checks above have
                # already confirmed their registered metadata is unchanged.
                pass
            registered_artifacts.append(artifact)

        db.flush()
        registered_ids = [item.id for item in registered_artifacts]
        db.commit()
        return {
            "forecast_run_id": run.id,
            "mode": mode,
            "registered_artifacts": len(validated),
            "artifact_ids": registered_ids,
            "resulting_state": resulting_state,
            "dry_run": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate/register static FireGuard OOF artifacts")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish unchanged artifacts already registered as validated",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate files and metadata without DB writes")
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
        result = register_manifest(manifest, publish=args.publish, dry_run=args.dry_run)
    except RegistryValidationError as exc:
        print(f"REGISTRY VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
