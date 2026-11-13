"""Load a registry bundle (from scripts/build_backend_registry.py) into the replay database.

Run from backend/ (backend environment, no PyTorch needed):

    python -m alembic upgrade head
    python scripts/load_registry_bundle.py --bundle <registry folder>/registry_bundle.json
    # real bundles only, after the lead approves publication:
    python scripts/load_registry_bundle.py --bundle <...>/registry_bundle.json --publish

INTEGRATION (Prototype). Role 5 built the registry (validation, lifecycle, storage checks) and the
API; Role 4 built the reveal package; scripts/build_backend_registry.py converts one into the
other. This script is the last step: it puts the bundle into the database WITHOUT bypassing any
Role 5 rule, by calling Role 5's own code for everything that matters:

  1. events      -> Role 5's import_catalogue() with the bundle's event_catalogue.csv
  2. artifacts   -> Role 5's register_manifest() (scripts/register_oof_artifacts.py), which runs
                    every check: OOF/held-out run, fold/group/dates, hashes, PASS_LEVEL_A,
                    root-contained paths, restricted/ namespace, checksums, sizes, lifecycle.
                    Real: first call registers as "validated"; only --publish promotes them.
                    Mock: Role 5 requires an existing MOCK_ONLY run, so this script creates that
                    run row first (the same way Role 5's seed_mock_replay.py does) and then
                    registers the files through the mock path, which can never publish.
  3. metrics     -> ReplayMetric rows. Role 5's metrics endpoint serves these rows (after reveal
                    only) but no Role 5 tool wrote them; the values are precomputed offline by
                    the bridge with Role 4's metric code. Rows are replaced per run, so loading
                    the same bundle twice is safe.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Allow direct execution from backend/ without requiring an editable install (as Role 5's scripts).
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "src"))

from sqlalchemy import delete, select  # noqa: E402

from fireguard.core.config import get_settings  # noqa: E402
from fireguard.core.database import SessionLocal  # noqa: E402
from fireguard.models import ForecastRun, ModelVersion, ReplayMetric, WildfireEvent  # noqa: E402
from fireguard.services.oof_registry import RegistryValidationError  # noqa: E402


def _load_script(name: str):
    """Import one of Role 5's CLI scripts as a module so its functions are reused, not copied."""
    spec = importlib.util.spec_from_file_location(name, BACKEND / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


register_cli = _load_script("register_oof_artifacts")
catalogue_cli = _load_script("import_wsts_event_catalogue")

# Fields of a bundle entry that make up a Role 5 real manifest (same list as the bridge writes).
REAL_MANIFEST_KEYS = (
    "event_key", "model_name", "model_version", "architecture", "checkpoint_sha256",
    "config_hash", "protocol_id", "fold_id", "split_name", "is_oof", "origin_date",
    "target_date", "horizon_hours", "held_out_group_id", "split_manifest_sha256",
    "input_manifest_sha256", "data_schema_version", "data_schema_sha256",
    "channel_manifest_sha256", "time_safety_status", "artifacts",
)  # fmt: skip


def ensure_mock_run(entry: dict[str, Any]) -> int:
    """Find or create the MOCK_ONLY event/model/run rows for one mock forecast; return run id.

    Mirrors backend/scripts/seed_mock_replay.py: mock runs carry protocol MOCK_ONLY,
    metadata mode=mock and a non-publishable time-safety value. Provenance hashes from the ML
    side are kept in metadata so the UI can show what produced the (synthetic) forecast.
    """
    with SessionLocal() as db:
        event = db.scalar(
            select(WildfireEvent).where(WildfireEvent.event_key == entry["event_key"])
        )
        if event is None:
            raise RegistryValidationError(f"{entry['event_key']} is not in the event catalogue")
        model = db.scalar(
            select(ModelVersion).where(
                ModelVersion.name == entry["model_name"],
                ModelVersion.version == entry["model_version"],
            )
        )
        if model is None:
            model = ModelVersion(
                name=entry["model_name"],
                version=entry["model_version"],
                architecture=entry["architecture"],
                checkpoint_sha256=entry["checkpoint_sha256"],
                model_config_json={"mode": "mock", "config_hash": entry["config_hash"]},
            )
            db.add(model)
            db.flush()
        origin = date.fromisoformat(entry["origin_date"])
        run = db.scalar(
            select(ForecastRun).where(
                ForecastRun.event_id == event.id,
                ForecastRun.model_version_id == model.id,
                ForecastRun.origin_date == origin,
                ForecastRun.fold == entry["fold_id"],
            )
        )
        if run is None:
            run = ForecastRun(
                event_id=event.id,
                model_version_id=model.id,
                origin_date=origin,
                target_date=date.fromisoformat(entry["target_date"]),
                horizon_hours=entry["horizon_hours"],
                fold=entry["fold_id"],
                split_name="test",
                is_oof=True,
                protocol="MOCK_ONLY",
                channel_manifest_sha256=entry["channel_manifest_sha256"],
                time_safety_status="MOCK_NOT_A_RESEARCH_RESULT",
                metadata_json={
                    "mode": "mock",
                    "held_out_group_id": entry["held_out_group_id"],
                    "source_protocol_id": entry["source_protocol_id"],
                    "split_manifest_sha256": entry["split_manifest_sha256"],
                    "input_manifest_sha256": entry["input_manifest_sha256"],
                    "data_schema_version": entry["data_schema_version"],
                    "data_schema_sha256": entry["data_schema_sha256"],
                    # Allow-listed input metadata (Role 5 INPUT_METADATA_ALLOWLIST).
                    "channel_manifest": ", ".join(entry["channels"]),
                    "level_a_status": "PASS (all strict channels)",
                    "data_provenance": "SYNTHETIC / plumbing run - not a FireGuard result",
                },
            )
            db.add(run)
        db.commit()
        return run.id


def replace_metrics(run_id: int, rows: list[dict[str, Any]]) -> None:
    """Replace the run's precomputed metric rows (idempotent reloads)."""
    with SessionLocal() as db:
        db.execute(delete(ReplayMetric).where(ReplayMetric.forecast_run_id == run_id))
        for r in rows:
            db.add(
                ReplayMetric(
                    forecast_run_id=run_id,
                    metric_name=r["metric_name"],
                    value=float(r["value"]),
                    threshold=r["threshold"],
                    aggregation=r["aggregation"],
                    calibrator_id=r["calibrator_id"],
                )
            )
        db.commit()


def load(bundle_path: Path, *, publish: bool) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    mode = bundle["mode"]
    if mode not in {"mock", "real"}:
        raise RegistryValidationError("bundle mode must be mock or real")
    if mode == "mock" and publish:
        raise RegistryValidationError("mock bundles can never be published")
    settings = get_settings()
    if mode == "mock" and settings.app_env.lower() != "development":
        raise RegistryValidationError("mock bundles load only with FIREGUARD_APP_ENV=development")

    inserted, updated = catalogue_cli.import_catalogue(bundle_path.parent / "event_catalogue.csv")
    results = []
    for entry in bundle["forecasts"]:
        if mode == "real":
            manifest = {"manifest_version": "1.0", "mode": "real"}
            manifest.update({k: entry[k] for k in REAL_MANIFEST_KEYS})
            result = register_cli.register_manifest(manifest, publish=False, dry_run=False)
            if publish:
                result = register_cli.register_manifest(manifest, publish=True, dry_run=False)
        else:
            run_id = ensure_mock_run(entry)
            manifest = {
                "manifest_version": "1.0",
                "mode": "mock",
                "forecast_run_id": run_id,
                "artifacts": entry["artifacts"],
            }
            result = register_cli.register_manifest(manifest, publish=False, dry_run=False)
        replace_metrics(result["forecast_run_id"], entry["metrics"])
        results.append(
            {
                "forecast_id": entry["forecast_id"],
                "forecast_run_id": result["forecast_run_id"],
                "state": result["resulting_state"],
            }
        )
    return {
        "mode": mode,
        "artifact_root": str(settings.artifact_root),
        "events_inserted": inserted,
        "events_updated": updated,
        "forecasts": len(results),
        "states": sorted({r["state"] for r in results}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Load a FireGuard registry bundle")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--publish", action="store_true", help="real bundles only")
    args = parser.parse_args()
    try:
        summary = load(args.bundle, publish=args.publish)
    except RegistryValidationError as exc:
        print(f"REGISTRY VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
