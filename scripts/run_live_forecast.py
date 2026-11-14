"""Run the live next-day forecast for one origin day and publish it as static files.

    python scripts/run_live_forecast.py --run-dir outputs/m1 --run m1_spatial_unet_T1_seed0 \
        --run-id m1_t1_v1 --artifact-root artifacts --backend-artifact-root backend/artifacts

Offline, like the OOF export: the backend never runs a model. The script first passes the
readiness gate (fireguard_ml/live/readiness.py): all fold checkpoints present, strict, real, on the
frozen split and byte-identical to the exported ``runs/<run-id>/model_version.json``; registry
bundle in ``real`` mode; Gate 2 = BEATS or MATCHES. Nothing is fetched or written before it passes.

It only READS the training outputs (``--run-dir``, ``runs/<run-id>``, ``registry/<run-id>``,
``reports/phase3``) and only WRITES under ``<artifact-root>/live/`` and
``<backend-artifact-root>/live/``.

The origin day defaults to yesterday (UTC), the latest day with complete near-real-time coverage.
FIRMS needs ``FIRMS_MAP_KEY`` in the environment (same variable as the backend), or pass a saved
FIRMS CSV with ``--firms-csv``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml

from fireguard_ml.live.active_fire import (
    CsvDetectionProvider,
    DetectionRasterizer,
    FirmsAreaProvider,
    FirmsError,
    split_time_safe,
)
from fireguard_ml.live.forecast import live_run_id_for, run_live_forecast
from fireguard_ml.live.frames import LiveTemporalSource
from fireguard_ml.live.grid import load_training_domain, regions_for_points
from fireguard_ml.live.readiness import LiveNotReadyError, check_live_readiness
from fireguard_ml.live.reflectance import LinearReflectanceMapping, MissingReflectanceProvider
from fireguard_ml.live.terrain import CachedTerrain, ConstantTerrain, TerrainMosaic

ROOT = Path(__file__).resolve().parents[1]


def build_terrain(cfg: dict, is_mock: bool):
    if is_mock:
        return ConstantTerrain()
    t = cfg["terrain"]
    if t["provider"] == "mosaic":
        return TerrainMosaic(ROOT / t["mosaic_path"])
    if t["provider"] == "earth_engine":
        from fireguard_ml.live.earth_engine import EarthEngineTerrain

        return CachedTerrain(EarthEngineTerrain(), ROOT / t["cache_dir"])
    raise SystemExit(f"unknown terrain.provider {t['provider']!r}")


def build_reflectance(cfg: dict, is_mock: bool):
    r = cfg["reflectance"]
    if is_mock or r["provider"] == "missing":
        return MissingReflectanceProvider()
    if r["provider"] == "earth_engine_v2":
        from fireguard_ml.live.earth_engine import EarthEngineVNP09GA

        mp = ROOT / r["mapping_path"]
        mapping = LinearReflectanceMapping.from_json(mp) if mp.is_file() else None
        return EarthEngineVNP09GA("v2", mapping, int(r.get("day_offset", 0)))
    raise SystemExit(f"unknown reflectance.provider {r['provider']!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True, help="training output, e.g. outputs/m1")
    ap.add_argument("--run", required=True, help="run name, e.g. m1_spatial_unet_T1_seed0")
    ap.add_argument("--run-id", required=True, help="exported model version, e.g. m1_t1_v1")
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--backend-artifact-root", type=Path, default=None,
                    help="publish here (backend artifact root); omit to not publish")
    ap.add_argument("--origin-date", type=date.fromisoformat, default=None,
                    help="forecast origin D (target D+1); default: yesterday UTC")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="default: config")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "live_forecast.yaml")
    ap.add_argument("--bbox", default=None, help="west,south,east,north; default: training domain")
    ap.add_argument("--max-regions", type=int, default=None)
    ap.add_argument("--firms-source", default=None, help="default: config")
    ap.add_argument("--firms-csv", type=Path, default=None,
                    help="saved FIRMS CSV instead of the API")
    ap.add_argument("--device", default=None, help="default: config (cpu)")
    ap.add_argument("--calibrated", action="store_true",
                    help="apply each fold's Platt calibrator before averaging")
    ap.add_argument("--allow-ungated", action="store_true",
                    help="skip the registry/Gate 2 checks; output is stamped UNGATED")
    ap.add_argument("--allow-mock", action="store_true",
                    help="accept synthetic checkpoints ('mock' in --run-id); smoke tests only")
    ap.add_argument("--dry-run", action="store_true", help="readiness and regions only; no writes")
    ap.add_argument("--no-inputs", action="store_true", help="do not keep inputs_raw.tif")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    origin = a.origin_date or (datetime.now(UTC).date() - timedelta(days=1))
    if origin >= datetime.now(UTC).date() and not a.allow_mock:
        raise SystemExit(f"--origin-date {origin} is not a completed UTC day")
    folds = a.folds or cfg["inference"]["folds"]

    try:
        report, bundles = check_live_readiness(
            run_dir=a.run_dir, run=a.run, folds=folds, run_id=a.run_id,
            artifact_root=a.artifact_root, repo_root=ROOT,
            allow_ungated=a.allow_ungated, allow_mock=a.allow_mock,
        )
    except LiveNotReadyError as exc:
        raise SystemExit(f"NOT READY: {exc}") from None
    if report.is_mock:
        print("*** MOCK CHECKPOINTS: synthetic smoke run. Not a forecast. ***")
    if report.gate_status != "GATED":
        print("*** UNGATED: " + "; ".join(report.gate_problems) + " ***")

    grid, reg, af = cfg["grid"], cfg["regions"], cfg["active_fire"]
    domain = load_training_domain(ROOT / grid["domain_manifest"])
    bbox = [float(v) for v in a.bbox.split(",")] if a.bbox else list(domain.bbox_wgs84)
    offset = float(af["day_utc_offset_hours"])
    first_day = origin - timedelta(days=report.T - 1)
    start = first_day - timedelta(days=1 if offset > 0 else 0)
    end = origin + timedelta(days=1 if offset < 0 else 0)
    try:
        if a.firms_csv is not None:
            provider = CsvDetectionProvider(a.firms_csv)
        else:
            provider = FirmsAreaProvider(os.environ.get("FIRMS_MAP_KEY", ""),
                                         a.firms_source or af["firms_source"])
        fetched = provider.detections(bbox, start, end)
    except FirmsError as exc:
        raise SystemExit(f"FIRMS: {exc}") from None
    retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
    dets, dropped_future = split_time_safe(fetched, origin, offset)
    excluded = set(af["exclude_confidence"])
    todays = [d for d in dets if d.day(offset) == origin and d.confidence not in excluded]
    regions, region_report = regions_for_points(
        [d.lon for d in todays], [d.lat for d in todays], domain,
        link_m=float(reg["cluster_link_m"]),
        min_cluster_points=int(reg["min_cluster_detections"]),
        max_regions=a.max_regions or int(reg["max_regions"]),
        half_size_deg=float(grid["half_size_deg"]), pixel_m=float(grid["pixel_m"]),
        crs=grid["crs"],
    )
    live_run_id = live_run_id_for(origin, a.run_id)
    summary = {"live_run_id": live_run_id, "origin_date": origin.isoformat(),
               "gate_status": report.gate_status, "is_mock": report.is_mock,
               "folds": report.folds, "firms_source": provider.source,
               "detections_fetched": len(fetched),
               "detections_after_origin_dropped": dropped_future,
               "detections_on_origin_day": len(todays), "region_report": region_report}
    if a.dry_run:
        print(json.dumps({**summary, "dry_run": True,
                          "region_ids": [r.region_id for r in regions]}, indent=2))
        return

    rasterizer = DetectionRasterizer(
        dets, provider.source, exclude_confidence=tuple(excluded), buffer_m=float(af["buffer_m"]),
        rule=af["rasterize_rule"], utc_offset_hours=offset,
    )
    source = LiveTemporalSource(regions, origin, report.channels, report.T, rasterizer,
                                build_reflectance(cfg, report.is_mock),
                                build_terrain(cfg, report.is_mock), is_mock=report.is_mock)
    thresholds = {str(b.fold): float(b.selection["threshold"]["threshold"]) for b in bundles}
    result = run_live_forecast(
        source=source, bundles=bundles, report=report, artifact_root=a.artifact_root,
        backend_artifact_root=a.backend_artifact_root,
        n_detections=region_report["cluster_sizes"], firms_source=provider.source,
        firms_retrieved_at=retrieved_at,
        extra_run_info={"summary": summary, "threshold_by_fold": thresholds,
                        "threshold_fold_mean": sum(thresholds.values()) / len(thresholds),
                        "active_fire_config": af, "config_file": str(a.config)},
        device=a.device or cfg["inference"]["device"], calibrated=a.calibrated,
        stale_after_hours=int(cfg["publish"]["stale_after_hours"]),
        write_inputs=not a.no_inputs,
    )
    print(json.dumps({**summary, **result}, indent=2))


if __name__ == "__main__":
    main()
