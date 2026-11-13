"""Pre-reveal replay catalogue of events/origins with a validated OOF artifact (Role 1, P4-R1-01).

    python scripts/build_replay_catalogue.py --artifact-root artifacts --run-id m3_t5_v1

Reads ``<artifact-root>/runs/<run-id>/oof_forecasts.jsonl`` (Role 3's export), checks every
probability raster against its source frame from ``data/manifests/phase3_input_manifest.csv``
(size, CRS, transform, dtype, value range, checksum) and writes
``<artifact-root>/runs/<run-id>/replay_catalogue.csv`` (+ ``replay_catalogue_rejected.csv``).
The catalogue holds no target path and no target value.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fireguard_ml.data.geo_artifacts import (
    CATALOGUE_FIELDS,
    build_catalogue,
    frame_from_manifest_row,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--input-manifest", type=Path,
                    default=ROOT / "data" / "manifests" / "phase3_input_manifest.csv")  # fmt: skip
    a = ap.parse_args()
    with a.input_manifest.open(encoding="utf-8") as f:
        frames = {r["event_id"]: frame_from_manifest_row(r) for r in csv.DictReader(f)}
    run = a.artifact_root / "runs" / a.run_id
    rows, rejected = build_catalogue(run / "oof_forecasts.jsonl", a.artifact_root, frames)
    with (run / "replay_catalogue.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CATALOGUE_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with (run / "replay_catalogue_rejected.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["event_id", "origin_date", "reason"])
        w.writerows(rejected)
    print(f"{len(rows)} forecasts catalogued, {len(rejected)} rejected")
    raise SystemExit(1 if rejected else 0)


if __name__ == "__main__":
    main()
