"""Phase 3 data release: file audit, terrain QA and the compact input manifest (Role 1, P3-R1-06).

    python scripts/build_phase3_data_release.py --data-root <folder holding 2018..2021>

Reads every GeoTIFF once (about 40 GB; budget 10-20 minutes on a local SSD) and writes:

* ``data/manifests/wsts_file_checksums.csv``   sha256, grid and band-schema check per TIFF
* ``data/manifests/phase3_input_manifest.csv`` one row per scored sample: ids, group, fold splits,
  input dates, origin, target date, frame size, CRS and transform. **No target values, no target
  file paths** (P3-R1-06).
* ``docs/evidence/role1/terrain_qa.csv``        per event: are slope/aspect/elevation identical on
  every day, value ranges, missing fractions of the strict channels
* ``docs/evidence/role1/phase3_data_release.json`` contract, contract hash, pinned loader commit and
  the SHA-256 of every manifest above, so an experiment can cite one release.

Nothing is written under the data folder. Exit code 1 if any file is unreadable, has the wrong
band schema, or an event's grid is inconsistent (the manifest is still written, marked FAILED).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

from fireguard_ml.data.wsts_dataset import (
    DATA_CONTRACT_VERSION,
    PINNED_LOADER_COMMIT,
    contract_dict,
    contract_hash,
    descriptions_ok,
    scan_events,
)

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests"
EVID = ROOT / "docs" / "evidence" / "role1"
STRICT = ("viirs_m11", "viirs_i2", "viirs_i1", "slope", "aspect", "elevation",
          "binary_active_fire_mask")  # fmt: skip
STRICT_RAW = {"viirs_m11": 0, "viirs_i2": 1, "viirs_i1": 2, "slope": 12, "aspect": 13,
              "elevation": 14}  # fmt: skip
TERRAIN = {"slope": 12, "aspect": 13, "elevation": 14}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def audit(data_root: Path, only: set[str] | None):
    events = scan_events(data_root)
    if only:
        events = {k: v for k, v in events.items() if k in only}
    files, terrain, problems = [], [], []
    t0 = time.time()
    for n, (ev, ef) in enumerate(sorted(events.items()), 1):
        grids, first_terrain, tdiff = set(), None, dict.fromkeys(TERRAIN, 0.0)
        nan_cells = defaultdict(int)
        cells = 0
        af_pos = af_lost = 0
        rng = {k: [np.inf, -np.inf] for k in TERRAIN}
        for d in ef.dates:
            p = ef.path(d)
            raw = p.read_bytes()
            rec = {"year": ef.year, "event_id": ev, "date": d.isoformat(),
                   "relative_path": f"{ef.year}/{ev}/{p.name}", "bytes": len(raw),
                   "sha256": sha256_bytes(raw)}  # fmt: skip
            try:
                with rasterio.open(p) as ds:
                    arr = ds.read().astype(np.float32)
                    rec.update(height=ds.height, width=ds.width, bands=ds.count,
                               crs=ds.crs.to_string() if ds.crs else "",
                               transform=" ".join(f"{v:.6f}" for v in tuple(ds.transform)[:6]),
                               schema_ok=descriptions_ok(ds.descriptions),
                               gfs_precip_band_name=ds.descriptions[17] or "")  # fmt: skip
            except Exception as exc:  # unreadable / truncated
                rec.update(height="", width="", bands="", crs="", transform="", schema_ok=False,
                           gfs_precip_band_name="", error=str(exc)[:200])  # fmt: skip
                problems.append(f"{rec['relative_path']}: unreadable")
                files.append(rec)
                continue
            if not rec["schema_ok"]:
                problems.append(f"{rec['relative_path']}: band schema differs")
            files.append(rec)
            grids.add((rec["height"], rec["width"], rec["crs"], rec["transform"]))
            t = {k: arr[b] for k, b in TERRAIN.items()}
            if first_terrain is None:
                first_terrain = t
            for k, v in t.items():
                both = ~np.isnan(v) & ~np.isnan(first_terrain[k])
                if both.any():
                    tdiff[k] = max(tdiff[k], float(np.abs(v[both] - first_terrain[k][both]).max()))
                if (~np.isnan(v)).any():
                    rng[k] = [
                        min(rng[k][0], float(np.nanmin(v))),
                        max(rng[k][1], float(np.nanmax(v))),
                    ]
            for k, b in STRICT_RAW.items():
                nan_cells[k] += int(np.isnan(arr[b]).sum())
            cells += arr.shape[1] * arr.shape[2]
            af = arr[22]
            af_pos += int((af > 0).sum())
            af_lost += int(((af > 0) & (af < 100)).sum())  # HHMM 00:xx -> hour 0 -> not in mask
        consistent = len(grids) == 1
        if not consistent:
            problems.append(f"{ev}: {len(grids)} different grids")
        terrain.append({
            "year": ef.year, "event_id": ev, "days": len(ef.dates), "grid_consistent": consistent,
            **{f"{k}_max_abs_change_across_days": round(tdiff[k], 6) for k in TERRAIN},
            **{f"{k}_min": round(rng[k][0], 4) for k in TERRAIN},
            **{f"{k}_max": round(rng[k][1], 4) for k in TERRAIN},
            **{f"{k}_nan_fraction": round(nan_cells[k] / max(cells, 1), 6) for k in STRICT_RAW},
            "active_fire_pixels": af_pos, "active_fire_pixels_before_0100_utc": af_lost,
        })  # fmt: skip
        if n % 25 == 0 or n == len(events):
            print(f"  {n}/{len(events)} events ({time.time() - t0:.0f}s)", flush=True)
    return events, files, terrain, problems


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def input_manifest(events, files) -> list[dict]:
    grid = {(r["event_id"]): r for r in files if r.get("transform")}
    with (MAN / "wsts_eval_samples.csv").open(encoding="utf-8") as f:
        samples = list(csv.DictReader(f))
    with (MAN / "p1_loyo_fold_manifest.csv").open(encoding="utf-8") as f:
        p1 = list(csv.DictReader(f))
    split = {(r["event_id"], r["fold_test_year"]): r["split"] for r in p1}
    group = {r["event_id"]: r["group_id"] for r in p1}
    folds = sorted({r["fold_test_year"] for r in p1})
    out = []
    for s in samples:
        ev = s["event_id"]
        if ev not in events:
            continue
        g = grid.get(ev, {})
        out.append({
            "sample_id": f"{ev}__{s['target_date']}",
            "event_id": ev, "event_year": s["event_year"], "group_id": group.get(ev, ""),
            "target_index": s["target_index"],
            "origin_date": s["last_input_date"], "target_date": s["target_date"],
            "max_context_days": 5,
            **{f"split_fold_{y}": split.get((ev, y), "") for y in folds},
            "height": g.get("height", ""), "width": g.get("width", ""),
            "crs": g.get("crs", ""), "transform": g.get("transform", ""),
            "data_contract_version": DATA_CONTRACT_VERSION,
        })  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--events", nargs="*", default=None, help="subset (testing only)")
    a = ap.parse_args()
    events, files, terrain, problems = audit(a.data_root, set(a.events) if a.events else None)
    status = "FAILED" if problems else ("PARTIAL" if a.events else "COMPLETE")
    write_csv(MAN / "wsts_file_checksums.csv", files)
    write_csv(MAN / "phase3_input_manifest.csv", input_manifest(events, files))
    write_csv(EVID / "terrain_qa.csv", terrain)
    hashes = {p.name: sha256_text_file(p) for p in [
        MAN / "wsts_file_checksums.csv", MAN / "phase3_input_manifest.csv", EVID / "terrain_qa.csv",
        MAN / "wsts_eval_samples.csv", MAN / "p1_loyo_fold_manifest.csv"]}  # fmt: skip
    release = {
        "status": status, "events": len(events), "tiffs": len(files),
        "problems": problems[:200], "n_problems": len(problems),
        "pinned_loader_commit": PINNED_LOADER_COMMIT,
        "contract": contract_dict(STRICT, 5), "contract_hash_T5": contract_hash(STRICT, 5),
        "contract_hash_T1": contract_hash(STRICT, 1), "contract_hash_T3": contract_hash(STRICT, 3),
        "file_sha256_lf_normalised": hashes,
        "active_fire_pixels": sum(r["active_fire_pixels"] for r in terrain),
        "active_fire_pixels_lost_from_binary_input_mask": sum(
            r["active_fire_pixels_before_0100_utc"] for r in terrain),
        "terrain_static_events": sum(all(r[f"{k}_max_abs_change_across_days"] == 0
                                         for k in TERRAIN) for r in terrain),
    }  # fmt: skip
    (EVID / "phase3_data_release.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({k: release[k] for k in ("status", "events", "tiffs", "n_problems",
                                              "terrain_static_events")}))  # fmt: skip
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
