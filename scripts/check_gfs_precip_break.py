#!/usr/bin/env python3
"""Does the GFS precipitation channel change meaning at 2019-11-07? (Role 3 follow-up to the GFS audit.)

FirePred.py sums the hourly forecast fields for dates <= 2019-11-07 and takes the LAST field after that.
GRIDMET precipitation (band 6, observed, mm/day) is the reference. If the GFS/GRIDMET ratio jumps at the
change date, the channel means different things on either side.

Run from the repo root:
    python scripts/check_gfs_precip_break.py --data-root /path/to/WildfireSpreadTS --every 5

Writes outputs/gfs_precip_break_check.csv and prints the table.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

import numpy as np
import rasterio

BAND_GRIDMET_PR = 6   # original index 5
BAND_GFS_PR = 18      # original index 17
EXPECTED_BANDS = 23
CHANGE_DATE = dt.date(2019, 11, 7)  # dates <= this use the "sum" rule in FirePred.py


def parse(path: Path):
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", path.stem)
    year = next((p for p in path.parts if re.fullmatch(r"20\d\d", p)), None)
    if not m or year is None:
        return None
    return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()

    files = []
    for p in sorted(Path(args.data_root).rglob("*.tif")):
        d = parse(p)
        if d is not None:
            files.append((d, p))
    if not files:
        sys.exit("No dated .tif files found. Expected .../<year>/<event>/<YYYY-MM-DD>.tif")
    files = files[:: max(args.every, 1)]
    print(f"Reading {len(files)} files...")

    rows = {}  # group -> list of (gm_mean, gfs_mean, gfs_max)
    for d, p in files:
        try:
            with rasterio.open(p) as src:
                if src.count != EXPECTED_BANDS:
                    continue
                gm, gfs = src.read([BAND_GRIDMET_PR, BAND_GFS_PR]).astype(np.float32)
        except Exception:  # noqa: BLE001
            continue
        ok = np.isfinite(gm) & np.isfinite(gfs)
        if not ok.any():
            continue
        rec = (float(gm[ok].mean()), float(gfs[ok].mean()), float(gfs[ok].max()))
        regime = "sum rule (<= 2019-11-07)" if d <= CHANGE_DATE else "last rule (> 2019-11-07)"
        for key in (f"year {d.year}", regime):
            rows.setdefault(key, []).append(rec)

    order = sorted(k for k in rows if k.startswith("year")) + sorted(k for k in rows if not k.startswith("year"))
    out_path = Path(args.out_root) / "outputs" / "gfs_precip_break_check.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["group", "n_files", "gridmet_pr_mean_mm", "gfs_pr_mean", "gfs_over_gridmet", "gfs_file_mean_median", "gfs_max_seen"]
    print("\n" + " | ".join(header))
    with open(out_path, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in order:
            a = np.array(rows[k])
            gm_mean, gfs_mean = a[:, 0].mean(), a[:, 1].mean()
            ratio = gfs_mean / gm_mean if gm_mean > 0 else float("nan")
            line = [k, len(a), f"{gm_mean:.3f}", f"{gfs_mean:.3f}", f"{ratio:.2f}", f"{np.median(a[:, 1]):.3f}", f"{a[:, 2].max():.1f}"]
            w.writerow(line)
            print(" | ".join(map(str, line)))
    print(f"\nWrote {out_path}")
    print("Read it like this: if gfs_over_gridmet is roughly stable across the two regimes, the channel is consistent "
          "and the year-to-year differences are weather. If it jumps by several times at the change date, the definition changed.")


if __name__ == "__main__":
    main()
