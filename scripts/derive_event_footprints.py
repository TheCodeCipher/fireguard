"""Derive one active-fire footprint per WSTS event (input to the related-fire grouping rule).

Role 1's metadata records ``actual_fire_perimeter_status = UNKNOWN`` and
``fire_footprint_proxy_status = NOT_DERIVED`` for all 607 events, and the frame bounding boxes are
~90 x 115 km image extents, not fire footprints (4,113 bbox pairs overlap). This script builds the
footprint Plan v4 needs from the data itself:

    footprint(event) = every 375 m cell whose active-fire band is > 0 on at least one day

"> 0" matches the official loader's target binarisation (``y = (y > 0)`` after NaN -> 0). All WSTS
grids are EPSG:32610 with 375 m pixels whose edges are multiples of 375 m, so a cell's global id
``row * 100000 + col`` (row = floor(y / 375), col = floor(x / 375) of the cell centre) is identical
across events and footprint overlap is an exact set intersection.

Only the last band of each TIFF is read, nothing is modified, and no raster is written.

Usage (from the repository root, with the scripts extra installed)::

    python scripts/derive_event_footprints.py --data-root <folder that holds 2018..2021>

Outputs (small, committed): ``data/manifests/wsts_event_footprints.csv`` and
``data/manifests/wsts_event_fire_cells.csv.gz``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

try:
    import rasterio
    from pyproj import Transformer
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f'Missing dependency ({exc.name}). Run: pip install -e "ml[scripts]" from the repo root.'
    ) from exc

PIXEL = 375.0
EXPECTED_CRS = "EPSG:32610"
YEARS = ("2018", "2019", "2020", "2021")
FOOTPRINT_COLUMNS = (
    "year",
    "event_id",
    "tiffs_listed",
    "tiffs_read",
    "tiffs_unreadable",
    "unreadable_files",
    "fire_days",
    "fire_first_date",
    "fire_last_date",
    "fire_cell_count",
    "footprint_centroid_x",
    "footprint_centroid_y",
    "footprint_centroid_lon",
    "footprint_centroid_lat",
    "crs",
)


def cell_ids(transform, mask: np.ndarray) -> np.ndarray:
    """Global 375 m cell ids of the True pixels of ``mask`` (row-major raster)."""
    rows, cols = np.nonzero(mask)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    gcol = np.floor(xs / PIXEL).astype(np.int64)
    grow = np.floor(ys / PIXEL).astype(np.int64)
    return grow * 100_000 + gcol


def cell_centres(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (ids % 100_000 + 0.5) * PIXEL, (ids // 100_000 + 0.5) * PIXEL


def process_event(event_dir: Path) -> tuple[dict, np.ndarray]:
    tiffs = sorted(event_dir.glob("*.tif"))
    cells: set[int] = set()
    fire_dates: list[date] = []
    unreadable: list[str] = []
    for tif in tiffs:
        try:
            with rasterio.open(tif) as src:
                if str(src.crs) != EXPECTED_CRS or abs(src.transform.a - PIXEL) > 1e-6:
                    raise ValueError(f"unexpected grid {src.crs} {src.transform.a}")
                af = src.read(src.count)  # active-fire band is the last band
                transform = src.transform
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, none is fatal
            unreadable.append(f"{tif.name}: {type(exc).__name__}")
            continue
        fire = np.isfinite(af) & (af > 0)
        if fire.any():
            fire_dates.append(date.fromisoformat(tif.stem))
            cells.update(cell_ids(transform, fire).tolist())
    ids = np.array(sorted(cells), dtype=np.int64)
    row = {
        "year": event_dir.parent.name,
        "event_id": event_dir.name,
        "tiffs_listed": len(tiffs),
        "tiffs_read": len(tiffs) - len(unreadable),
        "tiffs_unreadable": len(unreadable),
        "unreadable_files": "; ".join(unreadable),
        "fire_days": len(fire_dates),
        "fire_first_date": min(fire_dates).isoformat() if fire_dates else "",
        "fire_last_date": max(fire_dates).isoformat() if fire_dates else "",
        "fire_cell_count": int(ids.size),
        "crs": EXPECTED_CRS,
    }
    if ids.size:
        cx, cy = cell_centres(ids)
        row["footprint_centroid_x"] = round(float(cx.mean()), 1)
        row["footprint_centroid_y"] = round(float(cy.mean()), 1)
    else:
        row["footprint_centroid_x"] = row["footprint_centroid_y"] = ""
    return row, ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="folder that contains the 2018..2021 year folders",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    ap.add_argument(
        "--events",
        nargs="*",
        default=None,
        help="optional subset of event ids (for a quick trial run)",
    )
    args = ap.parse_args(argv)

    event_dirs = [
        d
        for y in YEARS
        if (args.data_root / y).is_dir()
        for d in sorted((args.data_root / y).iterdir())
        if d.is_dir()
    ]
    if args.events:
        event_dirs = [d for d in event_dirs if d.name in set(args.events)]
    if not event_dirs:
        print(f"No event folders found under {args.data_root}", file=sys.stderr)
        return 1

    to_lonlat = Transformer.from_crs(32610, 4326, always_xy=True)
    rows, cell_lines = [], []
    t0 = time.time()
    for k, d in enumerate(event_dirs, 1):
        row, ids = process_event(d)
        if ids.size:
            lon, lat = to_lonlat.transform(row["footprint_centroid_x"], row["footprint_centroid_y"])
            row["footprint_centroid_lon"], row["footprint_centroid_lat"] = (
                round(lon, 6),
                round(lat, 6),
            )
        else:
            row["footprint_centroid_lon"] = row["footprint_centroid_lat"] = ""
        rows.append(row)
        cell_lines.append((row["event_id"], " ".join(map(str, ids.tolist()))))
        if k % 25 == 0 or k == len(event_dirs):
            print(f"  {k}/{len(event_dirs)} events  ({time.time() - t0:.0f}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fp_path = args.out_dir / "wsts_event_footprints.csv"
    with fp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FOOTPRINT_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    cells_path = args.out_dir / "wsts_event_fire_cells.csv.gz"
    # mtime=0 makes the gzip bytes reproducible, so the file hash is stable between runs
    with open(cells_path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(b"event_id,cell_ids\n")
        for ev, ids in cell_lines:
            gz.write(f"{ev},{ids}\n".encode())

    summary = {
        "events": len(rows),
        "events_without_fire": sum(1 for r in rows if not r["fire_cell_count"]),
        "tiffs_listed": sum(r["tiffs_listed"] for r in rows),
        "tiffs_unreadable": sum(r["tiffs_unreadable"] for r in rows),
        "footprints_sha256": hashlib.sha256(fp_path.read_bytes()).hexdigest(),
        "fire_cells_sha256": hashlib.sha256(cells_path.read_bytes()).hexdigest(),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2))
    if summary["tiffs_unreadable"]:
        print(
            "WARNING: some TIFFs could not be read; they are listed per event in "
            f"{fp_path.name}. Re-copy those files and re-run before freezing groups.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
