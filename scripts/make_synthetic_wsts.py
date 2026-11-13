"""Write a tiny SYNTHETIC dataset in the exact WSTS file format for the demo (Integration).

    python scripts/make_synthetic_wsts.py --out demo_data/synthetic_wsts

WHY THIS EXISTS
    The real WildfireSpreadTS archive (~40 GB) cannot be shipped with the repository or used in CI,
    but the integrated pipeline has to be demonstrated end to end: Role 1's reader, Role 3's
    trainer and exporter, Role 1's artifact checks, Role 4's OOF validation and reveal package,
    Role 5's registry and API, and Role 6's UI. This script writes GeoTIFFs that every one of those
    steps accepts as WSTS input, so the demo runs the real code paths unchanged.

WHAT IS REAL AND WHAT IS NOT
    Real (copied from committed manifests): event IDs, years, file dates, frame size, CRS
    (EPSG:32610), affine transform, the 23-band layout and band descriptions, and the fold/split
    each event is in.
    Synthetic (generated here): every pixel value. Terrain is a smooth random surface, reflectance
    and weather are smooth random fields, and the active-fire band is a ring that grows from a seed
    point by a fixed amount each day. Nothing in these files resembles a real fire; any metric
    computed on them is meaningless.

HOW IT IS LABELLED (so it can never be mistaken for WSTS)
    * the root folder holds SYNTHETIC_NOT_WSTS.txt; build_backend_registry.py refuses to register
      anything derived from a data root containing it as a real artifact;
    * every GeoTIFF carries the dataset tag FIREGUARD_SYNTHETIC=1 and a description saying so;
    * the folder is written outside the repository by default and is git-ignored if placed in it
      (demo_data/ in .gitignore; *.tif is ignored everywhere anyway).

The default event set is fold 2020 of the frozen P1 manifest: 3 train, 2 validation and 3 test
events with the smallest frames, so the demo trains on a laptop CPU in a few minutes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

# Band descriptions and order are imported from Role 1's reader, so the synthetic files can only
# ever match the schema the reader checks (one source of truth; no second copy of the 23 names).
from fireguard_ml.data.wsts_dataset import RAW_BANDS, RAW_DESCRIPTIONS

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests"
MARKER = "SYNTHETIC_NOT_WSTS.txt"

# Fold 2020 demo events (see module docstring). Chosen by smallest frame among events with 5-6
# scored samples in each split; the split is read from the frozen manifest, never assumed.
DEFAULT_EVENTS = (
    "fire_25294989", "fire_21997854", "fire_21890063",  # fold 2020 train
    "fire_25294983", "fire_25205260",  # fold 2020 validation
    "fire_24461965", "fire_24604426", "fire_24025599",  # fold 2020 test
)  # fmt: skip


def parse_transform(text: str) -> Affine:
    """WSTS metadata stores the affine as rasterio's 3-line pretty print; take the 6 numbers."""
    numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]
    a, b, c, d, e, f = numbers[:6]
    return Affine(a, b, c, d, e, f)


def smooth_field(
    rng: np.random.Generator, h: int, w: int, scale: int = 24
) -> np.ndarray:
    """A smooth random surface in [0, 1]: coarse noise upsampled bilinearly (no SciPy needed)."""
    gh, gw = h // scale + 2, w // scale + 2
    coarse = rng.random((gh, gw))
    ys = np.linspace(0, gh - 1.001, h)
    xs = np.linspace(0, gw - 1.001, w)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    c00 = coarse[y0][:, x0]
    c01 = coarse[y0][:, x0 + 1]
    c10 = coarse[y0 + 1][:, x0]
    c11 = coarse[y0 + 1][:, x0 + 1]
    out = (
        c00 * (1 - fy) * (1 - fx)
        + c01 * (1 - fy) * fx
        + c10 * fy * (1 - fx)
        + c11 * fy * fx
    )
    return out.astype(np.float32)


def event_files(event: dict, rng: np.random.Generator) -> list[tuple[date, np.ndarray]]:
    """All daily 23-band arrays for one event, oldest first."""
    h, w = int(event["height"]), int(event["width"])
    first = date.fromisoformat(event["first_date"])
    n_days = int(event["tiff_count"])

    # Static layers, identical on every day (as in WSTS, where terrain does not change).
    elevation = 500 + 2000 * smooth_field(rng, h, w, 40)  # metres
    gy, gx = np.gradient(elevation, 375.0)  # metres per metre on the 375 m grid
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    # Aspect in degrees clockwise from north, the convention Role 1 recorded for ee.Terrain.aspect.
    aspect = ((np.degrees(np.arctan2(-gx, gy)) + 360) % 360).astype(np.float32)
    land_cover = np.floor(1 + 16.999 * smooth_field(rng, h, w, 30)).astype(np.float32)

    # Fire: a ring that starts at a random point near the frame centre and grows ~2 px per day.
    cy = h / 2 + rng.uniform(-h / 8, h / 8)
    cx = w / 2 + rng.uniform(-w / 8, w / 8)
    yy, xx = np.mgrid[0:h, 0:w]
    distance = np.hypot(yy - cy, xx - cx) + 3 * smooth_field(
        rng, h, w, 12
    )  # ragged edge

    days = []
    for i in range(n_days):
        d = first + timedelta(days=i)
        bands = np.empty((len(RAW_BANDS), h, w), dtype=np.float32)
        burnt = distance < 2.0 * i  # everything inside the front has burnt
        for b in (0, 1, 2):  # VIIRS M11, I2, I1: scaled reflectance, darker where burnt
            bands[b] = 800 + 4000 * smooth_field(rng, h, w) - 600 * burnt
        bands[3] = (
            0.6 * smooth_field(rng, h, w) - 0.3 * burnt
        )  # NDVI (excluded from strict)
        bands[4] = 0.5 * smooth_field(rng, h, w) - 0.2 * burnt  # EVI2 (excluded)
        for b in range(5, 12):  # GRIDMET weather (excluded): smooth, arbitrary ranges
            bands[b] = 10 * smooth_field(rng, h, w, 60)
        bands[7] = 360 * smooth_field(rng, h, w, 60)  # wind direction in degrees
        bands[12], bands[13], bands[14] = slope, aspect, elevation
        bands[15] = -4 + 8 * smooth_field(rng, h, w, 80)  # PDSI (excluded)
        bands[16] = land_cover
        for b in range(17, 22):  # GFS forecast channels (excluded)
            bands[b] = 10 * smooth_field(rng, h, w, 60)
        bands[19] = 360 * smooth_field(rng, h, w, 60)  # GFS wind direction in degrees
        # Active fire: HHMM detection time on the active ring (outer 2.5 px of the front), NaN
        # elsewhere, matching how WSTS stores band 23. 2030 UTC ~ the Suomi-NPP afternoon pass.
        ring = (distance < 2.0 * i + 2.5) & (distance >= 2.0 * i - 0.5)
        bands[22] = np.where(ring, np.float32(2030), np.float32(np.nan))
        days.append((d, bands))
    return days


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="dataset root to create")
    ap.add_argument("--events", nargs="*", default=list(DEFAULT_EVENTS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with (MAN / "wsts_event_metadata.csv").open(encoding="utf-8-sig") as f:
        meta = {r["event_id"]: r for r in csv.DictReader(f)}
    missing = [e for e in a.events if e not in meta]
    if missing:
        raise SystemExit(f"events not in wsts_event_metadata.csv: {missing}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / MARKER).write_text(
        "SYNTHETIC DATA - NOT WILDFIRESPREADTS.\n"
        "Written by scripts/make_synthetic_wsts.py for the FireGuard Prototype demo.\n"
        "Real event IDs, dates and frame geometry; every pixel value is generated.\n"
        "Never report anything computed from this folder as a FireGuard result.\n",
        encoding="utf-8",
    )
    rng = np.random.default_rng(a.seed)
    written = []
    for event_id in a.events:
        ev = meta[event_id]
        folder = a.out / ev["year"] / event_id
        folder.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "width": int(ev["width"]),
            "height": int(ev["height"]),
            "count": len(RAW_BANDS),
            "dtype": "float32",
            "crs": ev["crs"],
            "transform": parse_transform(ev["transform"]),
            "compress": "deflate",
            "predictor": 3,  # floating-point predictor: smooth fields compress well
        }
        for d, bands in event_files(ev, rng):
            path = folder / f"{d.isoformat()}.tif"
            with rasterio.open(path, "w", **profile) as ds:
                ds.write(bands)
                for i, desc in enumerate(RAW_DESCRIPTIONS, start=1):
                    ds.set_band_description(i, desc)
                ds.update_tags(
                    FIREGUARD_SYNTHETIC="1",
                    TIFFTAG_IMAGEDESCRIPTION="SYNTHETIC - not WildfireSpreadTS data",
                )
            written.append(str(path.relative_to(a.out)))
    print(json.dumps({"out": str(a.out), "events": len(a.events), "tiffs": len(written),
                      "label": "SYNTHETIC_NOT_WSTS"}))  # fmt: skip


if __name__ == "__main__":
    main()
