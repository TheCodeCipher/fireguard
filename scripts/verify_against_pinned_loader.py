"""Reproduce the pinned WSTS loader's samples with FireGuard's dataset code (Role 1, P3-R1-01).

Runs the real baseline loader (WildfireSpreadTS @ f62dc49, evaluation mode) on every sample of
the events found under --data-root, rebuilds the same tensors with
``fireguard_ml.data.wsts_dataset.process_raw`` plus the loader's own standardisation statistics,
``nan_to_num`` and centre crop, and compares all 40 channels and the target cell by cell.

    python scripts/verify_against_pinned_loader.py --loader-repo <WildfireSpreadTS checkout> \
        --data-root <folder holding 2018..2021> --T 1 5 \
        --out docs/evidence/role1/loader_reproduction.json

The checkout must be at f62dc49 with Role 1's local patch (docs/wsts_baseline_local_patch.diff,
T_co TypeVar) applied; nothing else is changed. Needs torch, torchvision, h5py, rasterio.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

from fireguard_ml.data.wsts_dataset import (
    PINNED_LOADER_COMMIT,
    PROCESSED_CHANNELS,
    PROCESSED_TO_RAW,
    process_raw,
    scan_events,
    target_from_band,
)

STATS_YEARS = (2018, 2019)  # any key of the loader's table; only used to replay its maths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--loader-repo", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--T", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    head = subprocess.run(["git", "-C", str(a.loader_repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()  # fmt: skip
    if head != PINNED_LOADER_COMMIT:
        raise SystemExit(f"loader checkout is at {head or '?'}, expected {PINNED_LOADER_COMMIT}")
    sys.path.insert(0, str(a.loader_repo / "src"))
    fsd = importlib.import_module("dataloader.FireSpreadDataset")
    import rasterio

    events = scan_events(a.data_root)
    years = sorted({e.year for e in events.values()})
    report = {"loader_commit": head, "data_root_events": sorted(events), "runs": []}
    for T in a.T:
        ds = fsd.FireSpreadDataset(
            data_dir=str(a.data_root), included_fire_years=years, n_leading_observations=T,
            crop_side_length=128, load_from_hdf5=False, is_train=False,
            remove_duplicate_features=False, stats_years=STATS_YEARS,
        )  # fmt: skip
        means = np.asarray(ds.means)[0, :, 0, 0]
        stds = np.asarray(ds.stds)[0, :, 0, 0]
        worst, n = np.zeros(40), 0
        y_mismatch = 0
        for i in range(len(ds)):
            year, fire, k = ds.find_image_index_from_dataset_index(i)
            xl, yl = ds[i]
            paths = ds.imgs_per_fire[year][fire][k : k + T + 1]
            raw = []
            for p in paths:
                with rasterio.open(p) as src:
                    raw.append(src.read().astype(np.float32))
            mine = process_raw(np.stack(raw[:-1]))  # [T, 40, H, W]
            for c in range(40):
                r = PROCESSED_TO_RAW[c]
                derived = c == 39 or 16 <= c <= 32
                if not derived:
                    mine[:, c] = (mine[:, c] - means[r]) / stds[r]
            mine = np.nan_to_num(mine, nan=0.0)
            mt = TF.center_crop(torch.from_numpy(mine), list(xl.shape[-2:])).numpy()
            yt = TF.center_crop(
                torch.from_numpy(target_from_band(raw[-1][22]).astype(np.int64)),
                list(yl.shape[-2:]),
            ).numpy()  # fmt: skip
            diff = np.abs(mt - xl.numpy()).max(axis=(0, 2, 3))
            worst = np.maximum(worst, diff)
            y_mismatch += int((yt != yl.numpy()).sum())
            n += 1
        report["runs"].append({
            "T": T, "samples": n,
            "max_abs_diff_per_channel": dict(zip(PROCESSED_CHANNELS, worst.tolist(), strict=True)),
            "max_abs_diff": float(worst.max()), "target_cells_mismatched": y_mismatch,
            "result": "MATCH" if worst.max() < 1e-4 and y_mismatch == 0 else "MISMATCH",
        })  # fmt: skip
        print(T, n, report["runs"][-1]["result"], float(worst.max()), y_mismatch)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
