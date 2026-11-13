#!/usr/bin/env python3
"""Shape smoke test of M3 and M4 on ONE REAL WSTS sample (Role 3, task P2-R3-02).

Run from the repo root:
    pip install -e "ml[dev,scripts]"          # once, from the repo root
    python scripts/smoke_temporal_real_sample.py --data-root /path/to/WildfireSpreadTS --T 5

What it proves: real frames, in file-date order, go through both models and come out as one logit per
cell at the original H x W. Evidence is written to outputs/temporal_real_sample_smoke.json.

What it does NOT do: it is not the training pipeline. Frames are the raw 23 GeoTIFF bands with the five
GFS bands dropped (18 bands), NaN set to 0, no normalisation and no land-cover one-hot. Weights are random.
The real input builder is Role 1's loader and the real padding utility is Role 2's; the bottom/right
zero-padding below is a stand-in for this test only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

try:
    from fireguard_ml.models.temporal_unet import ConvLSTMUNet, StackedDaysUNet
except ModuleNotFoundError as exc:  # the package is not installed in this interpreter
    raise SystemExit(
        f"cannot import fireguard_ml ({exc}).\n"
        'From the repo root run:  pip install -e "ml[dev,scripts]"'
    ) from exc

EXPECTED_BANDS = 23
GFS_BANDS_0BASED = [17, 18, 19, 20, 21]  # docs/wsts_data_dictionary.md; verified against FirePred.py band order
DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def dated_tifs(event_dir: Path):
    out = []
    for p in event_dir.glob("*.tif"):
        m = DATE_RE.fullmatch(p.stem)
        if m:
            out.append((dt.date(*map(int, m.groups())), p))
    return sorted(out)


def target_positive_count(path: Path) -> int:
    """Number of active-fire pixels in the last band of one daily raster."""
    with rasterio.open(path) as src:
        if src.count != EXPECTED_BANDS:
            return -1
        band = src.read(EXPECTED_BANDS).astype(np.float32)
    return int((np.nan_to_num(band, nan=0.0) > 0).sum())


def consecutive_windows(event_dir: Path, n_needed: int):
    files = dated_tifs(event_dir)
    for i in range(len(files) - n_needed + 1):
        run = files[i : i + n_needed]
        if all((run[k + 1][0] - run[k][0]).days == 1 for k in range(n_needed - 1)):
            yield run


def pick_event(root: Path, n_needed: int, max_events: int):
    """Prefer a window whose target day actually contains fire.

    Each event is padded with quiet days before ignition, so the first valid window usually has an
    all-negative target. That is fine for a shape test but weak as evidence, because it never shows
    the fire target decoding working. Fall back to the first valid window if no fire is found.
    """
    first = (None, None)
    for n, event_dir in enumerate(sorted(p for p in root.glob("20??/*") if p.is_dir())):
        if n >= max_events:
            break
        for run in consecutive_windows(event_dir, n_needed):
            if first[1] is None:
                first = (event_dir, run)
            if target_positive_count(run[-1][1]) > 0:
                return event_dir, run, "window chosen because its target day contains fire"
    return first[0], first[1], (
        f"no window with a non-empty target found in the first {max_events} events; "
        "using the first valid window, so the target is all-negative"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder containing the year folders (2018, 2019, ...)")
    ap.add_argument("--event-dir", default=None, help="optional: one event folder to use instead of the first valid one")
    ap.add_argument("--T", type=int, default=5, help="number of input days")
    ap.add_argument("--base", type=int, default=8, help="model width; small by default so this runs on CPU")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--max-events", type=int, default=40,
                    help="how many events to scan for a window whose target day contains fire")
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()

    n_needed = args.T + 1  # T input days + the target day
    if args.event_dir:
        event_dir = Path(args.event_dir)
        files = dated_tifs(event_dir)
        run = files[:n_needed] if len(files) >= n_needed else None
        if run and not all((run[k + 1][0] - run[k][0]).days == 1 for k in range(n_needed - 1)):
            sys.exit(f"{event_dir}: first {n_needed} files are not consecutive days")
        why = "window chosen by --event-dir"
    else:
        event_dir, run, why = pick_event(Path(args.data_root), n_needed, args.max_events)
    if not run:
        sys.exit(f"No event with {n_needed} consecutive daily .tif files found")

    frames = []
    for _, p in run:
        with rasterio.open(p) as src:
            if src.count != EXPECTED_BANDS:
                sys.exit(f"{p}: {src.count} bands, expected {EXPECTED_BANDS}")
            frames.append(src.read().astype(np.float32))
    if len({f.shape for f in frames}) != 1:
        sys.exit(f"frames of one event differ in shape: {[f.shape for f in frames]}")

    keep = [b for b in range(EXPECTED_BANDS) if b not in GFS_BANDS_0BASED]
    x = np.stack([f[keep] for f in frames[:-1]])          # [T, 18, H, W], oldest -> newest
    y = np.nan_to_num(frames[-1][-1], nan=0.0) > 0        # next-day active-fire mask (last band of last file)
    x = torch.from_numpy(np.nan_to_num(x, nan=0.0))[None]  # [1, T, 18, H, W]
    _, t, c, h, w = x.shape

    m = 2 ** args.depth
    hp, wp = -(-h // m) * m, -(-w // m) * m
    x_pad = F.pad(x, (0, wp - w, 0, hp - h))               # stand-in for Role 2's padding utility

    result = {
        "event": event_dir.name,
        "year": event_dir.parent.name,
        "input_dates_oldest_to_newest": [d.isoformat() for d, _ in run[:-1]],
        "target_date": run[-1][0].isoformat(),
        "raw_hw": [h, w],
        "padded_hw": [hp, wp],
        "input_tensor": list(x_pad.shape),
        "channels_per_frame": c,
        "target_positive_pixels": int(y.sum()),
        "target_positive_fraction": float(y.mean()),
        "window_selection": why,
        "models": {},
    }
    torch.manual_seed(0)
    for cls in (StackedDaysUNet, ConvLSTMUNet):
        model = cls(c_strict=c, T=t, base=args.base, depth=args.depth).eval()
        with torch.no_grad():
            logits = model(x_pad)
        probs = model.predict(x_pad, orig_hw=(h, w))
        assert logits.shape == (1, 1, hp, wp), logits.shape
        assert probs.shape == (1, 1, h, w), probs.shape
        assert torch.isfinite(logits).all(), "non-finite logits"
        result["models"][cls.model_type] = {
            "logits": list(logits.shape),
            "probs_after_unpad": list(probs.shape),
            "parameters": sum(p.numel() for p in model.parameters()),
        }

    out = Path(args.out_root) / "outputs" / "temporal_real_sample_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))
    print(f"\nPASS. Wrote {out}")


if __name__ == "__main__":
    main()
