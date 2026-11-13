#!/usr/bin/env python3
"""GFS / wind-direction audit for WSTS (Role 3, tasks P1-R3-01 and P1-R3-02).

Run from the repo root:
    python scripts/audit_gfs_channels.py --data-root /path/to/WildfireSpreadTS --every 10

It reads the RAW GeoTIFF bands, NOT the loader output. The loader applies sin()
to angle channels, which would squash every angle into [-1, 1] and hide the range.

Writes:
    outputs/gfs_channel_range_summary.csv
    outputs/gfs_wind_direction_histogram.png
    docs/gfs_channel_audit.md        (DRAFT: computed numbers + code-trace evidence)

WARNING: this overwrites docs/gfs_channel_audit.md. Section 8 of that file comes from STATIC_REVIEW below;
if you fill in the result tables by hand (8.4, 8.7), copy them into STATIC_REVIEW before re-running.

Dependencies: numpy, rasterio, matplotlib.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPECTED_BANDS = 23
PCTS = [1, 5, 25, 50, 75, 95, 99]

# name: (original 0-based index, 1-based rasterio band, processed loader index)
# Source: docs/data_dictionary.md sections 3 and 4.
CHANNELS = {
    "gridmet_wind_speed": (6, 7, 6),
    "gridmet_wind_direction": (7, 8, 7),
    "gfs_total_precipitation": (17, 18, 33),
    "gfs_wind_speed": (18, 19, 34),
    "gfs_wind_direction": (19, 20, 35),
    "gfs_temperature": (20, 21, 36),
    "gfs_specific_humidity": (21, 22, 37),
}


class Acc:
    """Exact min/max/mean/NaN counts plus a random sample for percentiles."""

    def __init__(self):
        self.n = 0
        self.nan = 0
        self.inf = 0
        self.fin = 0
        self.sum = 0.0
        self.min = np.inf
        self.max = -np.inf
        self.sample = []

    def add(self, a, rng, k):
        a = a.ravel()
        self.n += a.size
        nan_m = np.isnan(a)
        inf_m = np.isinf(a)
        self.nan += int(nan_m.sum())
        self.inf += int(inf_m.sum())
        f = a[~nan_m & ~inf_m]
        if f.size == 0:
            return
        self.fin += f.size
        self.sum += float(f.sum(dtype=np.float64))
        self.min = min(self.min, float(f.min()))
        self.max = max(self.max, float(f.max()))
        if f.size > k:
            f = rng.choice(f, k, replace=False)
        self.sample.append(f.astype(np.float32))

    def pooled(self):
        return np.concatenate(self.sample) if self.sample else np.array([], dtype=np.float32)

    def mean(self):
        return self.sum / self.fin if self.fin else float("nan")

    def pct(self):
        v = self.pooled()
        return list(np.percentile(v, PCTS)) if v.size else [float("nan")] * len(PCTS)

    def nan_frac(self):
        return self.nan / self.n if self.n else float("nan")


def quadrant_fractions(v):
    ang = np.mod(v, 360.0)
    q = np.floor(ang / 90.0).astype(int).clip(0, 3)
    return np.bincount(q, minlength=4) / max(v.size, 1)


def classify_direction(acc):
    v = acc.pooled()
    if v.size == 0:
        return "UNKNOWN", "no finite values found", None
    lo, hi = acc.min, acc.max
    qf = quadrant_fractions(v)
    if max(abs(lo), abs(hi)) <= 2 * np.pi + 0.05:
        return "UNKNOWN", f"range [{lo:.3f}, {hi:.3f}] looks like radians, not degrees", qf
    if lo >= -91 and hi <= 91:
        return "QUADRANT_LIMITED", f"all values within about [-90, 90] (min {lo:.2f}, max {hi:.2f})", qf
    if (hi - lo) >= 330 and qf.min() >= 0.02:
        return (
            "FULL_CIRCLE",
            f"range spans {hi - lo:.1f} degrees (min {lo:.2f}, max {hi:.2f}) and all four 90-degree sectors hold >= 2% of values",
            qf,
        )
    return "UNKNOWN", f"range [{lo:.2f}, {hi:.2f}] fits neither pattern cleanly", qf


def circ_abs_diff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


# GFS direction = atan2(v, u) in degrees: angle of the direction the wind blows TOWARD,
# counterclockwise from east (from the creation code, FirePred.py).
# Question: how does GRIDMET 'th' (angle, believed to be the direction wind comes FROM,
# clockwise from north) relate to it? Test four hypotheses against real paired pixels.
HYPOTHESES = {
    "gfs ~ 270 - th   (th = FROM, clockwise from N)": lambda th: 270.0 - th,
    "gfs ~ 90 - th    (th = TOWARD, clockwise from N)": lambda th: 90.0 - th,
    "gfs ~ th         (same convention)": lambda th: th,
    "gfs ~ th + 180   (opposite of same convention)": lambda th: th + 180.0,
}


# Hand-written review of the creation code and the Earth Engine catalogue. It is NOT computed from the data;
# it is kept here so that re-running the script does not wipe it from docs/gfs_channel_audit.md.
# Line numbers: SebastianGer/WildfireSpreadTSCreateDataset @ f6c5167 (DataPreparation/satellites/FirePred.py)
# and SebastianGer/WildfireSpreadTS @ f62dc49. Reviewed 2026-11-11.
STATIC_REVIEW = """## 8. Review of the creation code and the Earth Engine catalogue (hand-written, not computed)

Sources read on 2026-11-11: `FirePred.py` in SebastianGer/WildfireSpreadTSCreateDataset at commit `f6c5167`; the loader in SebastianGer/WildfireSpreadTS at commit `f62dc49` (the pinned baseline); the Earth Engine catalogue page for [NOAA/GFS0P25](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25) and the API pages for [`ee.Image.atan2`](https://developers.google.com/earth-engine/apidocs/ee-image-atan2) and [`ee.Number.atan2`](https://developers.google.com/earth-engine/apidocs/ee-number-atan2). Statuses use the project vocabulary: VERIFIED, ASSUMED, UNKNOWN, LEAKY/EXCLUDED.

### 8.1 Status register for Role 4

| Item | Finding | Status | Evidence |
|---|---|---|---|
| GFS band identity | Export order in `FirePred.py` lines 125-131 puts forecast rain, wind speed, wind direction, temperature, humidity at 0-based positions 17-21, directly before active fire (22). Matches section 2. | VERIFIED | code |
| GFS direction range | [-90, 90] in the data; half of all directions are indistinguishable | VERIFIED | sections 3, 4 |
| GFS direction formula | Published code is `v.atan2(u)` (line 74), which should be full-circle. The data is not. They disagree. | UNKNOWN | 8.2 |
| u/v available | Not exported (lines 68-76 keep only speed and direction) | VERIFIED | code |
| GFS run | 00Z run of UTC day D (index prefix `<D>00`) | VERIFIED | code line 61-62 |
| GFS lead hours actually selected | Intended F01-F24. Depends on the width of the hour field in `system:index`; not yet checked. | UNKNOWN | 8.3, EE check 1 |
| GFS Level A | Borderline at best (F24 is valid at 00:00Z on D+1); FAIL if EE check 1 shows more than F01-F24 | UNKNOWN | 8.3 |
| GFS Level B | Publication lag of the 00Z run not measured | UNKNOWN | 8.3 |
| Day boundary | UTC calendar day. Date strings carry no time zone (Earth Engine reads them as UTC) and fire detections are filtered on `acq_date`. | VERIFIED from code | lines 42-63, 116-120 |
| GFS precipitation meaning | Break at 2019-11-07 confirmed against observed precipitation. Neither regime is a daily total. A second, unexplained drift sits inside the post-change period. | VERIFIED (the break); the quantity the channel actually holds is UNKNOWN | 8.4 |
| GFS wind speed | Speed of the 24-field mean vector, not mean speed | VERIFIED | lines 68-73 |
| GRIDMET direction range | 0-360, all four sectors populated | VERIFIED | sections 3, 4 |
| GRIDMET direction convention | Degrees clockwise from north, direction the wind comes FROM | ASSUMED | 8.5 |
| GRIDMET day window | Which 24 hours the image selected for UTC day D covers is not fixed by the code | UNKNOWN | 8.5, EE check 5 |
| Loader angle handling | `sin()` only, on original 0-based band indices 7, 13, 19 (GRIDMET direction, aspect, GFS direction) | VERIFIED | loader `utils.py` line 162, `FireSpreadDataset.py` line 339, README note of Feb 2026 |

### 8.2 Why the GFS direction is quadrant-limited: what is and is not known

- The published code computes `forecast_v_wind.atan2(forecast_u_wind)` and converts radians to degrees (lines 74-76).
- Earth Engine's documentation for `ee.Number.atan2` says the left input is x and the right input is y; `ee.Image.atan2` is described as the angle of the vector [x, y] with image1 as the left operand. Read that way, `v.atan2(u)` is the angle of the point (x = v, y = u): a compass bearing, clockwise from north, of the direction the wind blows toward, covering the full circle.
- The data does not look like that. The range is exactly [-90, 90], and the axis test in section 5 fits the `90 - th` / `270 - th` family (median 32.3) and not the `th` / `th + 180` family (44.5, which is what unrelated angles give).
- Both facts are what `arctan(v / u)` in degrees would produce: an angle counter-clockwise from east with the sign of u thrown away. A wind blowing toward the north-east and one blowing toward the south-west get the same value.
- **Folded, not clipped.** The histogram (`docs/evidence/gfs_wind_direction_histogram.png`) distinguishes the two possibilities. It tapers smoothly toward both ends and the bins at -90 and +90 are the smallest in the plot. A channel that had been clipped, with out-of-range values clamped to the limits, would pile up into tall spikes at exactly those two values. The values were therefore folded by the arithmetic that produced them, not clamped afterwards, which is what `arctan(v / u)` gives: an angle of exactly +/-90 arises only as u approaches zero, which is rare. This also covers the "representative values" part of the Phase 1 brief; no separate value listing is needed.
- So the released rasters and the published code disagree. Possible reasons: the rasters were built with a different version of the code, or `ee.Image.atan2` behaved differently from its documentation. **Cause: UNKNOWN.** EE checks 3a and 3b in `scripts/ee_checks_gfs_gridmet.js` test the second possibility. Neither outcome changes the classification, because u and v are not in the files and the lost half cannot be recovered.
- Reading the 360-degree column of section 5: `270 - th` scores 42.6 only because GRIDMET `th` lies between 180 and 360 for about 74% of pixels (sector shares 0.439 + 0.304). For those winds u is positive and `arctan(v / u)` happens to equal the full angle. The folded channel gives the same numbers whether `th` means FROM or TOWARD, so this column cannot settle the GRIDMET convention either.

### 8.3 GFS timing

- Selection (lines 60-63): images whose `system:index` is, as a string, at least `<D>00F01` and at most `<D>00F24`. Temperature, humidity, u and v are averaged over the selected images.
- **Open risk, 1-minute check.** A string comparison only selects F01 to F24 if the hour field in the index is two digits wide. If it is three digits (F001 to F384), the same filter matches F010 to F239, which would mean forecast valid times up to about ten days after D. The index format could not be confirmed from the catalogue page. EE check 1 prints exactly what the filter returns. Until it is run, the selected lead hours are UNKNOWN.
- If check 1 returns F01-F24: valid times run from 01:00Z on D to 00:00Z on D+1. The last field sits on the start of the target day, so Level A is borderline (1 of 24 averaged fields) and needs Role 4's ruling.
- If check 1 returns more than F01-F24: Level A is FAIL for all five GFS channels.
- As coded, the frame for day D carries the forecast for day D itself, issued at the start of that UTC day. It is not a forecast of the target day D+1. Any claim that the GFS branch "uses the forecast for the day being predicted" would need Role 1 to show the loader shifts it, which the pinned loader does not do.
- Level B: a 00Z run is published some hours after 00:00Z. The lag is not measured here. UNKNOWN.
- UTC day versus local day: for the western United States, UTC day D runs from about 16:00-17:00 local time on D-1 to 16:00-17:00 local time on D.

### 8.4 GFS precipitation

- Code (lines 78-85): for dates up to and including 2019-11-07 the selected fields are summed; after that the last field is taken. The comment says the product changed from rain within a one-hour interval to cumulative rain. The following `rename` is not assigned, so the band keeps its source name; band order is unaffected.
- The Earth Engine catalogue describes `total_precipitation_surface` as cumulative precipitation over the previous 1 to 6 hours, depending on the forecast hour by ((F - 1) % 6) + 1, and says totals should be built from forecast hours that are multiples of 6 to avoid double counting. If that description holds for the period, summing F01-F24 counts the same rain several times, and taking the last field keeps only the final six hours. Neither is a 24-hour total.
- Supporting numbers from this audit: the yearly mean falls from 17.5 (2018) and 20.2 (2019) to 5.6 (2020) and 0.86 (2021), and the pooled maximum of 820 is above the catalogue's estimated band maximum of 626.75, which a sum of several fields can exceed and a single field should not.
- Weather differs between years, so those numbers alone do not prove a break. `scripts/check_gfs_precip_break.py` controls for it by comparing against observed GRIDMET precipitation; its result is below and it does prove the break. EE check 4 would additionally show the field's behaviour hour by hour on each side of the change date.
- Consequence for leave-one-year-out: the 2018 test fold is entirely in the "sum" regime, 2020 and 2021 entirely in the "last" regime, and 2019 is split. A model with this channel would see a different quantity at test time than in most of its training data.

**Result of `check_gfs_precip_break.py`** (run 2026-11-11 over 1,361 sampled files; GRIDMET band 6, observed mm/day, is the reference; evidence copy `docs/evidence/gfs_precip_break_check.csv`):

| Group | Files | Observed GRIDMET mean (mm/day) | GFS channel mean | GFS / GRIDMET | Median of per-file GFS means | Largest GFS value seen |
|---|---:|---:|---:|---:|---:|---:|
| sum rule (<= 2019-11-07) | 514 | 0.491 | 16.616 | 33.87 | 4.181 | 820.0 |
| last rule (> 2019-11-07) | 847 | 0.493 | 3.237 | 6.57 | 0.009 | 204.2 |
| 2018 | 378 | 0.452 | 16.256 | 36.00 | 3.718 | 820.0 |
| 2019 | 142 | 0.631 | 18.196 | 28.85 | 6.193 | 258.2 |
| 2020 | 429 | 0.353 | 5.154 | 14.60 | 0.717 | 204.2 |
| 2021 | 412 | 0.626 | 0.833 | 1.33 | 0.000 | 136.3 |

How to read it. Observed precipitation is the control and it is flat across the two regimes: 0.491 against 0.493 mm/day. Weather therefore does not explain the difference. Against that flat reference the channel falls by a factor of 5.2, from 33.9 times observed precipitation to 6.6 times. **The break is real and the channel means different things on either side of 2019-11-07.**

Each regime also fails on its own terms. Under the sum rule the channel is about 34 times observed daily precipitation, which is what repeatedly adding overlapping accumulation windows would give. Under the last rule the typical file is empty: the median of per-file means is 0.009 against 4.181 before the change, which is what keeping only the final accumulation bucket would give. Neither is a 24-hour total.

One finding beyond the documented change. 2020 and 2021 both lie wholly after the change date, yet their ratios are 14.60 and 1.33, and the median file in 2021 is 0.000. A second discontinuity sits inside the post-change period and the code does not explain it. UNKNOWN; EE check 4 is the next step.

Consequence. The released channel cannot be made comparable across folds by rescaling, because neither regime holds a known quantity. Using it would need precipitation rebuilt in Earth Engine from forecast hours that are multiples of 6, which is a dataset rebuild and out of scope, exactly as for wind direction. The recommendation is therefore DISABLE, not CONDITIONAL.

### 8.5 GRIDMET wind direction and day window (evidence for Role 4)

- Code (lines 49-56): `filterDate(<D>T00:00, <D>T23:59)` on IDAHO_EPSCOR/GRIDMET, then the median, which is expected to be a single image.
- Range: 0 to 360 with every 90-degree sector populated (sections 3 and 4). NaN fraction 1.07%, identical for wind speed and direction, so it is a property of the GRIDMET grid coverage and not of one band. Where those pixels are has not been mapped.
- Convention: meteorological data normally reports the direction the wind comes from, clockwise from north, and a 74% share between 180 and 360 degrees matches the westerly winds of the region under that reading. The histogram peaks near 250 degrees, which under the FROM reading is wind arriving from the west-southwest, the prevailing pattern for the western United States; under the TOWARD reading the same peak would mean wind arriving from the east-northeast, which is not. That supports the FROM reading without proving it, so the status stays ASSUMED until the band description on the catalogue page is recorded.
- Day window: the filter picks the GRIDMET image whose start time falls inside UTC day D. GRIDMET is a daily product for the United States and its "day" need not be the UTC day, so the selected image may summarise hours after 00:00Z on D+1. EE check 5 prints its start and end time. Level A for all GRIDMET channels is UNKNOWN until then; Level B (publication delay) is UNKNOWN.
- Related: the drought index (lines 93-96) is selected with `datetime.timestamp(today)` on a datetime without time zone, so the result depended on the clock setting of the machine that built the dataset. This can move the 5-day PDSI selection by one image at the boundaries.

### 8.6 Loader facts that affect direction channels (pinned commit f62dc49)

- Frames are read in sorted file-name order (`YYYY-MM-DD.tif`), inputs are all but the last, and the target is the last band of the last file (`FireSpreadDataset.py` lines 152-158).
- Angle channels get `sin()` only. The upstream README (note dated Feb 2026) confirms this loses information and is not fixed.
- Training-time flips and rotations correct angles assuming compass degrees in [0, 360]. The GFS direction band is not in that convention, so at least the vertical-flip correction is wrong for it. Another reason to keep geometric augmentation off (plan v4, section 4.5).

### 8.7 EE check results (fill in after running `scripts/ee_checks_gfs_gridmet.js`)

| Check | Question | Result | Date run / by |
|---|---|---|---|
| 1 | Number of images and forecast hours selected by the creation filter | NOT YET RUN | |
| 2 | `creation_time` and latest `forecast_time` of the selection | NOT YET RUN | |
| 3a / 3b | Operand order and range of `ee.Image.atan2` | NOT YET RUN | |
| 4 | Precipitation by forecast hour, before and after 2019-11-07 | NOT YET RUN | |
| 5 | GRIDMET image start and end time for UTC day D | NOT YET RUN | |
"""


def find_tifs(root: Path):
    out = {}
    for p in sorted(root.rglob("*.tif")):
        year = next((part for part in p.parts if re.fullmatch(r"20\d\d", part)), None)
        if year is None:
            continue
        out.setdefault(int(year), []).append(p)
    return out


def write_csv(path, all_acc, year_acc):
    cols = ["scope", "channel", "band_1based", "n_pixels", "nan_frac", "inf_count",
            "min", "max", "mean"] + [f"p{p}" for p in PCTS]
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        scopes = [("all", all_acc)] + [(str(y), year_acc[y]) for y in sorted(year_acc)]
        for scope, accs in scopes:
            for name, (_, band, _) in CHANNELS.items():
                a = accs[name]
                w.writerow([scope, name, band, a.n, f"{a.nan_frac():.6f}", a.inf,
                            f"{a.min:.6g}", f"{a.max:.6g}", f"{a.mean():.6g}"]
                           + [f"{x:.6g}" for x in a.pct()])


def plot_hist(path, all_acc, cls):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), dpi=100)
    for ax, name in zip(axes, ["gfs_wind_direction", "gridmet_wind_direction"]):
        a = all_acc[name]
        v = a.pooled()
        if v.size:
            lo = np.floor(a.min / 5.0) * 5.0
            hi = np.ceil(a.max / 5.0) * 5.0
            bins = np.arange(lo, hi + 5.0, 5.0)
            ax.hist(v, bins=bins, color="#c0522d" if "gfs" in name else "#2d6ac0")
        ax.set_title(f"{name}  ->  {cls[name][0]}   (min {a.min:.1f}, max {a.max:.1f})", fontsize=10)
        ax.set_xlabel("degrees (raw GeoTIFF value)")
        ax.set_ylabel("sampled pixel count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt(x):
    return f"{x:.4g}"


def build_report(ctx):
    all_acc, year_acc, cls, hyp_res, hyp_verdict = ctx["all_acc"], ctx["year_acc"], ctx["cls"], ctx["hyp_res"], ctx["hyp_verdict"]
    axis_res = ctx["axis_res"]
    gfs_cls, gfs_why, gfs_qf = cls["gfs_wind_direction"]
    gm_cls, gm_why, gm_qf = cls["gridmet_wind_direction"]

    # Recommendations (computed from the evidence above; edit if you disagree and say why)
    if gfs_cls == "FULL_CIRCLE":
        rec_down_gfs = "CONDITIONAL: range is full-circle, but Level-A is borderline (F24 issue, section 6) and the loader keeps sin() only. Needs Role 4 ruling and a cos channel."
    else:
        rec_down_gfs = "DISABLE: direction could not be shown to be full-circle."
    if gm_cls == "FULL_CIRCLE":
        rec_down_gm = "CONDITIONAL: range is full-circle. Needs the FROM/TOWARD convention confirmed and the GRIDMET day window checked (section 8.5, EE check 5), then Role 4's Level-A ruling."
    else:
        rec_down_gm = "DISABLE: direction could not be shown to be full-circle."
    rec_film = ("CONDITIONAL: benchmark-assumption experiment only, never strict M1-M4. Blocked until EE check 1 is run and Role 4 rules on Level A. "
                "If it goes ahead, use at most wind speed (19), temperature (21) and humidity (22): the direction band (20) is folded and the precipitation band (18) is not a daily total (section 8.4). "
                "See docs/gfs_branch_gating.md.")

    L = []
    A = L.append
    A("# GFS channel and wind-direction audit")
    A("")
    A(f"> DRAFT generated by `scripts/audit_gfs_channels.py` on {dt.date.today().isoformat()}. "
      "Numbers are computed from the raw GeoTIFFs (the files the HDF5 conversion is made from); sections 6 to 8 are from reading "
      "the dataset-creation code and the Earth Engine catalogue. Section 8 is hand-written static text. "
      "Review before committing. Reviewer: ________")
    A("")
    A("## 1. Scope and provenance")
    A(f"- Files sampled: {ctx['n_files']} of {ctx['n_total']} TIFFs (every {ctx['every']}th per year, years: {', '.join(map(str, sorted(year_acc)))}).")
    A(f"- Files skipped: {len(ctx['skipped'])}." + (" Reasons: " + "; ".join(f"{p.name}: {r}" for p, r in ctx['skipped'][:5]) if ctx['skipped'] else ""))
    A(f"- Pixels used for percentiles/histogram: random sample of up to {ctx['k']} finite pixels per file per channel. Min, max, mean and NaN counts are exact over all sampled files.")
    A("- Schema: `docs/data_dictionary.md` (Role 1's artifact, pinned baseline commit f62dc49). Creation code: SebastianGer/WildfireSpreadTSCreateDataset (`FirePred.py`).")
    A("- Every schema fact this audit relies on was also checked against the export order in `FirePred.py` lines 125-131, so the conclusions do not rest on the data dictionary alone. **Role 1 confirmed on 2026-11-13** that all five GFS channels match the data dictionary (raw 17-21, GeoTIFF bands 18-22, processed 33-37; no index or ordering differences), and that the manifest will mark processed 33-37 excluded from strict M1-M4.")
    A("")
    A("## 2. Channel identity")
    A("| Channel | Original index (0-based) | GeoTIFF band (1-based) | Loader processed index |")
    A("|---|---:|---:|---:|")
    for name, (o, b, p) in CHANNELS.items():
        A(f"| {name} | {o} | {b} | {p} |")
    A("")
    A("## 3. Raw ranges (sampled files, all years pooled)")
    A("| Channel | min | p1 | p50 | p99 | max | mean | NaN fraction |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in CHANNELS:
        a = all_acc[name]
        p = a.pct()
        A(f"| {name} | {fmt(a.min)} | {fmt(p[0])} | {fmt(p[3])} | {fmt(p[6])} | {fmt(a.max)} | {fmt(a.mean())} | {a.nan_frac():.4f} |")
    A("")
    A("Full table (including per-year rows): `outputs/gfs_channel_range_summary.csv`. Histogram: `outputs/gfs_wind_direction_histogram.png`. `outputs/` is git-ignored (`.gitignore` line 66), so the committed copies of this evidence live in `docs/evidence/`. Note for the maintainer: the execution guide lists these two files under `outputs/` as Phase 1 artifacts to commit, which the repository's own `.gitignore` forbids. The copies under `docs/evidence/` resolve that; say so if the guide should be amended.")
    A("")
    A("## 4. Direction classification")
    A(f"- **GFS wind direction (band 20): `{gfs_cls}`**. {gfs_why}.")
    if gfs_qf is not None:
        A(f"  - Share of values per sector (0-90, 90-180, 180-270, 270-360 degrees, taken mod 360): {', '.join(f'{x:.3f}' for x in gfs_qf)}.")
    A(f"- **GRIDMET wind direction (band 8): `{gm_cls}`**. {gm_why}.")
    if gm_qf is not None:
        A(f"  - Share of values per sector: {', '.join(f'{x:.3f}' for x in gm_qf)}.")
    A("- **Raw u/v components: NOT in the released rasters.** `compute_daily_features` returns only forecast wind speed and direction (derived from u/v inside Earth Engine); u and v are not exported. Direction cannot be rebuilt from anything else in the files.")
    if gfs_cls == "FULL_CIRCLE":
        A("- **How GFS direction is built (code trace):** u and v 10 m wind are averaged over lead hours F01-F24, then `atan2(v, u)` is converted to degrees, giving [-180, 180], counterclockwise from east, as the direction the wind blows toward.")
    else:
        A("- **Code trace versus data:** reading `FirePred.py` (u and v averaged over F01-F24, then `atan2(v, u)` converted to degrees) predicts a range of [-180, 180]. "
          f"The data shows [{fmt(all_acc['gfs_wind_direction'].min)}, {fmt(all_acc['gfs_wind_direction'].max)}]. "
          "The stored channel therefore does not behave like a full-circle atan2. **Cause: UNKNOWN.** The measured data takes precedence over the code reading. "
          "The values behave like `arctan(v / u)`; see section 8.2. This does not change the classification.")
    A("- **Loader effect:** the baseline applies `sin()` to angle channels and not `cos()` (data dictionary section 9). Sine alone cannot tell 30 degrees from 150 degrees, so even a full-circle channel loses information in the loader.")
    A("")
    A("## 5. Do the GFS and GRIDMET direction conventions agree?")
    A("Paired pixels where both channels are finite; lower median circular difference = better fit. A random relationship gives about 90 degrees in the 360-degree column and about 45 in the 180-degree column.")
    A("")
    A("| Hypothesis | 360-degree median diff | 360-degree mean | 180-degree (axis) median diff | 180-degree (axis) mean |")
    A("|---|---:|---:|---:|---:|")
    for h, (med, mean) in hyp_res.items():
        am, amean = axis_res[h]
        A(f"| {h} | {med:.1f} | {mean:.1f} | {am:.1f} | {amean:.1f} |")
    A("")
    A(f"- Verdict: {hyp_verdict}")
    A("- The convention of GRIDMET `th` should also be confirmed on its Earth Engine catalogue page (IDAHO_EPSCOR/GRIDMET). Until then treat it as UNKNOWN.")
    A("")
    A("## 6. GFS timing trace (from `FirePred.py`)")
    A("Selection rule: the forecast images with index between `<YYYYMMDD>00F01` and `<YYYYMMDD>00F24`, i.e. the **00Z run of day D**, lead hours 1 to 24, averaged. The loader pairs inputs of day D with the fire target of day D+1 (data dictionary section 6).")
    A("")
    A("| Question | Finding | Status |")
    A("|---|---|---|")
    A("| Which run? | 00Z run of day D | confirmed from code |")
    A("| Which valid times? | 01:00Z on D to 00:00Z on D+1 (F01 to F24) | confirmed from code |")
    A("| Level A (no look-ahead) | F24 is valid at 00:00Z on D+1, which is the start of the target day. It is 1 of 24 averaged fields, but the rule is \"nothing at or after the start of the target day\". Not a clean PASS. | UNKNOWN (borderline), needs Role 4 ruling |")
    A("| Level B (operationally available) | The 00Z run is published some hours after 00:00Z, and its F01 to F24 window covers time that has already passed on day D. Whether it counts as available depends on the forecast issue time the app assumes. | UNKNOWN |")
    A("| Is the day UTC? | Yes. Date strings carry no time zone (Earth Engine reads them as UTC) and detections are filtered on `acq_date`; no local-time shift is applied | confirmed from code |")
    A("")
    A("Remaining verification: run `scripts/ee_checks_gfs_gridmet.js` in the Earth Engine Code Editor and record the output in section 8.7. "
      "The table above assumes the filter selects F01 to F24; section 8.3 explains why that itself still needs checking. Write UNKNOWN until you have seen the output.")
    A("")
    A("## 7. Other findings from the code (hand to Role 4)")
    A("1. **GFS precipitation definition changes on 2019-11-07.** Up to that date the code sums the selected fields; after it, it takes the last one. The channel therefore means different things in 2018/most of 2019 versus 2020-2021. Details and the catalogue's description are in section 8.4. This matters for leave-one-year-out evaluation.")
    A("2. **GFS precipitation band is not renamed.** `forecast_rain.rename(...)` is called without assigning the result. Band order is unaffected (positional), but any name-based lookup would miss it.")
    A("3. **GFS wind speed is the speed of the 24-hour vector mean**, so it understates speed on days when the wind direction changes.")
    A("4. **VNP13A1 (NDVI/EVI2) window:** the code takes the last composite whose start lies in the 15 days up to the end of day D. A 16-day composite can extend past D. Needs Role 4's Level-A check.")
    A("")
    A("Per-year GFS precipitation (sampled):")
    A("")
    A("| Year | min | p50 | max | mean |")
    A("|---|---:|---:|---:|---:|")
    for y in sorted(year_acc):
        a = year_acc[y]["gfs_total_precipitation"]
        p = a.pct()
        A(f"| {y} | {fmt(a.min)} | {fmt(p[3])} | {fmt(a.max)} | {fmt(a.mean())} |")
    A("")
    L.extend(STATIC_REVIEW.rstrip("\n").split("\n"))
    A("")
    A("## 9. Recommendations")
    A("| Branch | Recommendation |")
    A("|---|---|")
    A("| GFS in strict M1-M4 | DISABLE (project rule, not subject to this audit) |")
    A(f"| Downwind baseline using GFS direction | {rec_down_gfs} |")
    A(f"| Downwind baseline using GRIDMET direction | {rec_down_gm} |")
    A(f"| GFS / FiLM experiment | {rec_film} |")
    A("")
    A("An UNKNOWN or DISABLE outcome is acceptable. Silently using a channel whose direction or timing is unverified is not.")
    A("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder containing the year folders (2018, 2019, ...)")
    ap.add_argument("--every", type=int, default=10, help="use every Nth TIFF within each year (default 10)")
    ap.add_argument("--pixels-per-file", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parents[1]),
                    help="repo root; outputs/ and docs/ are written here")
    args = ap.parse_args()

    root = Path(args.data_root)
    out_root = Path(args.out_root)
    by_year = find_tifs(root)
    if not by_year:
        sys.exit(f"No .tif files found under year folders in {root}")
    n_total = sum(len(v) for v in by_year.values())
    selected = [(y, p) for y, ps in sorted(by_year.items()) for p in ps[:: max(args.every, 1)]]
    print(f"Found {n_total} TIFFs; auditing {len(selected)} (every {args.every}th per year).")

    rng = np.random.default_rng(args.seed)
    names = list(CHANNELS)
    bands = [CHANNELS[n][1] for n in names]
    i_gfs = names.index("gfs_wind_direction")
    i_gm = names.index("gridmet_wind_direction")
    all_acc = {n: Acc() for n in names}
    year_acc = {y: {n: Acc() for n in names} for y in by_year}
    pair_g, pair_m = [], []
    skipped = []
    used = 0

    for i, (y, p) in enumerate(selected, 1):
        try:
            with rasterio.open(p) as src:
                if src.count != EXPECTED_BANDS:
                    skipped.append((p, f"{src.count} bands, expected {EXPECTED_BANDS}"))
                    continue
                arr = src.read(bands).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            skipped.append((p, f"read error: {e}"))
            continue
        used += 1
        for j, n in enumerate(names):
            all_acc[n].add(arr[j], rng, args.pixels_per_file)
            year_acc[y][n].add(arr[j], rng, args.pixels_per_file)
        g, m = arr[i_gfs].ravel(), arr[i_gm].ravel()
        ok = np.flatnonzero(np.isfinite(g) & np.isfinite(m))
        if ok.size:
            if ok.size > args.pixels_per_file:
                ok = rng.choice(ok, args.pixels_per_file, replace=False)
            pair_g.append(g[ok])
            pair_m.append(m[ok])
        if i % 50 == 0:
            print(f"  {i}/{len(selected)} files")

    if used == 0:
        sys.exit("No readable 23-band TIFFs. Check --data-root and the skipped reasons: " + str(skipped[:3]))

    cls = {n: classify_direction(all_acc[n]) for n in ["gfs_wind_direction", "gridmet_wind_direction"]}

    hyp_res, axis_res = {}, {}
    hyp_verdict = "no paired finite pixels, convention check not possible. UNKNOWN."
    if pair_g:
        g = np.concatenate(pair_g)
        m = np.concatenate(pair_m)
        for h, fn in HYPOTHESES.items():
            d = circ_abs_diff(g, fn(m))
            hyp_res[h] = (float(np.median(d)), float(np.mean(d)))
            da = np.abs(((g - fn(m) + 90.0) % 180.0) - 90.0)  # ignores the sign (180-degree ambiguity)
            axis_res[h] = (float(np.median(da)), float(np.mean(da)))
        ranked = sorted(hyp_res.items(), key=lambda kv: kv[1][0])
        best, second = ranked[0], ranked[1]
        if best[1][0] <= 60 and (second[1][0] - best[1][0]) >= 15:
            hyp_verdict = f"best fit is `{best[0].strip()}` (median {best[1][0]:.1f} deg, next best {second[1][0]:.1f}). Supported, not proven."
        else:
            hyp_verdict = (f"inconclusive. Best `{best[0].strip()}` has median {best[1][0]:.1f} deg, next best {second[1][0]:.1f}. "
                           "Treat the relationship as UNKNOWN.")

    if cls["gfs_wind_direction"][0] != "FULL_CIRCLE":
        hyp_verdict = ("NOT MEANINGFUL for the 360-degree column: the GFS channel is not full-circle, so it has lost half "
                       "of the direction information and cannot be compared with GRIDMET as a direction. "
                       "Use the 180-degree (axis) column instead, which ignores the lost sign.")

    (out_root / "outputs").mkdir(parents=True, exist_ok=True)
    (out_root / "docs").mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "outputs" / "gfs_channel_range_summary.csv", all_acc, year_acc)
    plot_hist(out_root / "outputs" / "gfs_wind_direction_histogram.png", all_acc, cls)
    report = build_report(dict(all_acc=all_acc, year_acc=year_acc, cls=cls, hyp_res=hyp_res, axis_res=axis_res,
                               hyp_verdict=hyp_verdict, n_files=used, n_total=n_total,
                               every=args.every, skipped=skipped, k=args.pixels_per_file))
    (out_root / "docs" / "gfs_channel_audit.md").write_text(report, encoding="utf8")

    print("\n=== SUMMARY ===")
    for n in names:
        a = all_acc[n]
        print(f"{n:26s} band {CHANNELS[n][1]:2d}  min {a.min:10.4g}  max {a.max:10.4g}  NaN {a.nan_frac():.4f}")
    for n, (c, why, qf) in cls.items():
        print(f"{n}: {c} -- {why}")
    for h, (med, mean) in hyp_res.items():
        print(f"  {h}: 360-deg median {med:.1f} mean {mean:.1f} | 180-deg median {axis_res[h][0]:.1f} mean {axis_res[h][1]:.1f}")
    print(hyp_verdict)
    print(f"\nFiles used {used}, skipped {len(skipped)}.")
    print(f"Wrote outputs/ and docs/ under {out_root}")


if __name__ == "__main__":
    main()
