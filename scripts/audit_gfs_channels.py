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
# Source: docs/wsts_data_dictionary.md sections 3 and 4.
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
        rec_down_gm = "CONDITIONAL: range is full-circle. Needs Role 4 Level-A/B ruling on GRIDMET timing and the convention conversion in section 5."
    else:
        rec_down_gm = "DISABLE: direction could not be shown to be full-circle."
    rec_film = "CONDITIONAL: benchmark-assumption experiment only, never strict M1-M4. Blocked until Level-A for GFS is ruled on by Role 4."

    L = []
    A = L.append
    A("# GFS channel and wind-direction audit")
    A("")
    A(f"> DRAFT generated by `scripts/audit_gfs_channels.py` on {dt.date.today().isoformat()}. "
      "Numbers are computed from the raw GeoTIFFs; the timing section is from reading the dataset-creation code. "
      "Review before committing. Reviewer: ________")
    A("")
    A("## 1. Scope and provenance")
    A(f"- Files sampled: {ctx['n_files']} of {ctx['n_total']} TIFFs (every {ctx['every']}th per year, years: {', '.join(map(str, sorted(year_acc)))}).")
    A(f"- Files skipped: {len(ctx['skipped'])}." + (" Reasons: " + "; ".join(f"{p.name}: {r}" for p, r in ctx['skipped'][:5]) if ctx['skipped'] else ""))
    A(f"- Pixels used for percentiles/histogram: random sample of up to {ctx['k']} finite pixels per file per channel. Min, max, mean and NaN counts are exact over all sampled files.")
    A("- Schema: `docs/wsts_data_dictionary.md` (pinned baseline commit f62dc49). Creation code: SebastianGer/WildfireSpreadTSCreateDataset (`FirePred.py`).")
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
    A("Full table (including per-year rows): `outputs/gfs_channel_range_summary.csv`. Histogram: `outputs/gfs_wind_direction_histogram.png`.")
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
          "Possible causes to check in Earth Engine: how `ee.Image.atan2` orders its arguments or handles quadrants, or a later transformation. This does not change the classification.")
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
    A("| Is the day UTC? | The code uses UTC dates in the index and in `acq_date`; local-time shift is not handled | UNKNOWN, verify |")
    A("")
    A("Remaining verification (Earth Engine Code Editor): print one image, for example")
    A("`print(ee.ImageCollection('NOAA/GFS0P25').filter(ee.Filter.eq('system:index','2018010600F24')).first());`")
    A("and record its creation/forecast time properties and the NOAA publication lag. Write UNKNOWN until you have seen them.")
    A("")
    A("## 7. Other findings from the code (hand to Role 4)")
    A("1. **GFS precipitation definition changes on 2019-11-07.** Before that date the code sums hourly values; after it, it takes the last value (cumulative). The channel therefore means different things in 2018/early 2019 versus late 2019-2021. Per-year statistics below should show it if it matters. This matters for leave-one-year-out evaluation.")
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
    A("## 8. Recommendations")
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
