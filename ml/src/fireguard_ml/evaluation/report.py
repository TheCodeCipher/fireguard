"""Build the report pack in ``results/`` from training, evaluation and comparison outputs.

Inputs (all written by ``scripts/train_all.py``)
    outputs/training/<model>/fold_<Y>/<run>.summary.json, .history.csv   training record
    outputs/evaluation/runs/<model>/post_reveal/                         test-split scores
    results/comparison/                                                  paired tests and gates
    docs/evidence/role4/baseline_results.json                            baseline test metrics

Outputs (``results/``)
    summary.md                          the one page to read first
    best_model.json                     which model the site and live forecasts use, and why
    metrics/test_metrics_by_fold.csv    pooled + event-macro test metrics, models and baselines
    metrics/training_runs.csv           one row per model and fold (epochs, time, hardware)
    metrics/training_history.csv        one row per epoch for every model and fold
    figures/*.png                       see FIGURES below

Best-model rule (fixed before results): the main model (M1-M4) with the highest mean
VALIDATION AUPRC over the four folds. The test split never chooses the model; test numbers are
then reported for it like for every other model. Ties go to the smaller model.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from fireguard_ml.evaluation.calibration import FoldCalibrator
from fireguard_ml.evaluation.metrics import (
    ScoreCounts,
    average_precision_from_counts,
    ece_equal_mass,
    merge_counts,
    reliability_table,
    summarise,
)

FOLDS = (2018, 2019, 2020, 2021)
MAIN = ("m1_unet", "m2_attention_res_unet", "m3_stacked_unet", "m4_convlstm_unet")
LABEL = {
    "m1_unet": "M1 spatial U-Net",
    "m2_attention_res_unet": "M2 attention residual U-Net",
    "m3_stacked_unet": "M3 stacked days (T=5)",
    "m4_convlstm_unet": "M4 ConvLSTM U-Net (T=5)",
    "m1_no_terrain": "M1 without terrain (ablation)",
    "m3_stacked_unet_T3": "M3 with 3 days (ablation)",
    "m3_stacked_unet_base48": "M3 at M4's size (ablation)",
    "distance_decay": "Distance-decay baseline",
    "climatology": "Climatology baseline",
}
# Categorical slots in fixed order (validated palette); baselines are neutral greys.
COLOR = {
    "m1_unet": "#2a78d6", "m2_attention_res_unet": "#eb6834", "m3_stacked_unet": "#1baf7a",
    "m4_convlstm_unet": "#eda100", "m1_no_terrain": "#e87ba4", "m3_stacked_unet_T3": "#008300",
    "m3_stacked_unet_base48": "#6250d6", "distance_decay": "#52514e", "climatology": "#a8a7a2",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIGURES = {
    "01_test_auprc_by_fold.png": "pooled test AUPRC per fold, every model and both soft baselines",
    "02_precision_recall_curves.png": "test precision-recall curves, one panel per fold",
    "03_training_curves.png": "validation AUPRC and training loss per epoch, every model and fold",
    "04_reliability.png": "reliability of the best model's probabilities, raw and Platt-calibrated",
    "05_per_event_vs_baseline.png": "per-event test AUPRC difference to the strongest baseline",
}


def _counts(npz: Path) -> ScoreCounts:
    z = np.load(npz)
    return ScoreCounts(z["scores"], z["pos"], z["neg"], float(z["sum_sq"]), float(z["sum_s_pos"]))


def _f(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, float) and math.isfinite(v) else "-"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def collect_training(training_root: Path, names) -> tuple[list[dict], list[dict]]:
    runs, hist = [], []
    for name in names:
        for fold in FOLDS:
            fdir = training_root / name / f"fold_{fold}"
            for sp in sorted(fdir.glob("*.summary.json")):
                s = json.loads(sp.read_text(encoding="utf-8"))
                hw = s.get("hardware") or {}
                runs.append({"model": name, "fold": fold, "run": s["run"],
                             "best_epoch": s["best_epoch"], "epochs_run": s.get("epochs_run"),
                             "stopped_early": s.get("stopped_early"),
                             "val_auprc": s["val_auprc"], "val_threshold": s.get("val_threshold"),
                             "parameters": s["parameters"], "minutes": round(s["seconds"] / 60, 1),
                             "device": hw.get("gpu_name") or hw.get("device"),
                             "batch_size": hw.get("batch_size"), "grad_accum": hw.get("grad_accum"),
                             "precision": hw.get("amp_dtype") or "float32"})
                hp = sp.with_name(sp.name.replace(".summary.json", ".history.csv"))
                if hp.exists():
                    with hp.open(encoding="utf-8") as f:
                        for r in csv.DictReader(f):
                            hist.append({"model": name, "fold": fold, **r})
    return runs, hist


def calibrated_metrics(c: ScoreCounts, cal: FoldCalibrator | None, fold: int) -> dict:
    """Brier and ECE after the fold's validation-fitted Platt calibration (from score counts)."""
    if cal is None or cal.a <= 0 or c.n == 0:
        return {}
    s = cal.transform(c.scores, fold)
    tot = c.pos + c.neg
    brier = float(((tot * s**2).sum() - 2 * (c.pos * s).sum() + c.n_pos) / c.n)
    return {"brier_calibrated": brier,
            "ece_calibrated": ece_equal_mass(ScoreCounts(s, c.pos, c.neg), 15)}


def test_metrics(eval_root: Path, names, baselines: dict, thresholds: dict,
                 calibrators: dict | None = None) -> list[dict]:
    rows = []
    for fold in FOLDS:
        clim_brier = baselines["folds"][str(fold)]["pooled"]["climatology"]["brier"]
        for b in ("distance_decay", "climatology"):
            p = baselines["folds"][str(fold)]["pooled"][b]
            macro = baselines["folds"][str(fold)]["event_macro"].get(b, {})
            rows.append({"model": b, "fold": fold, "pooled_auprc": p["auprc"], "pooled_f1": p["f1"],
                         "pooled_iou": p["iou"], "brier": p["brier"],
                         "bss_vs_climatology": p.get("bss_vs_climatology"),
                         "macro_auprc": (macro.get("auprc") or [None])[0]})
        for name in names:
            post = eval_root / "runs" / name / "post_reveal"
            npz = post / f"pooled_counts_fold_{fold}.npz"
            if not npz.exists():
                continue
            counts = _counts(npz)
            m = summarise(counts, threshold=thresholds.get((name, fold)))
            cm = calibrated_metrics(counts, (calibrators or {}).get((name, fold)), fold)
            if cm:
                cm["bss_calibrated"] = 1 - cm["brier_calibrated"] / clim_brier
            per_event = []
            with (post / "per_event_test_scores.csv").open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if int(r["fold_test_year"]) == fold and r["auprc"] not in ("", "nan"):
                        per_event.append(float(r["auprc"]))
            rows.append({"model": name, "fold": fold, "pooled_auprc": m["auprc"],
                         "pooled_f1": m.get("f1"), "pooled_iou": m.get("iou"), "brier": m["brier"],
                         "bss_vs_climatology": 1 - m["brier"] / clim_brier, "ece": m["ece"],
                         **cm,
                         "macro_auprc": float(np.mean(per_event)) if per_event else None,
                         "n_events_with_fire": len(per_event)})
    return rows


def choose_best(runs: list[dict]) -> dict | None:
    by = {}
    for r in runs:
        if r["model"] in MAIN:
            by.setdefault(r["model"], {})[r["fold"]] = r
    complete = {m: v for m, v in by.items() if set(v) == set(FOLDS)}
    if not complete:
        return None
    score = {m: float(np.mean([v[f]["val_auprc"] for f in FOLDS])) for m, v in complete.items()}
    params = {m: v[FOLDS[0]]["parameters"] for m, v in complete.items()}
    best = max(score, key=lambda m: (round(score[m], 4), -params[m]))
    return {"model": best, "mean_val_auprc": score[best], "candidates_mean_val_auprc": score,
            "rule": "highest mean validation AUPRC over the four P1 folds among M1-M4 "
                    "(test split not used; ties to the smaller model)"}


# ------------------------------------------------------------------------------- figures


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_figures(out: Path, metrics: list[dict], hist: list[dict], eval_root: Path, names,
                 best: str | None, thresholds: dict, calibrators: dict, per_event_diff: list[dict],
                 decay_curves: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    out.mkdir(parents=True, exist_ok=True)
    made = []
    models = [*names, "distance_decay", "climatology"]

    # 01 grouped bars
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        ys = [next((r["pooled_auprc"] for r in metrics if r["model"] == m and r["fold"] == f),
                   np.nan) for f in FOLDS]
        xs = np.arange(len(FOLDS)) + (i - (len(models) - 1) / 2) * width
        ax.bar(xs, ys, width * 0.9, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(range(len(FOLDS)), [f"test year {f}" for f in FOLDS])
    _style(ax, "Pooled test AUPRC by leave-one-year-out fold (higher is better)", "", "AUPRC")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(out / "01_test_auprc_by_fold.png", dpi=160)
    plt.close(fig)
    made.append("01_test_auprc_by_fold.png")

    # 02 PR curves
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, fold in zip(axes, FOLDS, strict=True):
        for m in names:
            npz = eval_root / "runs" / m / "post_reveal" / f"pooled_counts_fold_{fold}.npz"
            if not npz.exists():
                continue
            c = _counts(npz)
            tp, fp = np.cumsum(c.pos), np.cumsum(c.neg)
            ax.plot(tp / max(c.n_pos, 1), tp / np.maximum(tp + fp, 1), color=COLOR[m],
                    lw=1.6, label=f"{LABEL[m]} ({average_precision_from_counts(c):.3f})")
        if fold in decay_curves:
            r, p, ap = decay_curves[fold]
            ax.plot(r, p, color=COLOR["distance_decay"], lw=1.4, ls="--",
                    label=f"{LABEL['distance_decay']} ({ap:.3f})")
        _style(ax, f"test year {fold}", "recall", "precision" if fold == FOLDS[0] else "")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.suptitle("Precision-recall on the held-out year (AUPRC in brackets)", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "02_precision_recall_curves.png", dpi=160)
    plt.close(fig)
    made.append("02_precision_recall_curves.png")

    # 03 training curves
    if hist:
        fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 5.4), squeeze=False)
        shades = {2018: 1.0, 2019: 0.8, 2020: 0.6, 2021: 0.4}
        for j, m in enumerate(names):
            for fold in FOLDS:
                rs = [r for r in hist if r["model"] == m and int(r["fold"]) == fold]
                if not rs:
                    continue
                ep = [int(r["epoch"]) for r in rs]
                kw = dict(color=COLOR[m], alpha=shades[fold], lw=1.4, label=str(fold))
                axes[0, j].plot(ep, [float(r["val_auprc"]) for r in rs], **kw)
                axes[1, j].plot(ep, [float(r["train_loss"]) for r in rs], **kw)
            _style(axes[0, j], LABEL[m], "", "validation AUPRC" if j == 0 else "")
            _style(axes[1, j], "", "epoch (from 0)", "training loss" if j == 0 else "")
            for ax_ in axes[:, j]:
                ax_.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0, j].legend(fontsize=7, frameon=False, title="fold", title_fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "03_training_curves.png", dpi=160)
        plt.close(fig)
        made.append("03_training_curves.png")

    # 04 reliability of the best model (pooled over folds; raw and Platt-calibrated)
    if best:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        for kind, ls in (("raw", "-"), ("calibrated", "--")):
            parts = []
            for fold in FOLDS:
                npz = eval_root / "runs" / best / "post_reveal" / f"pooled_counts_fold_{fold}.npz"
                if not npz.exists():
                    continue
                c = _counts(npz)
                if kind == "calibrated":
                    cal = calibrators.get((best, fold))
                    if cal is None or cal.a <= 0:  # not monotone: skip rather than reorder
                        continue
                    c = ScoreCounts(cal.transform(c.scores, fold), c.pos, c.neg)
                parts.append(c)
            if not parts:
                continue
            rows = sorted(reliability_table(merge_counts(parts), 15), key=lambda b: b["mean_prob"])
            ax.plot([b["mean_prob"] for b in rows], [b["observed"] for b in rows], ls=ls,
                    marker="o", ms=3, lw=1.2, color=COLOR[best], label=f"{kind} probability")
        ax.plot([0, 1], [0, 1], color=GRID, lw=1)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("symlog", linthresh=1e-3)
        _style(ax, f"Reliability, {LABEL[best]} (all folds)", "mean predicted probability",
               "observed fire frequency")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "04_reliability.png", dpi=160)
        plt.close(fig)
        made.append("04_reliability.png")

    # 05 per-event differences to the strongest baseline
    if per_event_diff:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        data, labels, cols = [], [], []
        for m in names:
            d = [float(r["difference"]) for r in per_event_diff if r.get("candidate") == m
                 and r.get("reference") in ("distance_decay", "climatology")
                 and r.get("difference") not in ("", "nan", None)]
            if d:
                data.append(d)
                labels.append(LABEL[m].split(" (")[0])
                cols.append(COLOR[m])
        if data:
            bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                            widths=0.5, medianprops={"color": INK})
            for patch, c in zip(bp["boxes"], cols, strict=True):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
            ax.axhline(0, color=INK2, lw=0.8)
            _style(ax, "Per-event test AUPRC minus the strongest baseline (above 0 = better)",
                   "", "AUPRC difference")
            ax.tick_params(axis="x", labelsize=8)
            fig.tight_layout()
            fig.savefig(out / "05_per_event_vs_baseline.png", dpi=160)
            made.append("05_per_event_vs_baseline.png")
        plt.close(fig)
    return made


# ------------------------------------------------------------------------------- summary


def build_report(root: Path, names, training_root: Path | None = None,
                 eval_root: Path | None = None, results: Path | None = None, log=print) -> dict:
    results = results or root / "results"
    training_root = training_root or root / "outputs" / "training"
    eval_root = eval_root or root / "outputs" / "evaluation"
    baselines = json.loads((root / "docs/evidence/role4/baseline_results.json").read_text())
    runs, hist = collect_training(training_root, names)
    thresholds, calibrators = {}, {}
    for name in names:
        for fold in FOLDS:
            for sp in (training_root / name / f"fold_{fold}").glob("*.selection.json"):
                sel = json.loads(sp.read_text())
                thresholds[(name, fold)] = sel["threshold"]["threshold"]
                c = sel["calibrator"]
                calibrators[(name, fold)] = FoldCalibrator(**c)
    metrics = test_metrics(eval_root, names, baselines, thresholds, calibrators)
    _write_csv(results / "metrics" / "test_metrics_by_fold.csv", metrics)
    _write_csv(results / "metrics" / "training_runs.csv", runs)
    _write_csv(results / "metrics" / "training_history.csv", hist)
    comp = results / "comparison"
    gates = json.loads((comp / "gate_decisions.json").read_text()) if (
        comp / "gate_decisions.json").exists() else {}
    diffs = []
    if (comp / "per_event_differences.csv").exists():
        with (comp / "per_event_differences.csv").open(encoding="utf-8") as f:
            diffs = list(csv.DictReader(f))
    decay_curves = _decay_curves(root, baselines)
    best = choose_best(runs)
    figs = make_figures(results / "figures", metrics, hist, eval_root, names,
                        best["model"] if best else None, thresholds, calibrators, diffs,
                        decay_curves)
    if best:
        b = best["model"]
        best["baseline_check"] = gates.get("baseline_check_by_model", {}).get(b, {})
        best["test_pooled_auprc_by_fold"] = {r["fold"]: r["pooled_auprc"] for r in metrics
                                            if r["model"] == b}
        try:
            best["training_dir"] = (training_root / b).resolve().relative_to(
                root.resolve()).as_posix()
        except ValueError:
            best["training_dir"] = str(training_root / b)
        best["run"] = next(r["run"] for r in runs if r["model"] == b)
        # Models trained on the synthetic test data publish as "<model>_mock_v1": the export and
        # the backend refuse synthetic data under a name that could pass for a real model.
        mock = any(json.loads(f.read_text()).get("is_mock")
                   for f in (training_root / b).glob("fold_*/*.summary.json"))
        best["published_run_id"] = f"{b}_mock_v1" if mock else f"{b}_v1"
        (results / "best_model.json").write_text(json.dumps(best, indent=2, default=str) + "\n")
    _summary(results, names, runs, metrics, gates, best, figs)
    log(f"[report] wrote {results / 'summary.md'}; best model: "
        f"{best['model'] if best else 'none yet (not every model has all four folds)'}")
    return {"best": best, "figures": figs}


def _decay_curves(root: Path, baselines: dict) -> dict:
    cache = root / "outputs" / "role4" / "baseline_distance_counts.npz"
    if not cache.exists():
        return {}
    from fireguard_ml.baselines.distance_decay import DistanceCounts, decay_scores
    from fireguard_ml.data.event_splits import fold_events, read_manifest

    rows = read_manifest(root / "data/manifests/p1_loyo_fold_manifest.csv")
    z = np.load(cache, allow_pickle=False)
    counts = {k: DistanceCounts.from_arrays(z[k]) for k in z.files if k != "_problems"}
    out = {}
    for fold in FOLDS:
        ch = baselines["folds"][str(fold)]["chosen"]
        test = DistanceCounts.merge([counts[e] for e in fold_events(rows, fold, "test")
                                     if e in counts])
        c = test.to_score_counts(decay_scores(test, ch["decay_scale_km"], ch["decay_amplitude"]))
        tp, fp = np.cumsum(c.pos), np.cumsum(c.neg)
        out[fold] = (tp / max(c.n_pos, 1), tp / np.maximum(tp + fp, 1),
                     average_precision_from_counts(c))
    return out


def _summary(results: Path, names, runs, metrics, gates, best, figs) -> None:
    L = ["# FireGuard results summary (generated)", "",
         "> Written by `scripts/train_all.py` (step `report`). Do not edit by hand; re-run it. "
         "File guide: `results/README.md`. Protocol: leave-one-year-out (P1), each year is "
         "tested by a model that never saw it.", ""]
    if best:
        L += ["## Selected model", "",
              f"**{LABEL[best['model']]}** (`{best['model']}`), mean validation AUPRC "
              f"{best['mean_val_auprc']:.4f}. Rule: {best['rule']}.", ""]
        bc = best.get("baseline_check") or {}
        if bc:
            L.append(f"Against the strongest soft baseline (paired event-group bootstrap, Gate 2 "
                     f"rule): **{bc.get('decision')}**.")
            L.append("")
        L += ["| Model | Mean validation AUPRC |", "|---|---:|"]
        for m, v in sorted(best["candidates_mean_val_auprc"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {LABEL[m]} | {v:.4f} |")
        L.append("")
    else:
        L += ["No model has all four folds yet, so no model is selected.", ""]
    L += ["## Test metrics per fold", "",
          "Pooled = all test cells of the year together; macro = mean over fires. F1/IoU use the "
          "threshold chosen on validation. Brier, BSS (Brier skill score against climatology, "
          "above 0 = better) and ECE are for the calibrated probability (Platt scaling fitted on "
          "validation); the raw-sigmoid values are in metrics/test_metrics_by_fold.csv. The "
          "BCE + Dice loss ranks cells well but pushes raw outputs away from true frequencies, "
          "which is why calibration is applied.", "",
          "| Model | Fold | Pooled AUPRC | Macro AUPRC | F1 | IoU | Brier | BSS | ECE |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = [*names, "distance_decay", "climatology"]
    for m in order:
        for r in [r for r in metrics if r["model"] == m]:
            model_row = r["model"] in names
            brier = r.get("brier_calibrated") if model_row else r["brier"]
            bss = r.get("bss_calibrated") if model_row else r.get("bss_vs_climatology")
            L.append(f"| {LABEL[m]} | {r['fold']} | {_f(r['pooled_auprc'])} | "
                     f"{_f(r.get('macro_auprc'))} | {_f(r.get('pooled_f1'))} | "
                     f"{_f(brier, 5)} | {_f(bss)} | "
                     f"{_f(r.get('ece_calibrated'), 5)} |")
    L += ["", "2019 has the fewest fires (88 test events); its numbers vary most. Read every "
          "comparison per fold, not only as an average.", "",
          "## Comparisons and gates", ""]
    if gates:
        L.append(f"- Gate 2 (M1 vs strongest baseline): **{gates['gate2']['decision']}**")
        for m, v in gates.get("baseline_check_by_model", {}).items():
            if m != "m1_unet":
                L.append(f"- {LABEL.get(m, m)} vs strongest baseline: **{v['decision']}**")
        for k, v in gates.get("gate3", {}).items():
            L.append(f"- {k} ({v['adds']}): **{v['decision']}**")
        L.append(f"- Context 3 vs 5 days: **{gates['context_ablation_T3_vs_T5']['decision']}**")
        L += ["", "Full tables with confidence intervals: "
              "`results/comparison/model_comparison.md`."]
    else:
        L.append("Not run yet.")
    L += ["", "## Training", "",
          "| Model | Fold | Best epoch | Epochs run | Validation AUPRC | Minutes | Parameters | "
          "Batch x accumulation | Precision |", "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in runs:
        L.append(f"| {LABEL[r['model']]} | {r['fold']} | {r['best_epoch']} | {r['epochs_run']} | "
                 f"{r['val_auprc']:.4f} | {r['minutes']} | {r['parameters'] / 1e6:.2f} M | "
                 f"{r['batch_size']} x {r['grad_accum']} | {r['precision']} |")
    L += ["", "Epochs are counted from 0. Training stops after 6 epochs without a validation gain "
          "and keeps the best epoch's weights.", "", "## Figures", ""]
    for name in figs:
        L.append(f"- `results/figures/{name}`: {FIGURES[name]}")
    L.append("")
    (results / "summary.md").write_text("\n".join(L), encoding="utf-8")
