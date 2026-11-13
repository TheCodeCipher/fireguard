"""Paired comparisons, Gate 2 and Gate 3 for all models on frozen P1 (Role 4, P3-R4-03).

    python scripts/run_model_comparison.py \
        --model m1_unet=<role2 run>/post_reveal/per_event_test_scores.csv \
        --model m3_stacked_unet=artifacts/runs/<id>/post_reveal/per_event_test_scores.csv ...

Model names are assigned here (``name=path``) so the gate pairs in
``fireguard_ml.evaluation.comparison`` are unambiguous. Each model folder must hold
``model_version.json`` (Role 3 format) whose split hash equals the frozen P1 manifest; pooled
metrics come from ``post_reveal/pooled_counts_fold_<Y>.npz`` when present. Baselines come from
``docs/evidence/role4/baseline_per_event_test_scores.csv`` and the stage-A cache.

Writes reports/phase3/{ablation_results.csv, event_bootstrap_results.csv,
per_event_differences.csv, gate_decisions.json, model_comparison.md}. Models not supplied are
reported as NOT RUN; nothing is imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from fireguard_ml.baselines.distance_decay import DistanceCounts, decay_scores, fit_climatology
from fireguard_ml.data.event_splits import fold_events, read_manifest, validate_manifest
from fireguard_ml.evaluation.comparison import (
    CONTEXT_ABLATION,
    GATE3_PAIRS,
    SOFT_BASELINES,
    ComparisonError,
    context_ablation,
    event_macro,
    gate2,
    gate3,
    paired,
    per_event_differences,
    read_scores,
    strongest_baseline,
)
from fireguard_ml.evaluation.metrics import ScoreCounts, average_precision_from_counts, summarise

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests"
OUT = ROOT / "reports" / "phase3"
BASE_SCORES = ROOT / "docs" / "evidence" / "role4" / "baseline_per_event_test_scores.csv"
BASE_JSON = ROOT / "docs" / "evidence" / "role4" / "baseline_results.json"
CACHE = ROOT / "outputs" / "role4" / "baseline_distance_counts.npz"
ALL_MODELS = ("m1_unet", "m2_unet_no_terrain", "m3_stacked_unet", "m3_stacked_unet_T3",
              "m3_stacked_unet_base40", "m4_convlstm_unet")  # fmt: skip


def lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def frozen_hash() -> str:
    text = (MAN / "p1_split_hashes.txt").read_text(encoding="utf-8")
    if "status=FROZEN" not in text:
        raise SystemExit("P1 split is not FROZEN: no comparison may be reported")
    rec = {ln.split("  ", 1)[1]: ln.split("  ", 1)[0] for ln in text.splitlines()
           if ln and not ln.startswith("#")}  # fmt: skip
    actual = lf_sha(MAN / "p1_loyo_fold_manifest.csv")
    if rec["p1_loyo_fold_manifest.csv"] != actual:
        raise SystemExit("p1_loyo_fold_manifest.csv does not match its frozen hash")
    return actual


def baseline_validation_auprc(rows, base: dict) -> dict:
    """Validation pooled AUPRC of each soft baseline per fold (selects the strongest baseline)."""
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} missing: run scripts/run_baselines.py first (it writes it)")
    z = np.load(CACHE, allow_pickle=False)
    counts = {k: DistanceCounts.from_arrays(z[k]) for k in z.files if k != "_problems"}
    cfg = yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text())
    b = cfg["baselines"]["distance_bin_climatology"]
    out = {}
    for fold, f in base["folds"].items():
        y = int(fold)
        val = DistanceCounts.merge([counts[e] for e in fold_events(rows, y, "validation")
                                    if e in counts])  # fmt: skip
        train = DistanceCounts.merge([counts[e] for e in fold_events(rows, y, "train")
                                      if e in counts])  # fmt: skip
        ch = f["chosen"]
        clim = fit_climatology(train, y, "train", tuple(b["edges_km"]), b["prior_strength"])
        out[y] = {
            "distance_decay": average_precision_from_counts(val.to_score_counts(
                decay_scores(val, ch["decay_scale_km"], ch["decay_amplitude"]))),
            "climatology": average_precision_from_counts(val.to_score_counts(clim.scores(val))),
        }  # fmt: skip
    return out


def model_pooled(path: Path, fold: int, thr: float | None) -> dict:
    npz = path.parent / f"pooled_counts_fold_{fold}.npz"
    if not npz.exists():
        return {}
    z = np.load(npz)
    c = ScoreCounts(z["scores"], z["pos"], z["neg"], float(z["sum_sq"]), float(z["sum_s_pos"]))
    return summarise(c, threshold=thr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", default=[], help="name=per_event_csv")
    a = ap.parse_args()
    split_hash = frozen_hash()
    rows = read_manifest(MAN / "p1_loyo_fold_manifest.csv")
    validate_manifest(rows)
    cfg = yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text())["metrics"]["bootstrap"]
    boot = {"n_boot": cfg["n_boot"], "seed": cfg["seed"], "ci": cfg["ci"]}
    base = json.loads(BASE_JSON.read_text())

    models = dict(m.split("=", 1) for m in a.model)
    for name, p in models.items():
        if name not in ALL_MODELS:
            raise SystemExit(f"unknown model name {name}; use one of {ALL_MODELS}")
        mv = Path(p).parent.parent / "model_version.json"
        if not mv.exists():
            raise SystemExit(f"{name}: {mv} missing; cannot verify provenance")
        meta = json.loads(mv.read_text())
        if meta.get("is_mock"):
            raise SystemExit(f"{name}: MOCK run; refusing to report it")
        for fold, v in meta["folds"].items():
            if v["meta"]["split_manifest_hash"] != split_hash:
                raise SystemExit(f"{name} fold {fold}: trained on a different split manifest")

    # rename each model file's 'baseline' column to the assigned name, in memory
    tmp = []
    for name, p in models.items():
        with Path(p).open(encoding="utf-8") as f:
            rs = list(csv.DictReader(f))
        for r in rs:
            r["baseline"] = name
        tmp.append(rs)
    loaded = read_scores([BASE_SCORES])
    scores, group_of = loaded["scores"], loaded["group_of"]
    for rs in tmp:
        for r in rs:
            f, ev = int(r["fold_test_year"]), r["event_id"]
            if group_of.get(ev, r["group_id"]) != r["group_id"]:
                raise ComparisonError(f"{ev}: group id differs from the baseline file")
            scores[f].setdefault(r["baseline"], {})[ev] = {
                k: float(r[k]) if r[k] not in ("", "nan") else math.nan
                for k in ("auprc", "brier", "f1", "iou", "n_pos", "n_valid")}  # fmt: skip

    val_auprc = baseline_validation_auprc(rows, base)
    folds = sorted(scores)
    pairs_done, ablation, diffs = [], [], []
    strongest = {f: strongest_baseline(val_auprc[f]) for f in folds}
    for f in folds:
        test_events = set(fold_events(rows, f, "test"))
        for m in [*SOFT_BASELINES, *ALL_MODELS]:
            if m not in scores[f]:
                ablation.append({"fold": f, "model": m, "status": "NOT RUN"})
                continue
            if set(scores[f][m]) != {e for e in test_events if e in scores[f]["distance_decay"]}:
                raise ComparisonError(f"fold {f} {m}: events differ from the frozen test split")
            row = {"fold": f, "model": m, "status": "OK", "n_events": len(scores[f][m])}
            for k in ("auprc", "f1", "iou", "brier"):
                row[f"macro_{k}"], row[f"macro_{k}_n"] = event_macro(scores, f, m, k)
            pooled = (base["folds"][str(f)]["pooled"].get(m, {}) if m in SOFT_BASELINES else
                      model_pooled(Path(models[m]), f, None))  # fmt: skip
            for k in ("auprc", "brier", "f1", "iou"):
                row[f"pooled_{k}"] = pooled.get(k, "")
            ablation.append(row)
        comps = [("distance_decay", "climatology", "baseline sanity")]
        if "m1_unet" in scores[f]:
            comps.append(("m1_unet", strongest[f], "Gate 2: M1 vs strongest soft baseline"))
        comps += [p for p in (*GATE3_PAIRS, CONTEXT_ABLATION)
                  if p[0] in scores[f] and p[1] in scores[f]]  # fmt: skip
        for cand, ref, why in comps:
            r = paired(scores, group_of, f, cand, ref, "auprc", boot)
            pairs_done.append((why, r))
            diffs += per_event_differences(scores, f, cand, ref)

    g2 = gate2([r for w, r in pairs_done if w.startswith("Gate 2")])
    g3 = {f"{c} vs {r}": {"adds": w, **gate3([x for ww, x in pairs_done
                                              if x.candidate == c and x.reference == r])}
          for c, r, w in GATE3_PAIRS}  # fmt: skip
    ctx = context_ablation([x for _, x in pairs_done if (x.candidate, x.reference)
                            == CONTEXT_ABLATION[:2]])  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    _csv(OUT / "ablation_results.csv", ablation)
    _csv(OUT / "event_bootstrap_results.csv", [
        {"comparison": w, **{k: getattr(r, k) for k in r.__dataclass_fields__},
         "verdict": r.verdict} for w, r in pairs_done])  # fmt: skip
    _csv(OUT / "per_event_differences.csv", diffs)
    decisions = {"split_manifest_sha256": split_hash,
                 "evaluation_config_sha256": lf_sha(ROOT / "configs" / "evaluation.yaml"),
                 "strongest_baseline_by_validation_auprc": strongest,
                 "baseline_validation_auprc": val_auprc, "gate2": g2, "gate3": g3,
                 "context_ablation_T3_vs_T5": ctx,
                 "models_supplied": sorted(models)}  # fmt: skip
    (OUT / "gate_decisions.json").write_text(json.dumps(decisions, indent=2, default=str) + "\n")
    _report(decisions, ablation, pairs_done)
    print(json.dumps({"gate2": g2["decision"], "models": sorted(models)}))


def _csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _fmt(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, float) and math.isfinite(v) else ("-" if v in ("", None)
                                                                          else str(v))  # fmt: skip


def _report(dec, ablation, pairs_done) -> None:
    L = ["# Model comparison on frozen P1 (generated)", "",
         "> Generated by `scripts/run_model_comparison.py`; rules fixed in "
         "`docs/model_comparison_protocol.md`. Do not edit by hand.", "",
         f"- P1 manifest sha256 `{dec['split_manifest_sha256'][:16]}...`; evaluation config "
         f"sha256 `{dec['evaluation_config_sha256'][:16]}...`",
         f"- Models supplied: {', '.join(dec['models_supplied']) or 'none (baselines only)'}",
         "- **2019 caveat:** 74 raw events (88 test events, 61 test groups after grouping), the "
         "smallest fold; its intervals are the widest and it should not decide a gate alone.", "",
         "## Strongest soft baseline per fold (chosen on VALIDATION pooled AUPRC)", "",
         "| Fold | Decay (val) | Climatology (val) | Strongest |",
         "|---:|---:|---:|---|"]  # fmt: skip
    for f, v in sorted(dec["baseline_validation_auprc"].items()):
        L.append(f"| {f} | {v['distance_decay']:.4f} | {v['climatology']:.4f} | "
                 f"{dec['strongest_baseline_by_validation_auprc'][f]} |")  # fmt: skip
    L += ["", "## Test metrics per fold (pooled and event-macro)", "",
          "| Fold | Model | Status | Pooled AUPRC | Macro AUPRC (n) | Macro F1 | Macro IoU |",
          "|---:|---|---|---:|---:|---:|---:|"]  # fmt: skip
    for r in ablation:
        if r["status"] != "OK":
            L.append(f"| {r['fold']} | {r['model']} | NOT RUN | - | - | - | - |")
        else:
            L.append(f"| {r['fold']} | {r['model']} | OK | {_fmt(r['pooled_auprc'])} | "
                     f"{_fmt(r['macro_auprc'])} ({r['macro_auprc_n']}) | {_fmt(r['macro_f1'])} | "
                     f"{_fmt(r['macro_iou'])} |")  # fmt: skip
    L += ["", "## Paired event-group bootstrap (per-event AUPRC difference)", "",
          "| Comparison | Fold | Candidate - reference | 95 % CI | Groups | Events | Verdict |",
          "|---|---:|---|---:|---:|---:|---|"]  # fmt: skip
    for w, r in pairs_done:
        L.append(f"| {w} | {r.fold} | {r.candidate} - {r.reference} | {r.mean_diff:+.4f} "
                 f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.n_groups} | {r.n_events} | "
                 f"{r.verdict} |")  # fmt: skip
    L += ["", "## Gate decisions", "", f"- Gate 2 (M1): **{dec['gate2']['decision']}**"]
    for k, v in dec["gate3"].items():
        L.append(f"- Gate 3, {k} ({v['adds']}): **{v['decision']}**"
                 + (f", folds with CI above zero: {v['folds_above_zero']}" if "folds_above_zero"
                    in v else ""))  # fmt: skip
    L.append(f"- Context ablation (T=3 vs T=5): **{dec['context_ablation_T3_vs_T5']['decision']}**")
    L += [
        "",
        "NOT RUN means no result exists for that model; it is not a negative result. A "
        "REMOVE decision is a recorded null/negative finding and stays in this report.",
        "",
    ]
    (OUT / "model_comparison.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
