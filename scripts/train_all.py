"""Train, evaluate and compare every FireGuard model, then publish the best one. One command.

    python scripts/train_all.py --data-root ../WildfireSpreadTS/WildfireSpreadTS

Steps (each is skipped when its output already exists, so re-running the same command after an
interruption continues where it stopped; inside a fold, training resumes after the last finished
epoch):

    1 cache      convert the strict channels of every GeoTIFF once into a fast array cache
    2 baselines  persistence / distance-decay / climatology (only if their outputs are missing)
    3 train      M1, M2, M3, M4 in that order, each on the four leave-one-year-out folds
    4 evaluate   score every finished model on each fold's held-out test year
    5 compare    paired event-group bootstrap against the baselines and between models (gates)
    6 report     results/summary.md, metrics tables, figures, results/best_model.json
    7 publish    the best model's forecast maps for the replay site + backend registry bundle

Output folders
    <cache-dir>                     array cache (default: next to the dataset, ``wsts_cache``)
    outputs/training/<model>/       checkpoints, normalisers, calibration, per-epoch history
    outputs/evaluation/runs/<model> test-split scores for each model (no images)
    results/                        everything the report needs (start at results/summary.md)
    artifacts/runs/<best>_v1        the best model's forecast GeoTIFFs (replay + live)
    backend/artifacts               what the website serves

Useful options
    --models m1 m3          only these models (default: m1 m2 m3 m4)
    --folds 2018            only these folds (testing a setup; the report needs all four)
    --ablations             also train the predeclared ablations (configs/models.yaml)
    --only report           run one step: cache, baselines, train, evaluate, compare, report,
                            publish
    --batch-size / --workers / --device   override the automatic hardware choice
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))  # works without `pip install -e ml` too

STEPS = ("cache", "baselines", "train", "evaluate", "compare", "report", "publish")
FOLDS = (2018, 2019, 2020, 2021)
OUT = ROOT / "outputs"            # training state and per-model scores
RESULTS = ROOT / "results"        # the report pack
ARTIFACTS = ROOT / "artifacts"    # the published best model (replay + live)
BACKEND_ARTIFACTS = ROOT / "backend" / "artifacts"
LOG_PATH = OUT / "train_all.log"


def log(msg: str) -> None:
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str]) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {' '.join(map(str, cmd))}")


def model_specs(cfg: dict, keys: list[str], ablations: bool) -> list[dict]:
    """[{key, report_name, model_type, T, base, depth, lstm_kernel, channels, tag}] in order."""
    channels = list(cfg["data"]["channels"])
    specs = []
    for k in keys:
        m = cfg["models"][k]
        specs.append({"key": k, **m, "channels": channels, "tag": ""})
    if ablations:
        for name, a in cfg["ablations"].items():
            base = dict(cfg["models"][a["model"]])
            base.update({k: v for k, v in a.items() if k in ("T", "base", "report_name")})
            ch = [c for c in channels if c not in a.get("drop_channels", [])]
            tag = name.split("_", 1)[1] if a.get("drop_channels") else ""
            specs.append({"key": name, **base, "channels": ch, "tag": tag})
    return specs


def run_config(spec: dict, cfg: dict):
    from fireguard_ml.training.trainer import TemporalRunConfig

    tr = cfg["training"]
    return TemporalRunConfig(
        model=spec["model_type"], T=int(spec["T"]), channels=tuple(spec["channels"]),
        base=int(spec["base"]), depth=int(spec["depth"]),
        lstm_kernel=int(spec.get("lstm_kernel", 3)), dropout=float(tr["dropout"]),
        lr=float(tr["lr"]), min_lr=float(tr["min_lr"]),
        warmup_epochs=float(tr["warmup_epochs"]), weight_decay=float(tr["weight_decay"]),
        max_epochs=int(tr["max_epochs"]), patience=int(tr["patience"]),
        min_delta=float(tr["min_delta"]), effective_batch=int(tr["effective_batch"]),
        grad_clip=float(tr["grad_clip"]), augment_flips=bool(tr["augment_flips"]),
        amp=bool(tr["amp"]), seed=int(tr["seed"]), bce_weight=float(tr["bce_weight"]),
        dice_weight=float(tr["dice_weight"]), tag=spec["tag"],
        preprocessing_version=cfg["data"]["preprocessing_version"])


def protocol(cfg: dict):
    from fireguard_ml.training.trainer import ProtocolFiles

    d = cfg["data"]
    return ProtocolFiles(ROOT / d["split_manifest"], ROOT / d["split_hashes"],
                         ROOT / d["eval_samples"], ROOT / d["strict_channel_manifest"])


def max_frame_hw(cache_dir: Path) -> tuple[int, int]:
    hs, ws = [352], [320]
    for p in cache_dir.glob("*/meta.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        hs.append(m["height"])
        ws.append(m["width"])
    return max(hs), max(ws)


def training_done(spec: dict, cfg: dict, folds) -> bool:
    rc = run_config(spec, cfg)
    for f in folds:
        p = OUT / "training" / spec["report_name"] / f"fold_{f}" / \
            f"{rc.run_name()}.summary.json"
        if not p.exists() or json.loads(p.read_text())["config_hash"] != rc.config_hash():
            return False
    return True


# ------------------------------------------------------------------------------- steps


def step_cache(a, cfg) -> None:
    from fireguard_ml.data.frame_cache import build_frame_cache

    build_frame_cache(a.data_root, a.cache_dir, cfg["data"]["channels"], log=log)


def step_baselines(a, cfg) -> None:
    need = [ROOT / "outputs/role4/baseline_distance_counts.npz",
            ROOT / "docs/evidence/role4/baseline_per_event_test_scores.csv"]
    if all(p.exists() for p in need):
        log("[baselines] already computed; skipping")
        return
    run(["scripts/pipeline/run_baselines.py", "--data-root", a.data_root])


def step_train(a, cfg, specs) -> None:
    from fireguard_ml.data.frame_cache import CachedTemporalSource
    from fireguard_ml.training.hardware import plan_hardware
    from fireguard_ml.training.trainer import build_model, train_fold

    proto = protocol(cfg)
    hw_frame = max_frame_hw(a.cache_dir)
    for spec in specs:
        rc = run_config(spec, cfg)
        if training_done(spec, cfg, a.folds):
            log(f"[train] {spec['report_name']}: all requested folds finished; skipping")
            continue
        log(f"[train] {spec['report_name']} ({rc.run_name()})")
        hw = plan_hardware(lambda rc=rc: build_model(rc), len(rc.channels), rc.T, hw_frame,
                           rc.effective_batch, device=a.device, amp=rc.amp,
                           batch_size=a.batch_size, num_workers=a.workers, log=log)
        source = CachedTemporalSource(a.cache_dir, rc.channels, rc.T, proto.eval_samples)
        out = OUT / "training" / spec["report_name"]
        for fold in a.folds:
            train_fold(rc, source, proto, fold, out, hw=hw, log=log)


def step_evaluate(a, cfg, specs) -> list[dict]:
    from fireguard_ml.data.frame_cache import CachedTemporalSource
    from fireguard_ml.inference.temporal_oof import export_oof
    from fireguard_ml.training.hardware import pick_device

    proto = protocol(cfg)
    done = []
    for spec in specs:
        if not training_done(spec, cfg, FOLDS):
            log(f"[evaluate] {spec['report_name']}: not all four folds trained yet; skipping")
            continue
        rc = run_config(spec, cfg)
        tdir = OUT / "training" / spec["report_name"]
        ckpts = {f: tdir / f"fold_{f}" / f"{rc.run_name()}.pt" for f in FOLDS}
        out = OUT / "evaluation" / "runs" / spec["report_name"]
        mv = out / "model_version.json"
        if mv.exists():
            recorded = {k: v["checkpoint_sha256"]
                        for k, v in json.loads(mv.read_text())["folds"].items()}
            current = {str(f): json.loads((tdir / f"fold_{f}" / f"{rc.run_name()}.summary.json")
                                          .read_text())["checkpoint_sha256"] for f in FOLDS}
            if recorded == current:
                log(f"[evaluate] {spec['report_name']}: already scored; skipping")
                done.append(spec)
                continue
        log(f"[evaluate] {spec['report_name']}: scoring the held-out test year of each fold")
        source = CachedTemporalSource(a.cache_dir, rc.channels, rc.T, proto.eval_samples)
        export_oof(tdir, ckpts, source, proto, OUT / "evaluation",
                   spec["report_name"], str(pick_device(a.device)), write_rasters=False, log=log)
        done.append(spec)
    return done


def step_compare(a, cfg, specs) -> None:
    if a.smoke:
        log("[compare] skipped in --smoke (it needs every test fire of the real dataset)")
        return
    args = ["scripts/pipeline/run_model_comparison.py", "--out", RESULTS / "comparison"]
    for spec in specs:
        p = OUT / "evaluation" / "runs" / spec["report_name"] / "post_reveal" / \
            "per_event_test_scores.csv"
        if p.exists():
            args += ["--model", f"{spec['report_name']}={p}"]
    if len(args) == 1:
        log("[compare] no evaluated model yet; skipping")
        return
    run(args)


def step_report(a, cfg, specs) -> dict:
    from fireguard_ml.evaluation.report import build_report

    return build_report(ROOT, [s["report_name"] for s in specs], training_root=OUT / "training",
                        eval_root=OUT / "evaluation", results=RESULTS, log=log)


def step_publish(a, cfg, specs) -> None:
    best_path = RESULTS / "best_model.json"
    if not best_path.exists():
        log("[publish] no best model yet (run the report after all models are trained)")
        return
    best = json.loads(best_path.read_text())
    spec = next(s for s in specs if s["report_name"] == best["model"])
    rc = run_config(spec, cfg)
    run_id = best["published_run_id"]
    reg_dir = ARTIFACTS / "registry" / run_id
    if (reg_dir / "registry_bundle.json").exists() and \
            (ARTIFACTS / "runs" / run_id / "model_version.json").exists():
        log(f"[publish] {run_id} already published; skipping")
        return
    pipe = "scripts/pipeline"
    run([f"{pipe}/export_temporal_oof.py", "--run-dir", OUT / "training" / best["model"],
         "--run", rc.run_name(), "--source", "cache", "--cache-dir", a.cache_dir,
         "--artifact-root", ARTIFACTS, "--run-id", run_id, "--device", a.device or "auto"])
    run([f"{pipe}/build_replay_catalogue.py", "--artifact-root", ARTIFACTS, "--run-id", run_id])
    run([f"{pipe}/validate_oof_and_build_reveal.py", "--artifact-root", ARTIFACTS,
         "--run-id", run_id, "--data-root", a.data_root])
    run([f"{pipe}/build_backend_registry.py", "--artifact-root", ARTIFACTS, "--run-id", run_id,
         "--data-root", a.data_root, "--backend-artifact-root", BACKEND_ARTIFACTS,
         "--out", reg_dir])
    log(f"[publish] {run_id} ready: load it into the backend with the command in "
        "docs/RUNBOOK.md (section 3)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("--data-root", type=Path, required=True,
                    help="WSTS folder that holds 2018, 2019, 2020, 2021")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="array cache (default: <data-root>/../../wsts_cache, about 16 GB)")
    ap.add_argument("--models", nargs="+", default=["m1", "m2", "m3", "m4"])
    ap.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--only", choices=STEPS, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (default: best available)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny models, 2 epochs, outputs under outputs/smoke: checks the setup")
    a = ap.parse_args()
    a.data_root = a.data_root.resolve()
    if not (a.data_root / "2018").is_dir():
        raise SystemExit(f"{a.data_root} has no 2018 folder; pass the folder holding 2018..2021")
    a.cache_dir = (a.cache_dir or a.data_root.parent.parent / "wsts_cache").resolve()
    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    if a.smoke:  # same code path, tiny budget; never mixed with real outputs
        global OUT, RESULTS, ARTIFACTS, BACKEND_ARTIFACTS, LOG_PATH
        OUT = ROOT / "outputs" / "smoke"
        RESULTS, ARTIFACTS = OUT / "results", OUT / "artifacts"
        BACKEND_ARTIFACTS, LOG_PATH = OUT / "backend_artifacts", OUT / "train_all.log"
        cfg["training"].update(max_epochs=2, patience=2, effective_batch=4)
        for m in [*cfg["models"].values()]:
            m["base"] = 8
    specs = model_specs(cfg, a.models, a.ablations)
    steps = [a.only] if a.only else list(STEPS)
    t0 = time.time()
    log(f"=== train_all: steps {steps}, models {[s['report_name'] for s in specs]}, "
        f"folds {a.folds}")
    for step in steps:
        log(f"--- step {step}")
        if step == "cache":
            step_cache(a, cfg)
        elif step == "baselines":
            step_baselines(a, cfg)
        elif step == "train":
            step_train(a, cfg, specs)
        elif step == "evaluate":
            step_evaluate(a, cfg, specs)
        elif step == "compare":
            step_compare(a, cfg, specs)
        elif step == "report":
            step_report(a, cfg, specs)
        elif step == "publish":
            step_publish(a, cfg, specs)
    log(f"=== done in {(time.time() - t0) / 3600:.2f} h. Start with {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()
