"""Train FireGuard models in a Kaggle GPU notebook, one process per GPU, and save the results.

In the notebook (inputs: the code dataset and the array-cache dataset; accelerator GPU T4 x2):

    !python /kaggle/input/<code dataset>/kaggle_train.py --models m3

What it does
    1 copies the code into /kaggle/working and finds the array cache among the inputs (if the
      cache was uploaded as per-event zips, they are extracted to /tmp first)
    2 splits the folds over the GPUs (2 GPUs: 2018+2020 on one, 2019+2021 on the other) and runs
      scripts/train_all.py --only train for each group, so both GPUs train at the same time
    3 stops the training cleanly before Kaggle's 12-hour session limit (--hours, default 11.25);
      every finished epoch is already saved, so nothing is lost but the epoch in progress
    4 writes /kaggle/working/<models>_training.zip (copy its folders into
      Prototype/outputs/training/ on the laptop) and prints a per-fold status

To continue an interrupted session: add the previous notebook version's output as an input and
pass --resume-from /kaggle/input/<that output>.

Options
    --models m3            models to train (keys of configs/models.yaml)
    --folds 2018 2019 ...  default: all four
    --hours 11.25          wall-clock budget for training
    --resume-from PATH     earlier output (folder or zip) to continue from
    --smoke                tiny models, 2 epochs (checks the setup in minutes)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORKING_ROOT", "/kaggle/working"))
SCRATCH = Path(os.environ.get("KAGGLE_SCRATCH_ROOT", "/tmp"))
FOLDS = [2018, 2019, 2020, 2021]


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [kaggle] {msg}", flush=True)


def find_cache() -> Path:
    infos = sorted(
        p for p in INPUT.rglob("cache_info.json") if "Prototype" not in p.parts
    )
    if not infos:
        sys.exit(
            f"[kaggle] no cache_info.json under {INPUT}: add the array-cache dataset as input"
        )
    root = infos[0].parent
    if any(root.glob("*/x.npy")):
        return root
    zips = sorted(root.glob("*.zip"))
    if not zips:
        sys.exit(f"[kaggle] {root} has cache_info.json but no event folders or zips")
    out = SCRATCH / "wsts_cache"
    if not (out / ".extracted").exists():
        say(f"extracting {len(zips)} cache archives to {out} (one time per session)")
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "cache_info.json", out / "cache_info.json")
        for z in zips:
            with zipfile.ZipFile(z) as f:
                names = f.namelist()
                # Kaggle's CLI zips each event folder; the archive may or may not hold the folder.
                target = (
                    out
                    if any(n.startswith(z.stem + "/") for n in names)
                    else out / z.stem
                )
                f.extractall(target)
        (out / ".extracted").touch()
    return out


def gpu_count() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return 0
    return sum(1 for line in r.stdout.splitlines() if line.startswith("GPU "))


def restore(resume_from: Path, out_training: Path) -> None:
    src = resume_from
    if src.suffix == ".zip":
        tmp = SCRATCH / "resume"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(src) as f:
            f.extractall(tmp)
        src = tmp
    found = [p.parent for p in src.rglob("*.summary.json")] + [
        p.parent for p in src.rglob("*.last.pt")
    ]
    for fold_dir in sorted(set(found)):
        dst = out_training / fold_dir.parent.name / fold_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in fold_dir.iterdir():
            shutil.copy2(f, dst / f.name)
        say(f"restored {fold_dir.parent.name}/{fold_dir.name}")


def status(out_training: Path, models: list[str]) -> None:
    for m in sorted(p.name for p in out_training.iterdir() if p.is_dir()):
        for fold_dir in sorted((out_training / m).glob("fold_*")):
            done = list(fold_dir.glob("*.summary.json"))
            if done:
                s = json.loads(done[0].read_text())
                say(
                    f"{m} {fold_dir.name}: FINISHED, validation AUPRC {s['val_auprc']:.4f}, "
                    f"best epoch {s['best_epoch']}, settings fingerprint {s['config_hash'][:8]}"
                )
            elif list(fold_dir.glob("*.last.pt")):
                hist = list(fold_dir.glob("*.history.csv"))
                n = max(0, len(hist[0].read_text().splitlines()) - 1) if hist else 0
                say(
                    f"{m} {fold_dir.name}: UNFINISHED after {n} epochs (resume with --resume-from)"
                )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2],
    )
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=FOLDS)
    ap.add_argument("--hours", type=float, default=11.25)
    ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None, help="override, e.g. cpu (testing only)")
    a = ap.parse_args()
    t0 = time.time()
    deadline = t0 + a.hours * 3600

    proto = WORK / "Prototype"
    if not proto.exists():
        say("copying the code into the working folder")
        shutil.copytree(HERE / "Prototype", proto)
    out_root = proto / "outputs" / ("smoke" if a.smoke else "")
    out_training = out_root / "training"
    if a.resume_from:
        restore(a.resume_from, out_training)
    cache = find_cache()
    data_root = (
        SCRATCH / "data_root_placeholder"
    )  # train_all checks for a 2018 folder only
    (data_root / "2018").mkdir(parents=True, exist_ok=True)

    n_gpu = 0 if a.device == "cpu" else gpu_count()
    groups = [a.folds[i::n_gpu] for i in range(n_gpu)] if n_gpu > 1 else [a.folds]
    groups = [g for g in groups if g]
    workers = max(1, (os.cpu_count() or 2) // len(groups))
    say(
        f"{n_gpu} GPU(s), {os.cpu_count()} CPUs: fold groups {groups}, {workers} loader workers each"
    )

    procs = []
    for i, folds in enumerate(groups):
        cmd = [
            sys.executable,
            "scripts/train_all.py",
            "--data-root",
            str(data_root),
            "--cache-dir",
            str(cache),
            "--only",
            "train",
            "--models",
            *a.models,
            "--folds",
            *map(str, folds),
            "--workers",
            str(workers),
        ]
        if a.smoke:
            cmd.append("--smoke")
        if a.device:
            cmd += ["--device", a.device]
        env = dict(os.environ, PYTHONPATH=str(proto / "ml" / "src"))
        if n_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        procs.append(subprocess.Popen(cmd, cwd=proto, env=env))
        say(f"started folds {folds}" + (f" on GPU {i}" if n_gpu else ""))

    stopped = False
    while any(p.poll() is None for p in procs):
        if not stopped and time.time() > deadline:
            say(
                "time budget reached: stopping after saving (the current epoch is lost)"
            )
            for p in procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
            stopped = True
            stop_at = time.time()
        if stopped and time.time() - stop_at > 300:
            for p in procs:
                if p.poll() is None:
                    p.kill()
        time.sleep(15)

    codes = [p.returncode for p in procs]
    say(
        f"training processes ended with codes {codes} after {(time.time() - t0) / 3600:.2f} h"
    )
    if out_training.exists():
        status(out_training, a.models)
        name = "_".join(a.models) + ("_smoke" if a.smoke else "") + "_training.zip"
        zpath = WORK / name
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for f in sorted(out_training.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out_training))
        say(
            f"wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB): unzip it into "
            "Prototype/outputs/training/ on the laptop"
        )
    # Keep the saved output small: the training state is in the zip; drop the code copy.
    shutil.rmtree(proto, ignore_errors=True)
    if any(c != 0 for c in codes) and not stopped:
        sys.exit("[kaggle] a training process failed; see the log above")


if __name__ == "__main__":
    main()
