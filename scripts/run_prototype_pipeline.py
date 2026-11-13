"""Run the whole FireGuard offline pipeline on a small staged data root (Integration, Prototype).

    python scripts/run_prototype_pipeline.py --data-root <folder holding 2018..2021> --work <folder>

One command that chains every role's code, in the order the Phase 3/4 guide defines, and stops at
the first failing step:

    1. Role 1  build_phase3_data_release.py      file audit + input manifest of the staged events
    2. Role 3  train_fold                         M3 stacked-days U-Net, one fold, PLUMBING budget
    3. Role 3  export_oof                         held-out probability GeoTIFFs + pre-reveal records
    4. Role 1  build_replay_catalogue.py          every raster checked against its source frame
    5. Role 4  validate_oof_and_build_reveal.py   OOF/fold mapping, metric check, reveal package
    6. Integ.  build_backend_registry.py          Role 4 package -> Role 5 layout + registry bundle

The backend step (Role 5 database load) runs in the backend's own environment, so this script
prints that command at the end instead of running it.

WHAT THIS IS NOT
    It is not the training procedure. Training here is deliberately tiny (base width 8, a few
    epochs, one fold) and exists only to prove the pieces connect. Real training uses
    scripts/train_temporal.py, which stays blocked until Gate 2 sets training.enabled: true.
    Because of that, every output of this script is forced to MOCK at the backend boundary
    (build_backend_registry.py --mode auto sees the plumbing marker and/or the synthetic-data
    marker and the disabled training gate), so nothing produced here can be published or shown
    as a FireGuard result.

Works on the synthetic demo data (scripts/make_synthetic_wsts.py) and on a staged subset of the
real WSTS archive. With real data it is the "plumbing check" recorded in
docs/evidence/role1/end_to_end_plumbing_check.json, now reproducible with one command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from fireguard_ml.data.wsts_dataset import (
    DATA_CONTRACT_VERSION,
    WSTSTemporalSource,
    scan_events,
)
from fireguard_ml.inference.temporal_oof import export_oof
from fireguard_ml.training.temporal_trainer import (
    ProtocolFiles,
    TemporalRunConfig,
    train_fold,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "configs" / "temporal_unet.yaml"
SYNTHETIC_MARKER = "SYNTHETIC_NOT_WSTS.txt"


def step(n: int, title: str) -> float:
    """Print a step header and return the start time (so each step reports its duration)."""
    print(f"\n=== [{n}/6] {title}", flush=True)
    return time.time()


def run(cmd: list[str]) -> None:
    """Run another repository script with this interpreter; any non-zero exit stops the pipeline."""
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True,
                    help="output: data release, checkpoints, artifacts, registry")  # fmt: skip
    ap.add_argument(
        "--fold", type=int, default=2020, help="P1 fold to train and export"
    )
    ap.add_argument(
        "--run-id", default=None, help="default: prototype-<synthetic|plumbing>-m3"
    )
    ap.add_argument(
        "--epochs", type=int, default=3, help="plumbing budget (kept tiny on purpose)"
    )
    ap.add_argument(
        "--base", type=int, default=8, help="U-Net width (32 in the real config)"
    )
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    data_root, work = a.data_root.resolve(), a.work.resolve()
    synthetic = (data_root / SYNTHETIC_MARKER).exists()
    label = "synthetic" if synthetic else "plumbing"
    run_id = a.run_id or f"prototype-{label}-m3"
    cfg_y = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data_cfg = cfg_y["data"]
    events = sorted(scan_events(data_root))
    if not events:
        raise SystemExit(f"no WSTS events under {data_root}")
    print(json.dumps({"data_root": str(data_root), "synthetic_data": synthetic,
                      "events_on_disk": len(events), "fold": a.fold, "run_id": run_id,
                      "label": "MOCK / NOT A RESULT"}, indent=2))  # fmt: skip

    # Shared protocol files (Role 4's frozen manifests). The trainer re-verifies the split hash.
    proto = ProtocolFiles(
        ROOT / data_cfg["split_manifest"],
        ROOT / data_cfg["split_hashes"],
        ROOT / data_cfg["eval_samples"],
        ROOT / data_cfg["strict_channel_manifest"],
    )
    T = data_cfg["n_leading_observations"]
    channels = tuple(data_cfg["channels"])
    release_dir = work / "data_release"
    artifact_root = work / "ml_artifacts"
    run_dir = work / "checkpoints"

    # 1. Role 1: audit exactly the staged events and write their input manifest outside the repo.
    t = step(1, "Role 1 data release (staged events only, written to the work folder)")
    run([SCRIPTS / "build_phase3_data_release.py", "--data-root", data_root,
         "--events", *events, "--out-dir", release_dir])  # fmt: skip
    print(f"    done in {time.time() - t:.0f}s")

    # 2. Role 3: train one fold of M3 with a plumbing budget. train_fold is called directly
    # (not train_temporal.py) because train_temporal.py correctly refuses real data while
    # training.enabled is false; the outputs of this run are forced to MOCK in step 6.
    t = step(
        2,
        f"Role 3 train M3 fold {a.fold} (PLUMBING budget: base {a.base}, {a.epochs} epochs)",
    )
    cfg = TemporalRunConfig(
        model="m3_stacked_unet",
        T=T,
        channels=channels,
        base=a.base,
        depth=cfg_y["models"]["m3"]["depth"],
        max_epochs=a.epochs,
        patience=a.epochs,
        grad_accum=1,
        amp=False,
        seed=0,
        preprocessing_version=DATA_CONTRACT_VERSION,
        extra={"prototype_plumbing_run": True, "synthetic_data": synthetic},
    )
    source = WSTSTemporalSource(data_root, channels, T, proto.eval_samples)
    summary = train_fold(cfg, source, proto, a.fold, run_dir, a.device)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("run", "fold", "best_epoch", "val_auprc", "seconds")
            }
        )
    )
    print(f"    done in {time.time() - t:.0f}s")

    # 3. Role 3: export held-out forecasts with the fold checkpoint (checks fold mapping itself).
    t = step(3, "Role 3 export OOF forecasts")
    folds = {a.fold: run_dir / f"fold_{a.fold}" / f"{cfg.run_name()}.pt"}
    print(
        json.dumps(
            export_oof(run_dir, folds, source, proto, artifact_root, run_id, a.device)
        )
    )
    print(f"    done in {time.time() - t:.0f}s")

    # 4. Role 1: verify every raster against its source frame and build the replay catalogue.
    t = step(4, "Role 1 replay catalogue (raster vs source frame checks)")
    run([SCRIPTS / "build_replay_catalogue.py", "--artifact-root", artifact_root,
         "--run-id", run_id, "--input-manifest",
         release_dir / "manifests" / "phase3_input_manifest.csv"])  # fmt: skip
    print(f"    done in {time.time() - t:.0f}s")

    # 5. Role 4: independent OOF validation, metric recomputation and the reveal package.
    t = step(5, "Role 4 OOF validation + reveal package")
    run([SCRIPTS / "validate_oof_and_build_reveal.py", "--artifact-root", artifact_root,
         "--run-id", run_id, "--data-root", data_root, "--folds", a.fold])  # fmt: skip
    print(f"    done in {time.time() - t:.0f}s")

    # Mark the run as a plumbing run where the bridge (and any human) will look for it.
    (artifact_root / "runs" / run_id / "PROTOTYPE_PLUMBING_RUN.json").write_text(
        json.dumps({"plumbing_run": True, "synthetic_data": synthetic,
                    "reason": "tiny training budget before Gate 2; not a FireGuard result"},
                   indent=2) + "\n", encoding="utf-8")  # fmt: skip

    # 6. Integration: Role 4 reveal package -> Role 5 artifact layout and registry bundle.
    t = step(6, "Bridge to the Role 5 backend registry (mode auto -> mock here)")
    backend_artifacts = work / "backend_artifacts"
    registry = work / "registry"
    run([SCRIPTS / "build_backend_registry.py", "--artifact-root", artifact_root,
         "--run-id", run_id, "--data-root", data_root,
         "--data-release", release_dir / "evidence" / "phase3_data_release.json",
         "--backend-artifact-root", backend_artifacts, "--out", registry])  # fmt: skip
    print(f"    done in {time.time() - t:.0f}s")

    print(
        "\nPipeline finished. Load it into the backend (backend environment, from backend/):"
    )
    print(f'    $env:FIREGUARD_ARTIFACT_ROOT = "{backend_artifacts}"')
    print(
        f'    python scripts/load_registry_bundle.py --bundle "{registry / "registry_bundle.json"}"'
    )


if __name__ == "__main__":
    main()
