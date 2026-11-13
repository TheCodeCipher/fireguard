"""Train M1 / M3 / M4 for one or more P1 folds (Role 3, P3-R3-02/03; M1 added by Integration).

    python scripts/train_temporal.py --model m3_stacked_unet --T 5 --folds 2018 2019 2020 2021 \
        --source wsts --data-root <WSTS folder> --out outputs/temporal

``--source wsts`` (needs Role 1's ``fireguard_ml.data.wsts_dataset`` and ``--data-root``) is the
only source that produces reportable results. ``--source synthetic`` runs the identical fold loop on MOCK
data keyed to the real manifest (a pipeline dry run; its numbers mean nothing).

Real training of M3/M4 also requires ``training.enabled: true`` in configs/temporal_unet.yaml,
which may only be set once Gate 2 is recorded. M1 (``m1_spatial_unet``, T=1) is Gate 2's own
evidence and is never blocked by that flag (INTEGRATION, Prototype). Checkpoints go under --out;
outputs/ and *.pt are git-ignored, keep it that way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fireguard_ml.training.temporal_data import make_source
from fireguard_ml.training.temporal_trainer import ProtocolFiles, TemporalRunConfig, train_fold

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model", required=True, choices=["m1_spatial_unet", "m3_stacked_unet", "m4_convlstm_unet"]
    )
    ap.add_argument("--T", type=int, default=None, help="context length (default: config)")
    ap.add_argument("--folds", type=int, nargs="+", default=[2018, 2019, 2020, 2021])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=None, help="width override (size-matched M3 only)")
    ap.add_argument("--source", choices=["wsts", "synthetic"], required=True)
    ap.add_argument("--data-root", type=Path, default=None, help="folder holding 2018..2021")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "temporal_unet.yaml")
    ap.add_argument("--preprocessing-version", default="UNSET")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-epochs", type=int, default=None, help="override (dry runs only)")
    ap.add_argument("--samples-per-epoch", type=int, default=None, help="cap (dry runs only)")
    a = ap.parse_args()

    cfg_y = yaml.safe_load(a.config.read_text())
    tr, data = cfg_y["training"], cfg_y["data"]
    is_m1 = a.model == "m1_spatial_unet"
    # INTEGRATION: the Gate 2 flag blocks the temporal models only; M1 is what Gate 2 measures.
    if a.source == "wsts" and not tr["enabled"] and not is_m1:
        raise SystemExit(
            "BLOCKED: training.enabled is false (Gate 2 not recorded). Train and compare M1 "
            "first (--model m1_spatial_unet); see docs/prototype/RUNBOOK.md section 5."
        )
    if a.source == "synthetic" and (a.max_epochs or a.samples_per_epoch) is None:
        a.max_epochs, a.samples_per_epoch = 2, 64
    m = next(v for v in cfg_y["models"].values() if v["model_type"] == a.model)
    T = 1 if is_m1 else (a.T or data["n_leading_observations"])
    allowed_bases = {m["base"], cfg_y["ablation"]["m3_size_matched_base"]}
    if a.base is not None and (a.base not in allowed_bases or a.model != "m3_stacked_unet"):
        raise SystemExit(f"--base {a.base} is not a predeclared width for {a.model}")
    if not is_m1 and T not in cfg_y["ablation"]["context_lengths"]:
        raise SystemExit(f"T={T} is not a predeclared context length {cfg_y['ablation']}")
    cfg = TemporalRunConfig(
        model=a.model,
        T=T,
        channels=tuple(data["channels"]),
        base=a.base or m["base"],
        depth=m["depth"],
        lstm_kernel=m.get("lstm_kernel", 3),
        lr=tr["lr"],
        weight_decay=tr["weight_decay"],
        max_epochs=a.max_epochs or tr["max_epochs"],
        patience=tr["patience"],
        grad_accum=tr["grad_accum"],
        amp=tr["amp"],
        seed=a.seed,
        bce_weight=tr["bce_weight"],
        dice_weight=tr["dice_weight"],
        train_samples_per_epoch=a.samples_per_epoch,
        preprocessing_version=a.preprocessing_version,
    )
    proto = ProtocolFiles(
        ROOT / data["split_manifest"],
        ROOT / data["split_hashes"],
        ROOT / data["eval_samples"],
        ROOT / data["strict_channel_manifest"],
    )
    source = make_source(
        a.source, samples_csv=proto.eval_samples, channels=cfg.channels, T=T, data_root=a.data_root
    )
    if source.is_mock:
        print("*** MOCK DATA: synthetic pipeline dry run. Results are not P1 results. ***")
    for fold in a.folds:
        s = train_fold(cfg, source, proto, fold, a.out, a.device)
        print(
            json.dumps(
                {
                    k: s[k]
                    for k in (
                        "run",
                        "fold",
                        "is_mock",
                        "best_epoch",
                        "val_auprc",
                        "seconds",
                        "peak_gpu_memory_bytes",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
