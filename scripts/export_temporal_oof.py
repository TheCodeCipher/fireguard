"""Export out-of-fold forecasts for the selected temporal model (Role 3, P4-R3-02).

    python scripts/export_temporal_oof.py --run-dir outputs/temporal \
        --run m3_stacked_unet_T5_seed0 --source wsts --data-root <WSTS folder> \
        --artifact-root artifacts --run-id m3_t5_v1

Each fold's checkpoint predicts only that fold's test events; the mapping is checked against the
frozen manifest. Writes runs/<run-id>/... under --artifact-root (see inference/temporal_oof.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fireguard_ml.inference.temporal_oof import export_oof
from fireguard_ml.training.temporal_data import make_source
from fireguard_ml.training.temporal_trainer import ProtocolFiles

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--run", required=True, help="run name, e.g. m3_stacked_unet_T5_seed0")
    ap.add_argument("--folds", type=int, nargs="+", default=[2018, 2019, 2020, 2021])
    ap.add_argument("--source", choices=["wsts", "synthetic"], required=True)
    ap.add_argument("--data-root", type=Path, default=None, help="folder holding 2018..2021")
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "temporal_unet.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--events", nargs="+", default=None,
                    help="PROTOTYPE preview: export only these test events (run stays mock)")
    a = ap.parse_args()

    data = yaml.safe_load(a.config.read_text())["data"]
    proto = ProtocolFiles(
        ROOT / data["split_manifest"],
        ROOT / data["split_hashes"],
        ROOT / data["eval_samples"],
        ROOT / data["strict_channel_manifest"],
    )
    summary = json.loads((a.run_dir / f"fold_{a.folds[0]}" / f"{a.run}.summary.json").read_text())
    T = summary["config"]["T"]
    source = make_source(
        a.source,
        samples_csv=proto.eval_samples,
        channels=data["channels"],
        T=T,
        data_root=a.data_root,
    )
    if source.is_mock and "mock" not in a.run_id.lower():
        raise SystemExit("synthetic exports must have 'mock' in --run-id so nobody mistakes them")
    folds = {f: a.run_dir / f"fold_{f}" / f"{a.run}.pt" for f in a.folds}
    print(
        json.dumps(export_oof(a.run_dir, folds, source, proto, a.artifact_root, a.run_id, a.device,
                              set(a.events) if a.events else None))
    )


if __name__ == "__main__":
    main()
