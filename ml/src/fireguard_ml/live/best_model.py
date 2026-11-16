"""Which trained model the live forecast uses: the one ``results/best_model.json`` names.

``scripts/train_all.py`` writes that file after comparing M1-M4 (rule in
``evaluation/report.py``) and publishes the chosen model's held-out forecasts as
``artifacts/runs/<run_id>``. Live forecasting reads the same choice, so there is one source of
truth and nothing to type by hand once training has finished.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BestModel:
    model: str      # report name, e.g. m3_stacked_unet
    run_dir: Path   # outputs/training/<model>
    run: str        # run name, e.g. m3_stacked_unet_T5_seed0
    run_id: str     # published version, e.g. m3_stacked_unet_v1


def resolve_best_model(repo_root: Path, results: Path | None = None) -> BestModel:
    path = Path(results or Path(repo_root) / "results") / "best_model.json"
    if not path.is_file():
        raise SystemExit(f"{path} is missing: run scripts/train_all.py (it chooses the model), "
                         "or pass --run-dir, --run and --run-id explicitly")
    b = json.loads(path.read_text(encoding="utf-8"))
    run_dir = Path(b["training_dir"])
    if not run_dir.is_absolute():
        run_dir = Path(repo_root) / run_dir
    return BestModel(b["model"], run_dir, b["run"], b["published_run_id"])
