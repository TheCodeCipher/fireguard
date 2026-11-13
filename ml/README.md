# FireGuard Machine Learning (`ml/`)

The single Python package for data loading, models, training, offline OOF export and evaluation:
`ml/src/fireguard_ml`. Every ML role works in this one package; there is no second copy anywhere
in the repository (Role 2's former standalone `FireGuard_Role2_Starter/` was folded in here during
the Prototype integration, see `docs/prototype/INTEGRATION_LOG.md`).

The primary task is next-day active-fire forecasting on the WSTS benchmark. All experiments follow
the frozen P1 LOYO protocol (`data/manifests/p1_split_hashes.txt`) and the strict 7-channel input
contract `wsts_strict_input_v1` (`configs/data/strict_input_v1.yaml`).

## Who owns what

| Subpackage / file | Owner | Purpose |
|---|---|---|
| `data/wsts_dataset.py`, `data/geo_artifacts.py` | Role 1 | WSTS reader (`WSTSTemporalSource`), 40-channel processing identical to the pinned loader, raster/artifact checks |
| `data/channel_schema.py`, `data/padding.py`, `data/sample_adapter.py`, `data/temporal_adapter.py` | Role 2 | Named channel selection, pad-to-32/unpad, model-side sample adapters |
| `models/unet.py`, `training/losses.py`, `training/config.py`, `inference/spatial.py` | Role 2 | M1 U-Net, masked BCE+Dice, validated config, target-blind spatial OOF export |
| `models/temporal_unet.py`, `models/convlstm.py`, `models/checkpointing.py` | Role 3 (spatial checkpoint helpers: Role 2) | M3 stacked-days U-Net, M4 ConvLSTM U-Net, self-describing checkpoints |
| `training/temporal_data.py`, `training/normalization.py`, `training/temporal_trainer.py`, `inference/temporal_oof.py` | Role 3 | Temporal sample contract, fold-specific normalisation, training loop, temporal OOF export |
| `baselines/`, `evaluation/`, `data/event_splits.py`, `data/fire_complex_groups.py`, `data/channel_policy.py`, `data/observability.py` | Role 4 | Baselines, metrics, calibration, bootstrap, comparisons/gates, OOF validation, reveal package, split and leakage rules |
| `tests/test_integration_contracts.py` | Integration | Checks the agreements between roles (names, strict list, split hash, padding) |

## Install and test (Python 3.11, as fixed by the project plan)

PowerShell 5.1 (one command per line):

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e "ml[dev,scripts]"
cd ml
python -m ruff check src tests
python -m pytest
```

Expected: **all tests pass with 1 skipped** (the real-WSTS test, which runs when
`$env:FIREGUARD_WSTS_ROOT` points at the dataset). The exact count is recorded in
`docs/prototype/PROTOTYPE_STATUS.md`. On a GPU machine install the CUDA build of PyTorch instead
of the CPU one. Role 2's exact tested versions are in `constraints-py311-cpu.txt`
(`python -m pip install -c ml/constraints-py311-cpu.txt -e "ml[dev]"`).

The `scripts` extra (rasterio, pyproj, matplotlib, pyyaml) is needed by everything in the
repository's `scripts/` folder.

## Layout notes

- The package is `fireguard_ml`, not `fireguard`: `backend/src/fireguard` owns that name.
- Tests: Role 2's in `ml/tests/` (they resolve `../configs/unet.yaml` from `ml/`), everyone
  else's in `ml/src/fireguard_ml/tests/`. Both are collected through `testpaths` in
  `ml/pyproject.toml`.
- Configs: `../configs/unet.yaml` (M1, Role 2), `../configs/temporal_unet.yaml` (M3/M4, Role 3),
  `../configs/evaluation.yaml` (Role 4), `../configs/data/strict_input_v1.yaml` (Role 1).
- Real training is gated: `training.enabled: false` in `configs/temporal_unet.yaml` and
  `strict_approved: false` in `configs/unet.yaml` until Gate 2 and the lead's sign-offs are
  recorded in `docs/phase_gate_log.md`.
