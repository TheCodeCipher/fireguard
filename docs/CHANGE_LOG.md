# Change log: Prototype overhaul (November 2026)

What changed in the Prototype before the full training run, and why. Grouped by topic; the newest
pass is at the bottom. Older history is in `docs/history/`.

## 1. Workspace clean-up

| Change | Why |
|---|---|
| Removed the old branches' working copies (Archive, branches, fireguard, role-1, role-4) outside `Prototype/` | They were older or un-merged versions of the same work; everything that mattered is in `Prototype/`. |
| Removed the `mscf` ("simulate output") frontend module | Unused, and it showed made-up output. |
| Deleted the one-off `scripts/cleanup_workspace.ps1`, `finish_m1.ps1`, `demo_fold2018*.ps1`, `train_temporal.py`, `build_m1_preview_results.py` | Replaced by `scripts/train_all.py`; the clean-up script had done its job. |
| Deleted old outputs (`outputs/m1`, `prototype_work`, `artifacts/runs`, `artifacts/reveal_packages`, the M1 fold-2018 live preview) | Produced by the old trainer with old settings; every model is retrained. |

## 2. Models (`ml/src/fireguard_ml/models/architectures.py`, `configs/models.yaml`)

| Change | Why |
|---|---|
| New M2: attention residual U-Net (residual blocks, attention gates on the skip connections, day D only) | The plan's M2 ("U-Net + terrain + land cover") cannot be built honestly: land cover fails the leakage audit (a whole-year product, so it contains the future) and terrain is already in every model's inputs. With the new M2 each step changes one thing: M2 vs M1 = network design, M3 vs M1 = five days of history, M4 vs M3 = recurrence. |
| "M1 without terrain" kept as an optional ablation (`--ablations`) | Answers the plan's terrain question without spending a main model slot on it. |
| M4's ConvLSTM memory at the two coarsest levels (was the coarsest only) | Lets the recurrent memory see fire-front detail one level finer, at a modest parameter cost. |
| GroupNorm, light dropout (0.1, bottleneck only) in all four models | Batch sizes are small (GPU memory), where BatchNorm is unreliable; dropout is light because early stopping already limits over-fitting. |
| `temporal_unet.py` renamed `architectures.py`; `MODEL_REGISTRY` lists M1-M4 | The file holds all four models, not one. |
| Gate 3 pairs updated (`evaluation/comparison.py`, `docs/evaluation/model_comparison_protocol.md`) | They now match the one-change-per-step design above. |

## 3. Training (`ml/src/fireguard_ml/training/trainer.py`, `hardware.py`, `data/frame_cache.py`)

| Change | Why |
|---|---|
| Diagnosis of M1's "best epoch 1" | Epochs count from 0, so it was the second epoch. M1 did learn (training loss 0.66 to 0.49 in epoch 0) but stalled, barely above the distance-decay baseline (validation AUPRC 0.277 vs 0.2625). |
| Shuffled frames across fires (was one fire at a time) | Each update was dominated by a single fire. |
| Batch size measured on the GPU, gradient accumulation to 16 samples per update, bf16/fp16 mixed precision | Batch size 1 gave noisy updates; the GPU sat at 0.5 of 6 GB. |
| One-time float16 array cache next to the dataset (`../wsts_cache`, about 16 GB) | Decoding GeoTIFFs every epoch starved the GPU. Parity with the GeoTIFF reader is tested (max relative difference 5e-4). |
| Warm-up then cosine learning-rate decay, AdamW without weight decay on norms and biases, gradient clipping | Standard, stable schedule; the fixed learning rate plateaued. |
| Flips as augmentation (the aspect channel is negated on horizontal flips) | More variety without changing the physics; aspect must follow the flip. |
| Early stopping after 6 epochs without improvement (was 3), up to 40 epochs | The validation curve is noisy; 3 stopped too early. |
| Resume: each fold saves its latest state every epoch plus its best weights; after the fold only the best checkpoint stays | A power cut at epoch 5 of M2 resumes at epoch 6 of M2, not at M1. Tested: an interrupted run ends identical to an uninterrupted one. |
| Platt calibration and the F1 threshold fitted on validation only | The test year never influences any choice. |

## 4. Evaluation and report (`scripts/train_all.py`, `evaluation/report.py`)

| Change | Why |
|---|---|
| `scripts/train_all.py`: one command for cache, baselines, train, evaluate, compare, report, publish; every step skips finished work | One resumable command instead of a dozen manual ones. |
| All four models train back to back; gates are computed afterwards | No waiting for sign-off between models. Recorded for the lead in `docs/evaluation/phase_gate_log.md` (2026-11-14 entry). |
| Best model = highest mean validation AUPRC over the four folds | Selected without looking at test years. |
| `results/`: `summary.md`, `best_model.json`, metrics CSVs, five figures, comparison and calibration | Everything the report needs in one folder; `docs/REPORT_GUIDE.md` maps it to report sections. |
| `--smoke` mode | Tiny models, 2 epochs, outputs under `outputs/smoke/`: checks the setup in minutes without touching real outputs. |

## 5. Live forecast and website

| Change | Why |
|---|---|
| Live forecast uses the model named in `results/best_model.json`, with calibrated probabilities | It was hard-wired to an M1 preview; only the best model should be live. |
| Thresholds converted to the calibrated scale | They were applied to raw scores. |
| Keys read from `backend/.env` (`FIRMS_MAP_KEY`, `EARTHENGINE_PROJECT`) | One git-ignored place for secrets. |
| Reflectance falls back to the newest earlier day when today's image is not in Earth Engine yet | The live run failed whenever the latest day was late. |
| "Not validated" label instead of refusing when the model did not beat the baseline (`--require-gate` refuses) | The forecast is still useful to look at, but must be labelled honestly. |
| "Live forecast" and "Live monitor" tabs merged into one **Live** tab (forecast and detections as layers, plus a check against the next day's detections) | They showed the same place and day; the forecast tab had been hidden. |
| Results tab reads `results/comparison/`; the M1 preview card was removed | Shows the real comparison, not a preview. |

## 6. Structure and documentation

| Change | Why |
|---|---|
| `scripts/` split into `pipeline/`, `live/`, `data/`, `checks/`, `app/` with a README | A flat folder of 30 scripts was hard to navigate. |
| `docs/` grouped into `data/`, `evaluation/`, `contracts/`, `history/`, `project/`, `evidence/`, each listed in `docs/README.md` | Same reason; `reports/` was merged in. |
| Four READMEs merged into the root `README.md`; `docs/RUNBOOK.md`, `docs/REPORT_GUIDE.md`, `results/README.md` added | One entry point, one runbook, one report guide. |
| Commented-out dead code removed from `data/wsts_dataset.py` and the old trainer | About 350 lines that nothing ran. |
| Future work (dilated ConvLSTM; transfer learning to regions without enough data) in `docs/REPORT_GUIDE.md` | Requested for the report. |
| CI workflow path updated to `docs/data/leakage_audit.csv` | The file moved. |

Open decision: Role 2's original spatial U-Net path (`models/unet.py`, `inference/spatial.py`,
`training/config.py`, `smoke_run.py`, `real_sample_benchmark.py`, `configs/unet.yaml`,
`ml/tests/`) is a second M1 that nothing uses. Recommended for deletion once Role 2 agrees.

## 7. One-click start (2026-11-15)

| Change | Why |
|---|---|
| `scripts/app/start_app.py` + `FireGuard - Start app.bat`: migrations, loads the best model's bundle (once per bundle), rebuilds the website only when its code or `results/` changed, serves site and API together on port 8000, opens the browser | Starting the app took two terminals and five commands. |
| Backend serves the built website when `FIREGUARD_FRONTEND_DIST` is set (`backend/src/fireguard/main.py`, `core/config.py`, test `tests/integration/test_frontend_serving.py`) | One process and one port instead of a dev server plus an API. Unset (the default), the backend is unchanged. |
| `FireGuard - Train all.bat`, `FireGuard - Live forecast.bat` | The other two routine jobs, one double-click each. The `.bat` files use `Prototype/.venv`, then `WildFireDetection/.venv`, then `python` on PATH. |
| README, RUNBOOK sections 2-4 and `scripts/README.md` updated | Point to the double-click files; the manual commands stay documented. |

## 8. Training run and fixes found during it (2026-11-15 to 11-17)

| Change | Why |
|---|---|
| PyCharm project repaired (`.idea/FireGuard.iml`, `.idea/modules.xml`; project renamed "FireGuard") | Renaming the project to "FireGuard: WildFireDetection" made PyCharm write a module file with `:` in its name, which Windows forbids; with no module, the interpreter and terminal activation stopped working. Outside the repository. |
| `scripts/kaggle/kaggle_train.py`, `build_package.py`; RUNBOOK "Training one model on Kaggle" | M3 and M4 were trained on Kaggle's two T4s (one fold per GPU) while the laptop trained M1 and M2, cutting the total from about 2 days to about 1. |
| Native bfloat16 only on compute capability 8.0+ (`training/hardware.py`) | Newer PyTorch reports bfloat16 as supported on a T4 through slow emulation; T4s now use float16 with loss scaling. Laptop (RTX 3060, Ampere) unchanged. |
| GPU memory budget for the batch-size probe lowered from 80 % to 65 % (`training/hardware.py`) | During M2, PyTorch's memory cache grew from the probed 4.45 GB to about 5.3 GB of the 6 GB card; Windows spilled the rest into system RAM and epochs became 4-5x slower. M2 folds 2019-2021 were finished with `--batch-size 4`, M4 used batch 8 on the T4s; the update size stayed 16 throughout, so all runs are comparable. |
| Result | All 16 runs (4 models x 4 folds) finished with the expected settings fingerprints. M2 fold 2020 reached the 40-epoch cap (best epoch 36), the only run not ended by early stopping. |
