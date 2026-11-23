# Results

Written by `python scripts/train_all.py` (steps compare and report). Do not edit by hand: rerun.
Start with `summary.md`.

| File | What it holds |
|---|---|
| `summary.md` | the selected model and why, test metrics per fold for every model and baseline, gate decisions, training table, list of figures |
| `best_model.json` | the model the website and the live forecast use (rule: best mean validation AUPRC over the four folds) |
| `metrics/test_metrics_by_fold.csv` | pooled AUPRC, event-macro AUPRC, F1, IoU, Brier, BSS and ECE (raw and calibrated) per model and fold |
| `metrics/training_runs.csv` | one row per model and fold: best epoch, epochs run, validation AUPRC, minutes, parameters, GPU, batch |
| `metrics/training_history.csv` | one row per epoch: training loss, learning rate, validation AUPRC / Brier / ECE |
| `figures/01_test_auprc_by_fold.png` | test AUPRC per fold, all models and both soft baselines |
| `figures/02_precision_recall_curves.png` | precision-recall curves on each held-out year |
| `figures/03_training_curves.png` | learning curves of every model and fold |
| `figures/04_reliability.png` | reliability of the selected model, raw and calibrated |
| `figures/05_per_event_vs_baseline.png` | per-fire AUPRC difference to the strongest baseline |
| `comparison/model_comparison.md` | full comparison tables with 95 % bootstrap intervals and every gate |
| `comparison/gate_decisions.json` | the same decisions, machine-readable (read by the website) |
| `comparison/ablation_results.csv` | per fold and model: pooled and event-macro metrics (read by the website) |
| `comparison/event_bootstrap_results.csv` | every paired comparison with its interval and verdict |
| `comparison/per_event_differences.csv` | per-fire metric differences behind each comparison |

Until the first full training run only the baseline rows exist (`comparison/` shows every model
as NOT RUN).
