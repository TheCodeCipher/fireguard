# Scripts

Every script runs from the `Prototype` folder: `python scripts/<folder>/<name>.py --help`.

| Script | What it does |
|---|---|
| **`train_all.py`** | **the one command**: cache, baselines, train M1-M4 on all folds, evaluate, compare, report, publish the best model (docs/RUNBOOK.md) |

`app/` - running the website

| Script | What it does |
|---|---|
| `start_app.py` | database migrations, loads the best model, rebuilds the site when needed, serves site + API on one port (`FireGuard - Start app.bat`) |

`pipeline/` - the steps `train_all.py` calls (run them by hand only to redo one step)

| Script | What it does |
|---|---|
| `run_baselines.py` | persistence, distance-decay and climatology baselines on every fold |
| `export_temporal_oof.py` | held-out forecasts of a trained model (`--scores-only` for metrics without maps) |
| `run_model_comparison.py` | paired bootstrap tests, Gate 2 / Gate 3 decisions -> `results/comparison/` |
| `run_calibration_report.py` | reliability tables for a model -> `results/calibration/` |
| `build_replay_catalogue.py` | checks every forecast map against its source frame (replay catalogue) |
| `validate_oof_and_build_reveal.py` | independent validation of the held-out forecasts + reveal package |
| `build_backend_registry.py` | bundle the website loads (`backend/scripts/load_registry_bundle.py`) |

`live/` - daily live forecasting with the best model

| Script | What it does |
|---|---|
| `run_live_forecast.py` | forecast tomorrow for today's fires and publish it to the website |
| `score_live_forecasts.py` | score past live forecasts against the detections that followed |
| `check_live_input_parity.py` | one-time check that live inputs match the training data |
| `run_live_hindcast.py` | one-time check that the model keeps its skill on live-built inputs |
| `smoke_live_forecast.py` | end-to-end test of the live path on synthetic data |

`data/` - building and auditing the dataset and the frozen split (done in phases 1-3)

| Script | What it does |
|---|---|
| `build_phase3_data_release.py` | audit every GeoTIFF, write the input manifest |
| `build_groups_and_p1_splits.py` | fire-complex groups and the leave-one-year-out split (frozen) |
| `derive_event_footprints.py` | event footprints used for grouping |
| `wsts_raster_grid_audit.py` | grid/CRS consistency of every raster |
| `verify_against_pinned_loader.py` | our reader vs the official WSTS loader |
| `make_synthetic_wsts.py` | small fake dataset in WSTS format for tests and demos |

`checks/` - one-off audits, benchmarks and the synthetic demo

| Script | What it does |
|---|---|
| `audit_gfs_channels.py`, `check_gfs_precip_break.py`, `ee_checks_gfs_gridmet.js` | GFS / gridMET timing audits (why weather is excluded) |
| `run_observability_diagnostic.py` | metrics split by how observable the next day was |
| `smoke_temporal_real_sample.py`, `benchmark_temporal_memory.py` | early shape and memory checks of M3/M4 |
| `benchmark_spatial_inference.py`, `export_spatial_oof.py` | Role 2's original spatial U-Net path (not used by the pipeline) |
| `run_prototype_pipeline.py` | 5-minute demo of every part on synthetic data |
