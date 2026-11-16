# FireGuard

Next-day wildfire activity forecasting on the WildfireSpreadTS benchmark (607 western-US fires,
2018-2021). For a fire that is already burning, FireGuard predicts the probability that each
375 m cell shows a VIIRS active-fire detection the next day, compares four deep-learning models
against simple baselines under leave-one-year-out testing, and serves the best model through a
website with a historical replay and a daily live forecast.

## Start here

Double-click, in this folder (each opens a window that shows progress):

| File | What it does |
|---|---|
| `FireGuard - Start app.bat` | website + API on http://127.0.0.1:8000, opens the browser (builds the site and loads the best model first when needed) |
| `FireGuard - Train all.bat` | trains, evaluates and publishes every model; safe to re-run after an interruption |
| `FireGuard - Live forecast.bat` | tomorrow's forecast with the best model, then scores past forecasts |

| I want to... | Go to |
|---|---|
| run everything (train, evaluate, publish) | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) |
| read the results / write the report | [`results/summary.md`](results/summary.md) (after training), [`docs/REPORT_GUIDE.md`](docs/REPORT_GUIDE.md) |
| understand a folder or find a document | the map below, then [`docs/README.md`](docs/README.md) |

## Repository map

```
configs/        settings: models.yaml (the 4 models + training), evaluation.yaml (frozen),
                live_forecast.yaml, data/strict_input_v1.yaml (allowed input channels)
data/manifests/ frozen split (leave-one-year-out), scored samples, event metadata
ml/             the Python package fireguard_ml (models, training, evaluation, live) + tests
backend/        FastAPI server: replay API, live-forecast API, NASA FIRMS proxy
frontend/       React website: Overview, Replay lab, Results, Live, Method
scripts/        command-line entry points; train_all.py runs the whole pipeline (see scripts/README.md)
results/        everything the report needs, written by scripts/train_all.py
docs/           documentation, grouped by topic (docs/README.md)
notebooks/      exploratory data analysis of WSTS
```

Generated and git-ignored: `outputs/` (training checkpoints and scores), `artifacts/` (published
forecast maps), `backend/artifacts/` (what the website serves), and the array cache next to the
dataset (`../wsts_cache`).

## Inside `ml/src/fireguard_ml/`

| Folder | What is in it |
|---|---|
| `data/` | reading WSTS GeoTIFFs (`wsts_dataset.py`), the fast array cache (`frame_cache.py`), the frozen split (`event_splits.py`), fire-complex grouping, allowed-channel checks |
| `models/` | M1-M4 (`architectures.py`), ConvLSTM cell (`convlstm.py`), checkpoint format (`checkpointing.py`) |
| `training/` | one-fold training with resume (`trainer.py`), hardware planning (`hardware.py`), loss, normalisation, sample contract (`temporal_data.py`) |
| `evaluation/` | metrics, calibration, paired bootstrap, model comparison and gates, the report pack (`report.py`), replay checks |
| `inference/` | held-out (out-of-fold) forecast export (`temporal_oof.py`) |
| `baselines/` | persistence, distance decay, climatology |
| `live/` | live forecasting: FIRMS detections, Earth Engine inputs, grids, readiness checks, publishing |
| `tests/` | the test suite (`python -m pytest` in `ml/`) |

## The models

| | Input | Design | Question it answers |
|---|---|---|---|
| M1 | day D | U-Net, plain conv blocks | reference CNN |
| M2 | day D | residual blocks + attention-gated skips | does a stronger spatial design help? |
| M3 | days D-4..D | the 5 days stacked as channels | does recent history help? |
| M4 | days D-4..D | shared per-day encoder + ConvLSTM memory | does recurrence beat stacking? |

Baselines: persistence, dilated persistence, distance decay, distance-bin climatology. The model
with the best mean validation score is published to the website and used for the live forecast.
Details: `ml/src/fireguard_ml/models/architectures.py` and [`docs/REPORT_GUIDE.md`](docs/REPORT_GUIDE.md).

## Team

Role 1 data (Ananya), Role 2 spatial U-Net (Hasan), Role 3 temporal models (Arkodyuti),
Role 4 evaluation (Sania), Role 5 backend (Tarun), Role 6 frontend (Nikhil). Changes go through
pull requests; never push to `main`.

Data: WildfireSpreadTS (Gerard et al., 2023), NASA FIRMS. Research prototype, not for operational
use.
