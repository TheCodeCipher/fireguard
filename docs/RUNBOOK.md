# Runbook: every command, in order

Run everything from the `Prototype` folder in the JetBrains terminal (it already uses the
project's Python environment). Paths use `/`, which works in PowerShell too. The dataset is
expected at `../WildfireSpreadTS/WildfireSpreadTS` (the folder that holds `2018`..`2021`).

## 1. Check the setup (a few minutes)

```
python -m pip install -e "ml[dev,scripts]" -e "backend[dev]"
cd ml
python -m pytest
cd ..
python scripts/train_all.py --data-root ../WildfireSpreadTS/WildfireSpreadTS --smoke --models m1 m4 --folds 2018
```

The first `train_all.py` call also builds the array cache (`../wsts_cache`, about 16 GB, once;
roughly 15-30 minutes). `--smoke` trains tiny models for 2 epochs into `outputs/smoke/` and never
touches the real outputs. It passes when it ends with `=== done`. Delete `outputs/smoke` afterwards.

## 2. Train, evaluate, compare and publish everything

```
python scripts/train_all.py --data-root ../WildfireSpreadTS/WildfireSpreadTS
```

(or double-click `FireGuard - Train all.bat`, which runs the same command). It runs, in order: cache, baselines (only if missing), training of M1, M2, M3, M4 on the four
leave-one-year-out folds, test-set evaluation, comparison and gates, the report pack in `results/`,
and publishing of the best model for the website. Each fold logs one line per epoch; the full log
is `outputs/train_all.log`.

**If it stops** (power cut, sleep, Ctrl+C): run the same command again. Finished folds are skipped
and an unfinished fold continues after its last completed epoch. Only the best checkpoint of each
fold is kept once the fold is done.

Before a long run: plug in the laptop and set Windows to never sleep. The batch size and data
loading workers are chosen automatically for the GPU; override with `--batch-size` / `--workers`
if needed.

Optional extra experiments (predeclared ablations: M1 without terrain, M3 with 3 days, M3 at M4's
size), after the main run:

```
python scripts/train_all.py --data-root ../WildfireSpreadTS/WildfireSpreadTS --ablations
```

Run one step only with `--only cache|baselines|train|evaluate|compare|report|publish`.

### Training one model on Kaggle (optional, runs alongside the laptop)

Any model's folds can be trained on Kaggle's free T4 x2 GPUs (one fold per GPU at a time) and
copied back; the laptop accepts them because a finished fold is recognised by its settings
fingerprint, not by the machine that trained it.

1. `python scripts/kaggle/build_package.py` writes `outputs/kaggle/fireguard_kaggle_code.zip`.
2. On kaggle.com create two private datasets: the zip above, and the array cache folder
   (`../wsts_cache`, about 16 GB; zip it first or upload it with the Kaggle CLI).
3. New notebook, both datasets as inputs, Accelerator "GPU T4 x2", one cell:
   `!python $(find /kaggle/input -name kaggle_train.py | head -1) --models m3`
   (add `--folds ...` for some folds only), then Save Version > Save & Run All.
4. It stops itself before the 12-hour session limit. Its output `<models>_training.zip` is
   unzipped into `outputs/training/`. An unfinished fold continues in a new session with
   `--resume-from /kaggle/input/<previous output>`.
5. Train the remaining models on the laptop with `--only train --models ...` so it skips the
   Kaggle ones, then run the full command once everything is in `outputs/training/`.

The November 2026 run trained M3 and M4 this way (`outputs/training/*/fold_*/*.summary.json`
record the GPU each fold used).

## 3. See the results in the website

Double-click `FireGuard - Start app.bat`, or:

```
python scripts/app/start_app.py
```

It applies the database migrations, loads the best model's registry bundle (named by
`results/best_model.json`, once per bundle), rebuilds the website when its code or `results/`
changed, then serves the website and the API together on http://127.0.0.1:8000 and opens the
browser. Close the window (or Ctrl+C) to stop. Options: `--port`, `--no-browser`, `--rebuild`.
Node.js must be installed (the first start runs `npm ci`). Development mode below assumes the
launcher has run once, so the database and the best model's bundle are already loaded.

For website development with hot reload, run the two parts separately instead:

```
cd backend
python -m uvicorn fireguard.main:app --port 8000
```
```
cd frontend
npm run dev
```

and open http://localhost:5173.

## 4. Live forecast (after section 2)

One-time setup:

```
python -m pip install -r ml/requirements-live.txt
earthengine authenticate
```

Keys are read from `backend/.env`: `FIRMS_MAP_KEY` (already set) and `EARTHENGINE_PROJECT` (your
Google Cloud project with Earth Engine enabled). `.env` files are git-ignored.

Check once that the live inputs match the training data, and let it set the fire-mask options:

```
python scripts/live/check_live_input_parity.py --data-root ../WildfireSpreadTS/WildfireSpreadTS --fit-v2-mapping --write-config
python scripts/live/run_live_hindcast.py --data-root ../WildfireSpreadTS/WildfireSpreadTS
```

Both must report `PASS` (`docs/evidence/live/`). Then make a forecast:

```
python scripts/live/run_live_forecast.py --dry-run
python scripts/live/run_live_forecast.py
```

It uses the model in `results/best_model.json`, forecasts tomorrow (UTC) for every active fire
cluster in the model's area from yesterday's detections, and publishes to `backend/artifacts/live`.
The **Live** tab shows it next to the satellite detections. If the model did not match or beat the
strongest baseline, the map is labelled "Not validated" (`--require-gate` refuses instead).

Every day: double-click `FireGuard - Live forecast.bat` (it runs
`python scripts/live/run_live_forecast.py`, then `python scripts/live/score_live_forecasts.py`) (scores every past forecast whose day is over; the
running record is `artifacts/live/live_scores.csv`). To automate it, add a daily Windows Task
Scheduler task (around 12:30 IST, when the previous UTC day is complete) that runs those two
commands with the project interpreter (PyCharm: Settings > Python Interpreter shows its path).

## 5. Checks before a pull request

```
cd ml
python -m ruff check src tests
python -m pytest
cd ../backend
python -m ruff check src tests
python -m pytest tests
cd ../frontend
npm run typecheck
npm test
npm run build
```

Never commit `outputs/`, `artifacts/`, checkpoints or `.env` (all git-ignored; check `git status`).

## Troubleshooting

**Epochs suddenly several times slower (Windows):** the GPU's own memory is full and Windows is
spilling into system RAM (Task Manager > Performance > GPU: "Shared GPU memory" well above zero).
Press Ctrl+C and rerun the same command with `--batch-size 4` (or 2 for M4): the update size stays
16 through gradient accumulation, so results are unchanged, and the run resumes from its last epoch.

| Symptom | Fix |
|---|---|
| `... has no 2018 folder` | `--data-root` must point at the folder holding `2018`..`2021` |
| CUDA out of memory | rerun with a smaller `--batch-size` (the automatic choice leaves 20 % headroom) |
| `... holds a cache for other channels` | delete `../wsts_cache` (it is rebuilt) or pass `--cache-dir` |
| `[evaluate] ... not all four folds trained yet` | training is incomplete; rerun the same command |
| `NOT READY: ...` from the live forecast | the published model changed since the last run of section 2 |
| `FIRMS_MAP_KEY is not set` | add it to `backend/.env` |
| `Earth Engine is not initialised` | `earthengine authenticate` and set `EARTHENGINE_PROJECT` in `backend/.env` |
| Results tab shows no models | restart `npm run dev` after `train_all.py` finished |
