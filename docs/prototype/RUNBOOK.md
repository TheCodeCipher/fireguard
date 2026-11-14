# Prototype runbook (Windows PowerShell 5.1)

One command per line, no `&&`. Paths assume the integrated repository is at
`C:\Users\ad\PycharmProjects\WildFireDetection\Prototype` and the dataset at
`C:\Users\ad\PycharmProjects\WildFireDetection\WildfireSpreadTS\WildfireSpreadTS`. Adjust the two
variables below if yours differ. Run them first in every new PowerShell window.

```powershell
$P = "C:\Users\ad\PycharmProjects\WildFireDetection\Prototype"
$WSTS = "C:\Users\ad\PycharmProjects\WildFireDetection\WildfireSpreadTS\WildfireSpreadTS"
```

## 0. The repository

`Prototype` is a full git clone: branch `integration/prototype`, `origin` =
`https://github.com/TheCodeCipher/fireguard.git`, with all role branches as `origin/*` refs.
To share it as a pull request (never push to `main`):

```powershell
cd $P
git status -sb
git log --oneline -12
git push -u origin integration/prototype
```

Then open a PR `integration/prototype -> main` on GitHub for the lead. The commit author on the
integration commits is `Arkodyuti Bhattacharyya <ark0b@users.noreply.github.com>`; if you want your
usual address, run `git config user.email <yours>` before any further commits (existing commits
can stay as they are).

## 1. One-time setup

Python 3.11 (project plan). One virtual environment is enough on your machine; the backend never
imports PyTorch even if it is installed (a test checks this).

```powershell
cd $P
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e "ml[dev,scripts]"
python -m pip install -e "backend[dev]"
cd frontend
npm install
cd ..
```

(`cu121` = CUDA build for the RTX 3060. On a CPU-only machine use `.../whl/cpu`.)

## 2. Check everything (what CI runs)

```powershell
cd $P\ml
python -m ruff check src tests
python -m pytest
cd $P\backend
python -m ruff check src tests
python -m pytest tests
cd $P\frontend
npm run typecheck
npm test
npm run build
cd $P
```

Expected: see `PROTOTYPE_STATUS.md` section 1 (ML 268 passed + 1 skipped, backend 62 passed,
frontend 17 passed). The one ML skip runs on real data with
`$env:FIREGUARD_WSTS_ROOT = $WSTS`.

## 3. Demo on synthetic data (about 5 minutes on a laptop CPU)

Everything below runs the real code of every role. The data are synthetic and every replay is
labelled mock in the API and the UI.

```powershell
cd $P
python scripts\make_synthetic_wsts.py --out demo_data\synthetic_wsts
python scripts\run_prototype_pipeline.py --data-root demo_data\synthetic_wsts --work prototype_work\demo --epochs 25
```

Load it into the backend (fresh SQLite database in the work folder):

```powershell
cd $P\backend
$env:FIREGUARD_APP_ENV = "development"
$env:FIREGUARD_ARTIFACT_ROOT = "$P\prototype_work\demo\backend_artifacts"
$env:DATABASE_URL = "sqlite:///$($P -replace '\\','/')/prototype_work/demo/fireguard_demo.db"
python -m alembic upgrade head
python scripts\load_registry_bundle.py --bundle "$P\prototype_work\demo\registry\registry_bundle.json"
python -m uvicorn fireguard.main:app --port 8000
```

In a second PowerShell window:

```powershell
cd C:\Users\ad\PycharmProjects\WildFireDetection\Prototype\frontend
npm run dev
```

Open http://localhost:5173, go to **Replay lab**, pick a fire and a day, **Load forecast**, then
**Reveal actual fire**. API docs: http://127.0.0.1:8000/docs.

Live monitor: add `FIRMS_MAP_KEY=<your key>` to `backend\.env` (create it from
`backend\.env.example`) and restart uvicorn. Without a key the page offers the backend's labelled
four-point fixture.

## 4. Real data: two-minute smoke check (do this first)

Runs every role's code on five real WSTS events (fold 2021, one tiny model) to prove your
machine, CUDA and the data path work before the long run. Verified on these exact five events
in the integration sandbox (2026-11-13). The output is labelled mock: it covers only part of a fold.

```powershell
cd $P
python scripts\run_prototype_pipeline.py --data-root $WSTS --work prototype_work\real_smoke --fold 2021 --events fire_25294989 fire_25294983 fire_21997854 fire_24333267 fire_21615469 --epochs 3 --device cuda
```

Pass: it ends with `Pipeline finished.` and step 6 prints `"mode": "mock"` with the reason
`OOF coverage ... < 1.0`. If CUDA is not found, check `python -c "import torch; print(torch.cuda.is_available())"`.

## 5. Real data: train M1 and run Gate 2

This is the work everything else waits for. M1 (`m1_spatial_unet`) is the shared U-Net on day D
only; it is Gate 2's own evidence, so it trains with `training.enabled: false` (that flag gates
M3/M4 only). Run from `$P`, one line at a time.

```powershell
cd $P
python scripts\build_phase3_data_release.py --data-root $WSTS
python scripts\run_baselines.py --data-root $WSTS
python scripts\train_temporal.py --model m1_spatial_unet --source wsts --data-root $WSTS --out outputs\m1 --preprocessing-version wsts_strict_input_v1 --device cuda
python scripts\export_temporal_oof.py --run-dir outputs\m1 --run m1_spatial_unet_T1_seed0 --source wsts --data-root $WSTS --artifact-root artifacts --run-id m1_t1_v1 --device cuda
python scripts\build_replay_catalogue.py --artifact-root artifacts --run-id m1_t1_v1
python scripts\validate_oof_and_build_reveal.py --artifact-root artifacts --run-id m1_t1_v1 --data-root $WSTS
python scripts\run_model_comparison.py --model m1_unet=artifacts\runs\m1_t1_v1\post_reveal\per_event_test_scores.csv
python scripts\build_backend_registry.py --artifact-root artifacts --run-id m1_t1_v1 --data-root $WSTS --backend-artifact-root backend\artifacts --out artifacts\registry\m1_t1_v1
```

What each line does and what to look for:

| Line | Does | Pass condition |
|---|---|---|
| data release | Role 1 audit of every TIFF + input manifest | `"status": "COMPLETE"`, `"n_problems": 0` |
| baselines | Role 4 baselines + the cache Gate 2 needs (`outputs\role4\`) | finishes; numbers equal `docs\baseline_results.md` |
| train | M1, all four folds (15 epochs max, early stop after 3 without validation gain) | prints a progress line every 500 samples and one per epoch; checkpoints in `outputs\m1\fold_<Y>\` |
| export | held-out forecast GeoTIFF for every test sample, fold model per fire | `"forecasts"` about 10,572 minus training-only samples |
| catalogue | every raster checked against its source frame | `0 rejected` |
| validate | Role 4's independent check + reveal package | `"coverage": 1.0`, `"metric_mismatches": 0` |
| comparison | Gate 2 verdict, writes `reports\phase3\*` | prints the Gate 2 decision (BEATS / MATCHES / FAILS) |
| registry | backend bundle | `"mode": "real"` |

Time: the training reads each frame from disk; on an RTX 3060 expect very roughly 5 to 15 minutes
per epoch per fold, so a few hours for all four folds. Watch the first epoch's `seconds` value
and multiply. To run one fold at a time, add `--folds 2018` (then 2019, 2020, 2021) to the train
line; export needs all four before the comparison.

Then record the result: paste the comparison output into `docs\phase_gate_log.md` (Gate 2 row)
for the lead. If Gate 2 = BEATS or MATCHES and the lead signs off, section 7 unlocks M3/M4.
If it FAILS, that is a valid, reportable result (plan section on Gate 2): stop and diagnose, do
not train a bigger model to hide it.

## 6. See the real M1 replays in the site

```powershell
cd $P\backend
$env:FIREGUARD_APP_ENV = "development"
$env:FIREGUARD_ARTIFACT_ROOT = "$P\backend\artifacts"
$env:DATABASE_URL = "sqlite:///$($P -replace '\\','/')/backend/fireguard_real.db"
python -m alembic upgrade head
python scripts\load_registry_bundle.py --bundle "$P\artifacts\registry\m1_t1_v1\registry_bundle.json" --publish
python -m uvicorn fireguard.main:app --port 8000
```

Second window: `cd $P\frontend` then `npm run dev`, open http://localhost:5173. The Results tab
reads `reports\phase3\` at build time, so after the comparison it shows the M1 column and the
Gate 2 verdict (restart `npm run dev` if it was already running). `--publish` here is for your
local database; publishing on a shared deployment needs the lead's approval.

## 7. After Gate 2 is recorded: M3 / M4

Only when `docs/phase_gate_log.md` records Gate 2 = BEATS or MATCHES and the lead has signed off.

1. Set `training.enabled: true` in `configs/temporal_unet.yaml` (one-line PR, cite the gate row).
2. Train, export and compare exactly as section 5, with `--model m3_stacked_unet` (run name
   `m3_stacked_unet_T5_seed0`, run id e.g. `m3_t5_v1`) and then `--model m4_convlstm_unet`, adding
   each to the comparison: `--model m1_unet=... --model m3_stacked_unet=... --model m4_convlstm_unet=...`.
3. Which model the replay publishes is decided by Gate 3 in `docs/model_comparison_protocol.md`.

Other real runs still owed: `python scripts\run_observability_diagnostic.py --data-root $WSTS`,
`python scripts\benchmark_temporal_memory.py --T 5 3` (M4 at T=5 must fit in 6 GB), and Earth
Engine check 5 (`scripts\ee_checks_gfs_gridmet.js`). Never commit
`data\evaluation_only\next_day_observability.csv`, `outputs\` or `artifacts\` (git-ignored; check
`git status` anyway).

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Replay lab says no replays | nothing loaded into this database | run section 3's load step with the same `DATABASE_URL` |
| `REGISTRY VALIDATION FAILED: mock bundles load only with FIREGUARD_APP_ENV=development` | env var missing | `$env:FIREGUARD_APP_ENV = "development"` |
| `ARTIFACT_CHECKSUM_MISMATCH` / not found in the UI | `FIREGUARD_ARTIFACT_ROOT` differs from the one used when loading | use the same folder for loading and serving |
| `refusing --mode real: ...` | a mock condition holds (synthetic data, plumbing run, training disabled) | intended; see the listed reasons |
| Map has no background | basemap tiles need internet | forecast layers still draw; or set `VITE_BASEMAP_STYLE` in `frontend\.env.local` |
| `train_temporal.py: BLOCKED: training.enabled is false` | M3/M4 before Gate 2 | intended; train M1 (section 5) |
| `fold <Y> <model>: events differ from the frozen test split` | comparison given a subset run | export all four folds of the full dataset |
| `OOF coverage ... < 1.0` in the registry step | subset or interrupted export | re-run export for all folds |
| Results tab still says "not run" after the comparison | the page reads the reports at build time | restart `npm run dev` / rebuild |

## 9. Live forecast (after section 5 has finished)

Live forecasting reads the training outputs and never writes to them: it writes only under
`artifacts\live\` and `backend\artifacts\live\`. It runs on CPU. Do not run any of it while
the section 5 commands are still running; they need the machine and re-import the code.

1. **Install the extra packages**, pinned to what training used so pip cannot move torch / numpy:

   ```powershell
   python -m pip freeze | Select-String '^(torch|numpy|scipy|rasterio)==' | Out-File -Encoding ascii $env:TEMP\live-pin.txt
   python -m pip install -r ml\requirements-live.txt -c $env:TEMP\live-pin.txt
   earthengine authenticate
   $env:EARTHENGINE_PROJECT = "<your Google Cloud project>"
   $env:FIRMS_MAP_KEY = "<your FIRMS key>"
   ```

2. **Smoke test** (synthetic checkpoints, about a minute, writes only `prototype_work\live_smoke\`):
   `python scripts\smoke_live_forecast.py`. It must print `"status": "PASSED"`.

3. **Input parity** (needs FIRMS + Earth Engine; uses no training output):

   ```powershell
   python scripts\check_live_input_parity.py --data-root $WSTS --fit-v2-mapping
   ```

   Read `docs\evidence\live\live_input_parity.json`. Copy `active_fire_best` into
   `configs\live_forecast.yaml` (`rasterize_rule`, `day_utc_offset_hours`). If the `v2`
   reflectance check fails, stop: live forecasts would be built on shifted inputs.

4. **Hindcast** (scores each event with the fold that held it out):

   ```powershell
   python scripts\run_live_hindcast.py --data-root $WSTS --run-dir outputs\m1 --run m1_spatial_unet_T1_seed0 --run-id m1_t1_v1 --artifact-root artifacts
   ```

   `docs\evidence\live\live_hindcast.json` must say `"status": "PASS"`.

5. **First live day**: a dry run (readiness + regions, no writes), then the real run:

   ```powershell
   python scripts\run_live_forecast.py --run-dir outputs\m1 --run m1_spatial_unet_T1_seed0 --run-id m1_t1_v1 --artifact-root artifacts --backend-artifact-root backend\artifacts --dry-run
   python scripts\run_live_forecast.py --run-dir outputs\m1 --run m1_spatial_unet_T1_seed0 --run-id m1_t1_v1 --artifact-root artifacts --backend-artifact-root backend\artifacts
   ```

   It refuses (`NOT READY`) until Gate 2 is recorded as BEATS or MATCHES. `--allow-ungated`
   overrides only that and the registry check, and stamps the output UNGATED (red banner).
   Start the API with the same `FIREGUARD_ARTIFACT_ROOT` (`backend\artifacts`) and open the
   **Live forecast** tab.

6. **Every day** (Task Scheduler, about 06:30 UTC): `run_live_forecast.py` as above (origin =
   yesterday UTC by default), then score the previous run once its target day has passed:
   `python scripts\score_live_forecasts.py --artifact-root artifacts --live-run-id live_<origin>_m1_t1_v1`.
   Scores accumulate in `artifacts\live\live_scores.csv` next to dilated persistence.

| Symptom | Cause | Fix |
|---|---|---|
| `NOT READY: ... differs from the one exported` | a checkpoint changed after `export_temporal_oof.py` | re-run section 5 from the export step |
| `NOT READY: Gate 2 decision is 'NOT_RUN'` | section 5 not finished | finish section 5, or `--allow-ungated` for development |
| `TerrainUnavailable` / `earthengine-api is not installed` | step 1 not done | install and authenticate Earth Engine |
| Live forecast tab: "No live forecast yet" | no run published to this artifact root | step 5, same `FIREGUARD_ARTIFACT_ROOT` |
