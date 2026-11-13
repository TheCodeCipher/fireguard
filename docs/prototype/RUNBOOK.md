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

Expected: see `PROTOTYPE_STATUS.md` section 1 (ML 265 passed + 1 skipped, backend 61 passed,
frontend 15 passed). The one ML skip runs on real data with
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

## 4. Real-data plumbing check (optional, before Gate 2)

Same chain on a few real events; still forced to mock in the backend because training is gated.

```powershell
cd $P
python scripts\run_prototype_pipeline.py --data-root $WSTS --work prototype_work\real_plumbing --events fire_25294989 fire_21997854 fire_25294983 fire_24461965 fire_24604426 --epochs 3 --device cuda
```

## 5. Real runs still owed (the "needs you" list)

These need the full dataset and were never run. Each must be run as written; paste the printed
summary back for the docs.

```powershell
cd $P
python scripts\build_phase3_data_release.py --data-root $WSTS
python scripts\run_observability_diagnostic.py --data-root $WSTS
python scripts\benchmark_temporal_memory.py --T 5 3
```

Pass conditions: the release prints `"status": "COMPLETE"` and `"n_problems": 0`; the memory
benchmark fits M4 at T=5 on 6 GB (if it reports out of memory, stop and record it). Never commit
`data\evaluation_only\next_day_observability.csv` (it is git-ignored by path rules; check
`git status` anyway). Earth Engine check 5 (GRIDMET day window) is
`scripts\ee_checks_gfs_gridmet.js`, run in the Earth Engine code editor; its result goes into
`docs\gfs_channel_audit.md` section 8.7 and `docs\leakage_audit.csv`.

## 6. After Gate 2 is recorded (real results, publishable)

Only when `docs/phase_gate_log.md` records Gate 2 = BEATS or MATCHES and the lead has signed off.

1. Set `training.enabled: true` in `configs/temporal_unet.yaml` (one-line PR, cite the gate row).
2. Train and export (per `docs/handoff/role3_phase4_handoff.md`):

```powershell
python scripts\train_temporal.py --model m3_stacked_unet --T 5 --source wsts --data-root $WSTS --out outputs\temporal --preprocessing-version wsts_strict_input_v1
python scripts\export_temporal_oof.py --run-dir outputs\temporal --run m3_stacked_unet_T5_seed0 --source wsts --data-root $WSTS --artifact-root artifacts --run-id m3_t5_v1 --device cuda
python scripts\build_replay_catalogue.py --artifact-root artifacts --run-id m3_t5_v1
python scripts\validate_oof_and_build_reveal.py --artifact-root artifacts --run-id m3_t5_v1 --data-root $WSTS
python scripts\build_backend_registry.py --artifact-root artifacts --run-id m3_t5_v1 --data-root $WSTS --backend-artifact-root backend\artifacts --out artifacts\registry\m3_t5_v1
```

   `build_backend_registry.py` prints `"mode": "real"` only when no mock condition holds.
3. Register (validated, not yet visible), then publish after approval:

```powershell
cd $P\backend
$env:FIREGUARD_ARTIFACT_ROOT = "$P\backend\artifacts"
python scripts\load_registry_bundle.py --bundle "$P\artifacts\registry\m3_t5_v1\registry_bundle.json"
python scripts\load_registry_bundle.py --bundle "$P\artifacts\registry\m3_t5_v1\registry_bundle.json" --publish
```

Which model gets exported (M3 or M4, T=5 or T=3) is decided by Gate 3 in
`docs/model_comparison_protocol.md`, not by this runbook.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Replay lab says no replays | nothing loaded into this database | run section 3's load step with the same `DATABASE_URL` |
| `REGISTRY VALIDATION FAILED: mock bundles load only with FIREGUARD_APP_ENV=development` | env var missing | `$env:FIREGUARD_APP_ENV = "development"` |
| `ARTIFACT_CHECKSUM_MISMATCH` / not found in the UI | `FIREGUARD_ARTIFACT_ROOT` differs from the one used when loading | use the same folder for loading and serving |
| `refusing --mode real: ...` | a mock condition holds (synthetic data, plumbing run, training disabled) | intended; see the listed reasons |
| Map has no background | basemap tiles need internet | forecast layers still draw; or set `VITE_BASEMAP_STYLE` in `frontend\.env.local` |
| `train_temporal.py: BLOCKED: training.enabled is false` | Gate 2 not recorded | intended |
