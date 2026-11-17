# FireGuard Role 5 — Phase 3 & Phase 4 backend overlay

This is an incremental overlay for the existing Role 5 Phase 2 API/replay implementation. It is **not** a full repository replacement and does not include datasets, real model outputs, credentials, checkpoints, local databases or generated caches.

## Preconditions

Apply this overlay to the FireGuard repository on top of the existing Role 5 Phase 2 implementation (`role-5/backend/phase2-api-replay` or its successor). The repository must already contain:

- Phase 1 migration `a4291d9fbbaa`;
- Phase 2 migration `b6c2d4e91a20`;
- the existing FireGuard backend package and root Alembic configuration.

Do not apply this archive to an older Phase 1-only branch.

## Apply on Windows PowerShell

1. Create/switch to the Role 5 working branch and make sure the working tree is clean:

   ```powershell
   git switch role-5/backend/phase2-api-replay
   git pull --ff-only
   git status --short
   ```

2. Extract this archive into the repository root, preserving `backend/` and `docs/` paths. Existing Phase 2 files in this overlay are intentional updates; review them before staging.

3. Validate the overlay before committing:

   ```powershell
   cd backend
   python -m pip install -e ".[dev]"
   python -m alembic upgrade head
   python -m alembic current
   python -m ruff check src tests
   python -m pytest tests -v
   cd ..
   git diff --check
   git status --short
   ```

   Expected Alembic head: `c31f6a2b9d40`. The migration expects the Phase 1 and Phase 2 revisions already present. Do not delete, rewrite or stamp over those migrations.

## What this adds

- A static OOF artifact registry using the existing `forecast_runs` and `forecast_artifacts` tables.
- Registry manifest validation for OOF status, held-out split, event-complex identity, fold/protocol, next-day 24-hour target, Level-A time-safety, checkpoint/config/data/split/channel hashes, artifact checksums and raster geospatial metadata.
- Artifact lifecycle: first registration produces `validated`; a separate explicit `--publish` operation promotes only unchanged validated records. Published records cannot be demoted by revalidation; revoked records are terminal. Development mocks remain mock-only.
- Fail-closed local storage path checks, symlink escape rejection, checksums and size verification.
- Replay retrieval based only on precomputed static files; no Torch/GPU inference in FastAPI.
- Forecast-first / reveal-actual-after session flow, bounded audit history, post-reveal-only target and metric access, and `private, no-store` response headers for post-reveal artifacts.
- OpenAPI, API, artifact-storage and manifest contracts, plus unit/integration tests.

## Registering real OOF artifacts

Run from `backend/` after the data catalogue, model metadata, frozen split evidence and offline files exist:

```powershell
# Validate files and manifest without database writes
python scripts/register_oof_artifacts.py --manifest path\to\oof_manifest.json --dry-run

# First persistent registration; artifacts become validated but are not replay-visible
python scripts/register_oof_artifacts.py --manifest path\to\oof_manifest.json

# Separate publication after evidence approval; requires existing unchanged validated records
python scripts/register_oof_artifacts.py --manifest path\to\oof_manifest.json --publish
```

`--publish` requires both a `probability_map` and `actual_target`. Real post-reveal files (including target-derived metrics) must be under `artifacts/restricted/`. Never commit these files to Git. The manifest template uses placeholder hashes and metadata; do not treat the examples as real evidence.

## Validation performed while preparing this overlay

- Backend test suite: **53 passed** (Python 3.13 environment in this build container).
- Python compile check: passed for `src` and `tests`; all Python source/test lines are within the configured 100-character limit.
- Isolated SQLite migration check from the Phase 2 schema through `c31f6a2b9d40`: passed; legacy Phase 2 `available` artifacts were demoted to `generated`.
- Isolated downgrade check: passed; `generated` artifacts remain non-published after downgrade (fail-closed).
- Ruff fixes were applied for relationship type annotations and 100-character line length. Ruff itself was not available in the build container, so the exact `ruff check` command must still be run in the repository virtual environment.
- A live PostgreSQL integration test and live NASA FIRMS network test were not performed. No real OOF artifacts are included or published.

## Commit after local validation

```powershell
git add backend docs
git commit -m "feat(backend): add Phase 3 and Phase 4 OOF replay registry"
git push origin role-5/backend/phase2-api-replay
```

Integrate into `main` only through the repository's configured review and CI process. Do not bypass branch protection. If the team has renamed the Role 5 branch, push to the current approved Role 5 branch instead.
