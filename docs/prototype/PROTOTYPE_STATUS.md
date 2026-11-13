# Prototype status (2026-11-13)

What the integrated `integration/prototype` branch contains, what has been checked, and what is
still open. Status words follow the Phase 3/4 guide section 7: `DONE`, `CONDITIONAL`, `BLOCKED`,
`NOT RUN`, `DEFERRED`, `UNAVAILABLE`. "Code done" never means "result exists": **no model has been
trained on real data and no FireGuard metric exists yet.**

## 1. Checks run on this branch (fresh clone, Linux, Python 3.11, Node 22)

| Suite | Command | Result |
|---|---|---|
| ML lint | `ruff check src tests` (in `ml/`) | clean |
| ML tests | `pytest` (in `ml/`) | **265 passed, 1 skipped** (the skip needs `FIREGUARD_WSTS_ROOT`) |
| Scripts | `python -m compileall -q scripts` | clean |
| Backend lint | `ruff check src tests` (in `backend/`) | clean |
| Backend migrations | `alembic upgrade head` | head `c31f6a2b9d40` |
| Backend tests | `pytest tests` | **61 passed** |
| Backend runtime separation | import `fireguard.main`, check `sys.modules` | torch not imported |
| Frontend | `npm run typecheck`, `npm test`, `npm run build` | clean, **15 passed**, build OK |
| End-to-end demo | `docs/evidence/integration/prototype_demo_run.json` | every stage passed on SYNTHETIC data (see 3) |

Not run here: anything on the real dataset (not in this environment), GPU training, PostgreSQL, live NASA
FIRMS (no key), Windows. Run section 2 of `RUNBOOK.md` on your machine to confirm.

## 2. Architecture as integrated

```
WSTS GeoTIFFs ──► Role 1 WSTSTemporalSource ──► Role 3 trainer (M3/M4)   Role 2 U-Net (M1)
 (or synthetic)    40 ch, strict 7, pad/mask       fold-specific norm      spatial OOF export*
                         │                                │
                         ▼                                ▼
              Role 1 data release              Role 3 export_oof: probability.tif +
              (input manifest, hashes)         oof_forecasts.jsonl + post_reveal scores
                         │                                │
                         └──────────► Role 1 replay catalogue (raster == source frame)
                                                          │
                                       Role 4 OOF validation + metric recomputation
                                       + reveal package (pre_reveal / reveal_store)
                                                          │
                                       Integration bridge: build_backend_registry.py
                                       (Role 5 layout, per-day metrics, mock/real gate)
                                                          │
                                       Role 5 registry (load_registry_bundle.py ->
                                       register_manifest) -> retrieve-only replay API
                                                          │
                                       Role 6 frontend: forecast first, reveal after
   NASA FIRMS ──► Role 5 proxy/cache ──────────────────► Role 6 Live monitor (observation only)
```

`*` Role 2's spatial OOF export writes its own provisional format, not Role 3's; see 6.2.

## 3. What the demo proves (and does not)

On synthetic data in the exact WSTS file format (real event IDs, dates and frames), one command
ran: Role 1 release (8 events, 85 TIFFs, 0 problems) -> Role 3 M3 training (fold 2020, plumbing
budget) -> export (17 forecasts) -> Role 1 catalogue (17 verified, 0 rejected) -> Role 4 validation
(17 accepted, 0 rejected, 0 metric mismatches, 17 reveal sessions) -> bridge (mock, three reasons)
-> backend registry (17 forecasts, status mock, publish refused) -> API and UI.

Security checks on the live API: reveal before forecast `409 FORECAST_NOT_VIEWED`; target, metrics
and a guessed target artifact before reveal `403 ACTUAL_NOT_REVEALED`; artifact without session
rejected; target served with `Cache-Control: private, no-store`; Role 4's `find_target_leaks` on
the event, origin, inputs and forecast responses: `[]`; the UI made no post-reveal request before
reveal.

It proves the pieces fit and the leakage rules hold. It says nothing about forecasting real fire.

## 4. Role-by-role

### Role 1: data (Ananya; Phase 3/4 code from the Role 3 session)

| Task | Status | Evidence / what is missing |
|---|---|---|
| P3-R1-01 pinned loader, 23->40 mapping, target semantics | DONE | loader reproduction MATCH (3 events, 6e-8, 0 target mismatches), `docs/data_dictionary.md` 12.1-12.3 |
| P3-R1-02 strict contract | DONE | `configs/data/strict_input_v1.yaml`; aspect sin/cos PROPOSED, not approved |
| P3-R1-03 terrain / land-cover QA | CONDITIONAL | code in the release script; full-archive run NOT RUN |
| P3-R1-04 HLS coverage | NOT RUN | `docs/hls_historical_coverage_audit.md` = FEASIBILITY ONLY |
| P3-R1-05 deterministic batches, masks, alignment | DONE | `WSTSTemporalSource`, tests; exercised end to end |
| P3-R1-06 data release | CONDITIONAL | script done; full run must print `COMPLETE`, 0 problems (RUNBOOK 5) |
| P4-R1-01/02 replay catalogue, raster checks | DONE | exercised on synthetic and on 4 real events |
| P4-R1-03 provenance | DONE | `docs/data_provenance.md`; `afall` excluded |

### Role 2: spatial U-Net (Hasan)

| Task | Status | Evidence / what is missing |
|---|---|---|
| P3-R2-01/02/03 M1, tuning, terrain ablation | BLOCKED | Gate 1 sign-off, then real training |
| P3-R2-04 robustness/efficiency | CONDITIONAL | synthetic only (`reports/phase3/role2_spatial_robustness.json`) |
| P3-R2-05 explanations | NOT STARTED | explanation layer UNAVAILABLE |
| P4-R2-01/02/03 release, parity, OOF entry point | CONDITIONAL / BLOCKED | code and synthetic parity done; no checkpoint |

Integration changes: channel names unified with Role 1 (they differed for 24 of 40), config on
the current split hash with the strict 7 names listed (`strict_approved: false`).

### Role 3: temporal (you)

| Task | Status | Evidence / what is missing |
|---|---|---|
| P3-R3-01/02 M3 / M4 | CONDITIONAL | code, tests, real-sample shape checks; real training BLOCKED (Gate 2) |
| P3-R3-03 sequence ablation T=3 | NOT RUN | predeclared in config |
| P3-R3-04 GFS/FiLM disabled | DONE | strict config + tests; EE checks 1, 2, 5 NOT RUN |
| P3-R3-05 GNN | DEFERRED | |
| P3-R3-06 architecture/branch report | DONE | `reports/phase3/role3_architecture_and_branch_decisions.md`; ablation report NOT RUN |
| P4-R3-01 candidate selection | BLOCKED | Gate 3 |
| P4-R3-02 OOF export | DONE (code) | exercised end to end; real export after training |

### Role 4: evaluation (Sania; Phase 3/4 code from the Role 3 session)

| Task | Status | Evidence / what is missing |
|---|---|---|
| P3-R4-01 frozen P1 | DONE | rule v2, hash `34e29090…`, all hashes re-verified; lead sign-off pending |
| P3-R4-02 comparison protocol | DONE | `docs/model_comparison_protocol.md`; lead sign-off of two rules pending |
| P3-R4-03 ablations, paired bootstrap | NOT RUN | needs M1-M4 results |
| P3-R4-04 observability | CONDITIONAL | script and evaluation-only test; full run NOT RUN; MTBS NOT AVAILABLE |
| P3-R4-05 fold calibration/thresholds | DONE (code) | Platt fix + regression test |
| P3-R4-06 explanations | UNAVAILABLE | |
| P4-R4-01/02/03 OOF validation, reveal, provenance | DONE (code) | exercised end to end |

### Role 5: backend (Tarun)

| Task | Status | Evidence / what is missing |
|---|---|---|
| P3-R5-01..04 registry, layout, validation, runtime separation | DONE | backend tests; migration `c31f6a2b9d40` |
| P4-R5-01..05 catalogue, retrieval, reveal, metrics, tests | DONE | plus the integration `horizon_hours` fix, without which no real artifact could be registered |
| Live FIRMS, PostgreSQL | NOT RUN | needs a key / a Postgres instance |

### Role 6: frontend (no owner work existed; built in this integration)

| Task | Status | Evidence / what is missing |
|---|---|---|
| P3-R6-01..05 shell, layers, mock flow, Live monitor, map/export UX | DONE | `frontend/`, 15 tests, screenshots |
| P4-R6-01..04 real catalogue, retrieval, reveal/compare, downloads | DONE against the real API with mock artifacts | real artifacts after Gate 3 |
| P1-R6-03 / HLS audit (with Role 1) | NOT RUN | |
| Citizen reporting, explainability page | DEFERRED | the guide defers them until the core replay flow works |

## 5. Gates

| Gate | Status | Blocking item |
|---|---|---|
| G1-B leakage register, G1-D frozen P1 | READY FOR LEAD SIGN-OFF | lead |
| Gate 2 (M1 vs strongest baseline) | NOT RUN | Role 2 M1 training + per-event file |
| Gate 3 (M3, M4, terrain) | NOT RUN | Gate 2, then Role 3 training |
| Gate 4 (replay) | CONDITIONAL | mechanics demonstrated on mock artifacts; needs published real OOF artifacts |

## 6. Gaps that remain, with the concrete next step

1. **Real numbers.** Everything in RUNBOOK section 5, then Gate 2. Owner: you (runs), Role 2 (M1).
2. **Role 2's M1 OOF format.** Role 2's `run_spatial_oof_inference` writes a provisional format;
   Roles 1/4 and the bridge read Role 3's (`docs/temporal_oof_export_contract.md`). Simplest fix,
   for Role 2: emit `oof_forecasts.jsonl` + `model_version.json` + `post_reveal/
   per_event_test_scores.csv` with the same fields (or train M1 through Role 3's trainer at T=1
   if Role 2 agrees). Until then M1 can be compared (per-event CSV) but not replayed.
3. **Lead sign-offs** in `docs/phase_gate_log.md`: frozen split, leakage register, the two
   comparison rules, aspect sin/cos.
4. **Messages to send** (unchanged from the hand-off): new P1 hash to everyone; Role 2: use
   `WSTSTemporalSource(..., T=1)`, `binary_active_fire_mask`, the per-event CSV +
   `model_version.json` format; Role 5: the `horizon_hours` fix and `load_registry_bundle.py`
   (please review, it is your code area); Role 1 (Ananya): section 12 of the dictionary and the
   `--out-dir` option.
5. **Two files cited but never committed** by their owners (INTEGRATION_LOG section 7).
6. **Not verified on Windows by me:** the PowerShell commands in RUNBOOK. They follow the repo's
   existing conventions; report any that fail.
