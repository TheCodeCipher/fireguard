# Role 2 Phase 3/4 preparation status

**Guide reviewed:** `FireGuard_Phase3_Phase4_Rolewise_Execution_Guide.md` v1.0, 2026-11-13.  
**State:** Independent scaffolding and synthetic checks in progress. **No official model training, P1 comparison, checkpoint release, or real OOF artifact has been produced.**

## Work started within the guide's pre-gate allowance

- Added `adapt_inputs_for_inference()`: it accepts inputs and a source validity mask, selects the configured channels, bottom/right pads, and never accepts a target tensor.
- Added spatial U-Net checkpoint save/load metadata for model version, channel names, data contract, config/split hashes, fold, seed, environment, and train/validation group IDs.
- Added `run_spatial_oof_inference()`: verifies the split hash, resolves the event's unique test fold, checks checkpoint provenance and group isolation, reads only `x` and `input_valid_mask`, and writes probabilities + validity mask + provenance. The release-manifest format is a provisional local contract pending the team-lead-owned Phase 4 contract.
- Added `scripts/export_spatial_oof.py`, a command-line entry point for the offline exporter. It accepts a release manifest, event/sample identifiers and input archive, and prints the provenance metadata after export. A synthetic end-to-end CLI test checks the output artifact and the `target_accessed: false` declaration.
- Added synthetic export parity tests. A deliberately unreadable target member is placed in the fixture archive; the inference path succeeds by reading only inputs and masks. The tests also reject wrong fold/hash/group metadata and exercise full-frame shapes 304x247 and 305x207.
- Added `scripts/benchmark_spatial_inference.py` and a synthetic efficiency report. Two native dimensions pad/unpad correctly; the report records timing, model size, CPU/RAM and process peak memory. It uses a randomly initialized model and synthetic 8-channel data only.
- Added a rare-positive masked-loss sensitivity check. It is an implementation test, not a change to the project evaluation protocol.

## Checks run

ML acceptance suite: **143 passed**; Ruff clean for the ML package and both Role 2 Phase 3/4 scripts. Backend checks are unchanged: 34 passed, Ruff clean, SQLite migration at `b6c2d4e91a20`, Alembic reports no drift.

## Local synthetic efficiency evidence

See `reports/phase3/role2_spatial_robustness.json`. Python 3.11.9, PyTorch 2.15.0+cpu, Intel i5-12450H system (the OS reports build 10.0.26200), 16,890,978,304 bytes RAM, CPU only. U-Net base width 4, 8 synthetic channels, 30,721 parameters; saved synthetic checkpoint 141,437 bytes. The 304x247 case padded to 320x256 and unpadded exactly; mean inference was about 24.75 ms. The 305x207 case padded to 320x224 and unpadded exactly; mean was about 21.31 ms. Process peak working set was about 321.6 MB.

These numbers are synthetic CPU plumbing measurements. They are **not** WSTS performance, accepted compute-budget evidence, or an official M1 benchmark.

## Gate-dependent tasks still not started

- `P3-R2-01/02`: re-establish and tune M1 on the approved data/channel schema and frozen P1 protocol — blocked on Phase 2 / Gate 2 evidence, explicit Global Gate 1, Role 1 masks/loader, Role 4 manifest reconciliation and compute limits.
- `P3-R2-03`: define and run terrain ablation only after the exact channel contract and approved comparisons are recorded.
- `P3-R2-05`: explanations are deferred until a frozen candidate and team-lead output contract exist.
- `P4-R2-01`: no checkpoint release can be frozen because no approved trained M1 checkpoint exists.
- `P4-R2-02`: parity is tested on a synthetic checkpoint; it has not been validated on a representative real WSTS sample or geospatial raster export.
- `P4-R2-03`: fold-safe code and the offline CLI are implemented and synthetically exercised, but there is no actual per-fold trained checkpoint to produce valid OOF probabilities.

## Role 2 task status

| Task | Status | Evidence / outstanding dependency |
|---|---|---|
| `P3-R2-01` M1 under frozen protocol | **BLOCKED** | Official training requires the approved strict schema, validated loader/masks, reconciled frozen P1 manifests and recorded gate approval. |
| `P3-R2-02` validation-only M1 tuning | **BLOCKED** | Requires a valid M1 run and frozen split/calibration protocol. |
| `P3-R2-03` terrain M2 ablation | **BLOCKED** | Requires the agreed input contract and Phase 3 baseline; current MODIS year-Y land cover stays excluded. |
| `P3-R2-04` spatial robustness/efficiency/export checks | **CONDITIONAL — synthetic implementation complete** | Synthetic shape, determinism, target-blind export, CLI and CPU efficiency checks are recorded above; real-data/export acceptance remains pending. |
| `P3-R2-05` explanations | **NOT STARTED** | Start only if a candidate is retained and the team defines the explanation/output contract. |
| `P4-R2-01` freeze spatial checkpoint | **BLOCKED** | No approved trained checkpoint exists. |
| `P4-R2-02` inference/export parity | **CONDITIONAL — synthetic parity complete** | Real WSTS and geospatial raster parity require the approved loader/masks and artifact metadata. |
| `P4-R2-03` fold-safe offline inference | **CONDITIONAL — scaffold complete** | CLI and fold/provenance enforcement are tested with synthetic fixtures; real OOF awaits trained per-fold checkpoints and approved artifact contract. |

## Required before official runs

1. Lead records Global Gate 1 and later Gate 2 status as required by the project guide.
2. Role 1 supplies/validates the pinned loader, full-frame alignment, source masks and nodata semantics.
3. Roles 1/3/4 approve the canonical aspect transformation and ordered strict model channels; do not change 40 to 41 until that is recorded.
4. Role 4 reconciles manifest freeze status and hashes and publishes the authoritative evaluation contract.
5. Lead confirms compute limits and the Role 4 Phase 4 artifact contract.

Until then, keep M1/M2/M3/M4 training and official comparisons off. Phase 4 UI/API work may proceed with clearly labeled mock artifacts in Role 5/6 branches.
