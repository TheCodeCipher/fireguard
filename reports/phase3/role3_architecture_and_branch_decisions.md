# Temporal architecture and branch decisions (Role 3, P3-R3-04/05/06)

Date: 2026-11-13. Status words as in the leakage register: PASS, FAIL, UNKNOWN, NOT RUN.

## 1. M3 versus M4: what each represents

| | M3 stacked-days U-Net | M4 shared encoder + ConvLSTM |
|---|---|---|
| Temporal representation | None explicit. The T frames are concatenated as `T x C` input planes, so time is just more channels and the first convolution mixes days immediately. | Each frame is encoded separately by the same encoder; a ConvLSTM runs over the T bottleneck maps in date order, so order and recurrence are built in. |
| Spatial detail from earlier days | Kept: the stacked input feeds every skip connection, so fine detail from all days reaches the decoder. | Lost below the bottleneck: decoder skips come from the newest frame only; earlier days arrive only through the 1/32-resolution recurrent state. |
| Cost | One encoder pass; first layer grows with T. | T encoder passes; size independent of T. |
| Parameters (C=7, T=5, base 32) | 8.36 M | 13.07 M |

Details, measured sizes and the size-matched control: `docs/temporal_architecture_note.md`.
Hypothesis H2 (recurrence helps) is tested only with paired event-level results on identical folds
(`reports/phase3/temporal_ablation_report.md`). Both models are implemented and tested; neither
has been trained on WSTS data (Gate 2 not recorded). Both pass forward/backward on real WSTS samples.

## 2. Weather and optional branches

| Branch | Decision | Evidence | What would change it |
|---|---|---|---|
| GFS channels 33-37 in M1-M4 | **EXCLUDED** from all strict configs; enforced by config tests and the checkpoint guard | `docs/gfs_branch_gating.md`; register Level A UNKNOWN | Nothing: strict models never take GFS |
| GFS / FiLM benchmark-assumption experiment (H5) | **DISABLED, not trained.** No `gfs_benchmark_assumption.yaml` exists | GFS checks 1 and 2 not run; Level A UNKNOWN (F24 sits on the target-day boundary) | EE checks 1-2 recorded, Role 4 Level A = PASS, separate team approval. Even then at most wind speed, temperature, humidity; results labelled benchmark-assumption; Level B reported separately |
| GFS wind direction (anywhere, incl. downwind weighting) | **DISABLED, final** | Stored direction covers only [-90, 90]; half of directions are indistinguishable (audit sections 4, 8.2) | A rebuild from raw u/v (out of scope) |
| GFS precipitation | **DISABLED, final** | Changes meaning at 2019-11-07 (33.9x to 6.6x observed); neither regime is a daily total (audit 8.4) | A rebuild (out of scope) |
| GRIDMET channels | **EXCLUDED** | Level A UNKNOWN (register status `PENDING_ROLE3_EE_CHECK_5`: the day window is not checked); Level B FAIL. A Level-B failure does not establish a Level-A failure, so the status stays UNKNOWN, not FAIL | **Role 3 (me) runs EE check 5**, records it in audit 8.7; Role 4 rules Level A |
| Downwind baseline with GRIDMET direction | **DISABLED** | Same as GRIDMET; FROM/TOWARD convention assumed | As above, plus the convention confirmed |
| PDSI | **EXCLUDED** | Register: Level A UNKNOWN, `PENDING_TEMPORAL_SUPPORT` (5-day product; the composite window around day D is not established) | Temporal support established and Role 4 Level A = PASS |
| NDVI/EVI2, year-Y land cover | **EXCLUDED** | Register Level A FAIL | - |
| active_fire_time | **EXCLUDED** | Level A PASS; excluded because the HHMM encoding is unresolved (register `PASS_LEVEL_A_ENCODING_PENDING`) | Role 1/4 settle the encoding |
| Aspect as sin + cos | **NOT ADOPTED** (proposal pending) | Proposal sent; schema unchanged until approval is recorded | Approval recorded; register and config change in one commit |

## 3. GNN (P3-R3-05)

**DEFERRED.** The guide allows a GNN only after M1/M3/M4 training, evaluation and resource
budgets are stable; none of those has been trained on WSTS. No graph code, node definition or
adjacency has been written, and no prototype is presented. If time remains after Phase 4, the
graph must be declared in full (nodes, adjacency radius and weights, temporal aggregation,
sample boundaries inside the same P1 folds, compute budget, matched-parameter CNN comparator)
and approved before any training. No such declaration has been made.

## 4. Negative and null results, limitations

- No temporal result exists yet. That is a blocker, not a finding.
- The split defect found and fixed in Role 4's grouping (footprint rule v2) means any temporal
  numbers from before 2026-11-13 would have been on an invalid split; none were produced.
- Hardware: one 6 GB laptop GPU. M4 at T=5 on the largest frame is unmeasured on it;
  `scripts/benchmark_temporal_memory.py` measures it before scheduling. Fallback width (24) would
  be declared before training and applied to both models.
- Statics (slope, aspect, elevation) are repeated in every frame, which wastes first-layer
  capacity in M3 and encoder compute in M4. Kept for parity with the loader and M1.
