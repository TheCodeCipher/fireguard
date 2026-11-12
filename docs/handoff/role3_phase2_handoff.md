# FireGuard Role Hand-off Report

- Member name: Arkodyuti Bhattacharyya
- Assigned role: Role 3, Temporal / Weather / GNN Engineer
- Phase completed: Phase 2 (prepared ahead of schedule; cannot be marked complete before Gate 1 passes)
- Date: 2026-11-12
- Branch / PR links: (fill in; keep on a feature branch until Gate 1 is recorded)

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P2-R3-01 | Stacked-days U-Net (M3) interface | Done | `ml/src/fireguard_ml/models/temporal_unet.py` (`StackedDaysUNet`); `configs/temporal_unet.yaml` |
| P2-R3-02 | ConvLSTM module and M4 interface, tested | Done on synthetic input; real-sample run pending | `ml/src/fireguard_ml/models/convlstm.py`; `ConvLSTMUNet`; `ml/src/fireguard_ml/tests/test_convlstm_shapes.py`; `scripts/smoke_temporal_real_sample.py` |
| P2-R3-03 | Gate GFS/downwind branches; document no-GFS default | Done | `docs/gfs_branch_gating.md`; guards in `checkpointing.py` and tests |
| (support) | Self-describing checkpoints for later OOF export | Done | `ml/src/fireguard_ml/models/checkpointing.py` |
| (support) | Architecture note | Done | `docs/temporal_architecture_note.md` |
| (support) | ML package metadata and CI so the tests actually run | Done | `ml/pyproject.toml`; `.github/workflows/ml-ci.yml` |

Not done on purpose: no M3/M4 training (Phase 3, after Gate 2), no `gfs_benchmark_assumption.yaml` (audit has not passed Level A), no GNN.

## 2. Acceptance tests
| Test | Command / procedure | Result | Evidence |
|---|---|---|---|
| Temporal frame order matches the approved WSTS loader semantics | Read of pinned loader (f62dc49) lines 137-158 and 230 | PASS from code; Role 1's confirmation on a loaded multi-day sample pending | `docs/temporal_input_contract.md` section 1 |
| M3 receives the same strict channels and data splits as M1 | `c_strict` is a constructor argument; config points to Role 4's manifest | READY, cannot be shown until Role 4 freezes the strict list and manifests | `configs/temporal_unet.yaml` |
| ConvLSTM forward pass works on synthetic and representative real feature-map shapes | `cd ml && pytest`; `python scripts/smoke_temporal_real_sample.py --data-root <WSTS>` | Synthetic PASS (10 x 11 bottleneck of a 320 x 352 frame). Real sample: NOT YET RUN on the dataset | pytest output; `outputs/temporal_real_sample_smoke.json` once run |
| ConvLSTM output preserves spatial dimensions for the decoder | `cd ml && pytest` | PASS | `test_cell_preserves_spatial_shape`, `test_output_shape_matches_padded_input` |
| Strict configs contain no GFS channels | `cd ml && pytest -k "strict_config or forecast"` | PASS | 6 tests |
| GFS/FiLM remains separate and labelled benchmark-assumption | n/a | PASS: branch disabled, conditions written down | `docs/gfs_branch_gating.md` |
| Downwind baseline supported by verified direction or explicitly disabled | n/a | PASS: disabled, with reasons | `docs/gfs_branch_gating.md` |
| Optional GNN has not displaced required work | n/a | PASS: not started | |

Full suite: from the repo root, `pip install -e "ml[dev]"` then `cd ml && pytest -q` gives 28 passed, and `ruff check src` is clean (run 2026-11-12, Python 3.13, CPU PyTorch). `.github/workflows/ml-ci.yml` runs both on every push to a role branch and on every PR to main.

## 3. Dataset and source decisions
- Verified: frame order (oldest to newest) and target definition from the pinned loader code; UTC day boundary from the creation code.
- Assumed: `T = 5` input days (plan v4 working candidate).
- Unknown: final `c_strict` (35 is only the upper bound); behaviour when an event has a missing day.
- Excluded/leaky: GFS channels from M3 and M4.

## 4. Interface hand-offs
- Sent to: Role 2 (model contract, padding, skip-connection choice), Role 4 (strict list, checkpoint metadata for OOF export), Role 1 (frame order)
- Artifact/schema: `docs/temporal_input_contract.md` section 4; `docs/temporal_architecture_note.md`
- Confirmed by receiver: (pending)

## 5. Open issues and blockers
| Issue | Impact | Owner | Due date |
|---|---|---|---|
| Run the real-sample smoke script and commit the JSON | Last open acceptance item | Role 3 | Week 3 |
| Frozen strict channel list, sin/cos decision | Sets `c_strict` | Role 4 | Gate 1 |
| Shared padding utility; statics as separate tensor or not | Replaces the stand-in padding in the smoke script | Role 2 | Week 3 |
| M4 has 56% more parameters than M3 at equal width | H2 comparison is confounded unless a size-matched M3 (`base = 40`) is also run | Roles 2, 4, lead | before Phase 3 |
| Baseline loader center-crops evaluation frames and augments with flips/rotations | Conflicts with plan v4 | Roles 1, 2 | Week 3 |

## 6. Phase sign-off
- Role-specific acceptance: BLOCKED on external inputs (Role 4's strict list, Role 1's confirmation) and one run of mine (real-sample smoke test).
- Global gate impact: NONE. Gate 2 depends on M1 and the baselines, not on M3/M4.
- Recommended next action: after Gate 1, run the smoke script, get the contract signed by Roles 1, 2 and 4, then merge.
