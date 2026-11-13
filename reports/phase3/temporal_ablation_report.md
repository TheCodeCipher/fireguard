# Temporal models and sequence ablation (Role 3, P3-R3-01/02/03)

**Status: NOT RUN.** No M3, M4 or ablation run has been trained on WSTS data. Nothing in this
report is a P1 result.

Why not run:

1. Gate 2 (M1 beats the baselines on P1) is not recorded. The Phase 3 guide allows only
   scaffolding and synthetic tests before it.
2. (Resolved 2026-11-13) Role 1's WSTS reader now exists (`fireguard_ml.data.wsts_dataset`); a
   real-data plumbing run (4 events, untrained model) passed end to end. Only Gate 2 blocks.

The pipeline that will produce these results is implemented and tested on synthetic data (see
"Readiness" below). The tables are the predeclared layout; they are filled in only by running the
commands listed.

## 1. Predeclared runs (fixed before any result exists)

| Run | Model | T | Width | Seeds | Folds | Purpose |
|---|---|---:|---:|---|---|---|
| A | M3 stacked-days U-Net | 5 | 32 | 0 | 2018-2021 | primary temporal model |
| B | M4 shared encoder + ConvLSTM | 5 | 32 | 0 | 2018-2021 | H2: does recurrence help |
| C | M3 | 3 | 32 | 0 | 2018-2021 | sequence-length ablation |
| D | M3 | 5 | 40 | 0 | 2018-2021 | size-matched control; only if B beats A |

All runs use the same strict channels, folds, scored samples (`wsts_eval_samples.csv`), target,
padding and valid masks as M1. Test usage does not change between runs: run C scores the identical
rows as A (it reads the 3 newest of the same 5 input days).

Decision rules: **locked by Role 4 in `docs/model_comparison_protocol.md` (2026-11-13)**, which
supersedes the provisional "three of four folds" rule first proposed here:

- Comparisons are paired group-bootstrap differences in per-event AUPRC on each fold's test split.
- Gate 3: M3 is kept over M1, and M4 over M3, only if the 95 % interval is above zero on at least
  two folds; otherwise the module is removed and the null result is recorded.
- Run D (M3 at base 40) is run only if B beats A; if B does not also beat D, the gain is
  attributed to capacity, not recurrence.
- Run C (T=3) is preferred over A if it is not significantly worse on any fold.
- A single event, a training curve or validation AUPRC is never evidence for a claim.

## 2. Results

| Fold | M1 | A: M3 T5 | B: M4 T5 | C: M3 T3 | D: M3 w40 | B - A [95% CI] | C - A [95% CI] |
|---|---|---|---|---|---|---|---|
| 2018 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | - | - |
| 2019 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | - | - |
| 2020 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | - | - |
| 2021 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | - | - |

Reference only (Role 4, frozen split v2): distance-decay baseline test AUPRC 0.3484 / 0.2256 /
0.3111 / 0.3959 for folds 2018-2021.

## 3. Commands that will fill section 2

```
python scripts/benchmark_temporal_memory.py --T 5 3
python scripts/train_temporal.py --model m3_stacked_unet  --T 5 --source wsts --out outputs/temporal
python scripts/train_temporal.py --model m4_convlstm_unet --T 5 --source wsts --out outputs/temporal
python scripts/train_temporal.py --model m3_stacked_unet  --T 3 --source wsts --out outputs/temporal
```

Each run writes per-fold checkpoint, normaliser, validation selection, per-epoch log and a summary
with config/split hashes, hardware, wall time and peak GPU memory. Per-event test scores for the
bootstrap come from the OOF export (`post_reveal/per_event_test_scores.csv`, same columns as Role
4's baseline file).

## 4. Readiness (what is tested now, on MOCK data only)

| Check | Test |
|---|---|
| Frames consecutive, oldest first, target = last + 1, target never an input | `test_contract_violations_are_rejected` |
| No-future data: model input contains only days before the target | same (dates checked per sample) |
| Padding bottom/right, invalid, reversible; loss ignores padding | `test_padding_is_bottom_right_invalid_and_exactly_reversible`, `test_loss_ignores_padding_and_matches_role2_when_installed` |
| Normalisation uses training events only | `test_normaliser_refuses_non_training_events` |
| M3 and M4 train end to end; the test split is never loaded | `test_train_fold_end_to_end_never_touches_test_events[m3/m4]` |
| Refuses unfrozen or tampered split manifests and excluded channels | `test_training_refuses_*` |
| ConvLSTM forward/backward and output shapes | `test_convlstm_shapes.py` (Phase 2) |
| Strict config equals the register | `test_temporal_config_lists_exactly_the_registers_strict_channels` |

A full dry run of the CLI against the real frozen manifest with synthetic tensors (M4, T=3, folds
2019 and 2021, 1 epoch of 16 samples, CPU) completed in 1 min 55 s and the export wrote 3,686
forecasts for 210 held-out events. Those numbers measure plumbing only.

## 5. Resource limits

Target hardware is one RTX 3060 laptop GPU (6 GB). Batch 1 with gradient accumulation and AMP.
Peak memory for M4 at T=5 on a 384 x 320 frame has NOT been measured on that GPU yet; run
`benchmark_temporal_memory.py` first. If M4 T=5 does not fit, the fallback (to be declared before
training, not after) is `base = 24` for both M3 and M4, keeping them comparable.
