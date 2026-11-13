# Model comparison protocol (locked before any model result) — Role 4, P3-R4-02

- Locked: 2026-11-13, before any M1-M4 test result exists. Changing a rule after a model result
  is seen requires a new dated entry in `docs/phase_gate_log.md` stating why.
- Operational definitions marked **(Role 4 operationalisation)** turn plan v4 wording into a
  mechanical rule; they need the team lead's sign-off in the gate log.
- Code: `fireguard_ml.evaluation.comparison`; CLI: `scripts/run_model_comparison.py`.

## 1. Same inputs for every model

| Item | Fixed to | Checked by |
|---|---|---|
| Split | `p1_loyo_fold_manifest.csv`, sha256 `34e2909009ad3bb7...`, `status=FROZEN`, footprint rule v2 | every runner re-hashes it; the comparison refuses a model whose `model_version.json` records another hash |
| Scored samples | `wsts_eval_samples.csv` (10,572 rows, first target index 5), the same for T = 1, 3, 5 | Role 3 `scored_keys`; Role 1 reader keys; comparison refuses a model whose per-fold event set differs from the baselines' |
| Target | D+1 active-fire band, NaN -> 0, > 0 | Role 1 reader (reproduced against the pinned loader) |
| Strict data contract | `wsts_strict_input_v1`, 7 channels | register + `assert_strict_channels` before training; recorded in every OOF record |
| Valid cells | every native cell; padding invalid | `pad_samples`; metrics take the valid mask |
| Normalisation | per fold, training groups only | `fit_normalizer` raises on non-training events (`test_normaliser_refuses_non_training_events`) |
| Thresholds, calibrators, early stopping | per fold, validation groups only, natural prevalence | `select_f1_threshold` / `fit_platt` refuse other splits; one selection record per fold |

## 2. Metrics

- Primary: AUPRC on the test split of each fold, **pooled** over all valid cells and **event-macro**
  (mean of per-event AUPRC; events with no next-day fire are excluded from the mean and counted).
- Secondary: F1 and IoU at the fold's validation threshold; Brier and BSS (reference: distance-bin
  climatology of the same fold); ECE over 15 equal-mass bins; reliability tables.
- Always per fold. An average across folds may be shown only next to the per-fold table.
- **2019 caveat** in every summary: 74 raw events (88 test events, 61 test groups), widest
  intervals; it may not decide a gate on its own.

## 3. Paired comparison

Per event: candidate AUPRC minus reference AUPRC. Paired bootstrap over fire-complex groups
(2,000 replicates, seed 0, 95 % percentile interval; all events of a drawn group are kept; pixels
and days are never resampled). Verdict per fold: BETTER if the interval is above 0, WORSE if
below 0, otherwise NO_DIFFERENCE_SHOWN. Per-event differences are published
(`reports/phase3/per_event_differences.csv`).

## 4. Strongest baseline **(Role 4 operationalisation)**

Per fold, the soft baseline (distance decay or distance-bin climatology) with the higher
**validation** pooled AUPRC. The test split is never used to choose it. With the frozen split this
is distance decay on all four folds (validation AUPRC 0.2625 / 0.3139 / 0.3115 / 0.2711 vs
0.2527 / 0.3046 / 0.3015 / 0.2612).

## 5. Gate 2 (M1 vs strongest baseline) **(Role 4 operationalisation)**

Plan v4: M1 must "match or beat" the strongest baseline. Mechanically:
- **FAILS** if any fold is WORSE;
- **BEATS** if no fold is WORSE and at least 2 folds are BETTER;
- **MATCHES** otherwise. MATCHES and BEATS pass the gate; the report states which.

## 6. Gate 3 (added modules)

Retain an added module only if the paired AUPRC interval is above zero on **at least two**
evaluated folds; otherwise REMOVE and report the null/negative result. Fixed pairs:

| Candidate | Reference | Module tested |
|---|---|---|
| M3 stacked U-Net (T=5) | M1 | temporal context |
| M4 ConvLSTM (T=5) | M3 | recurrence |
| M4 | M3 at base 40 (13.06 M parameters) | recurrence at matched capacity (control; run only if M4 beats M3) |
| M1 (with terrain) | M2 (without terrain) | terrain channels |

Sequence-length ablation (not a module): M3 T=3 vs M3 T=5. T=3 is preferred if it is not WORSE on
any fold (same skill, cheaper); otherwise T=5 stays. This replaces the provisional "3 of 4 folds"
rule in Role 3's ablation report.

Strict no-GFS models are evaluated first. HLS, GFS/FiLM and GNN runs, if their gates ever open,
are reported in separately labelled tables and never enter these pairs.

## 7. What NOT RUN means

A model without a per-event file is NOT RUN. Nothing is imputed, and NOT RUN is never reported as a
negative result. A REMOVE decision stays in the report.
