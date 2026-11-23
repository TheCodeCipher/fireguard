# Metrics contract

- Owner: Role 4 | Updated: 2026-11-13 | Status: proposed for Roles 1-3 review
- Implements Plan v4 sections 4.4, 4.6, 6.5, 6.6 and execution guide Role 4 tasks 9-13.
- Code: `ml/src/fireguard_ml/evaluation/{metrics,bootstrap,calibration}.py`; config
  `configs/evaluation.yaml`; tests `test_metrics_masks.py`, `test_validation_calibration_isolation.py`.

Every number reported for FireGuard, by any role, is computed with these definitions and code.
If a definition here and a result disagree, the result is wrong.

## 1. What is scored

| Item | Rule |
|---|---|
| Samples | Exactly the rows of `data/manifests/wsts_eval_samples.csv` that belong to the fold's split (`first_target_index = 5`, so one-day models, baselines and five-day models score identical event-days) |
| Target | Active-fire band of day D+1, NaN -> 0, then `> 0` (official loader) |
| Valid cells | Every original in-frame cell. Padding added for the U-Net (pad-to-32) is **always** excluded through `valid_mask`; so is any cell a data owner marks invalid |
| Frames | Full event frames at natural prevalence. Never crops, never positive-biased sampling (those are for training only) |
| Event | All scored samples of one WSTS event pooled together |

## 2. Metrics

| Metric | Definition | Notes |
|---|---|---|
| **AUPRC (primary)** | Average precision: sum over distinct score thresholds of (recall gain) x (precision at that threshold). Identical to scikit-learn `average_precision_score`, ties included | Needs continuous scores. NaN for an event with no positive cell |
| F1 (positive class) | 2TP / (2TP + FP + FN) at the fold's validation threshold | NaN when there are no positives and no predictions |
| IoU | TP / (TP + FP + FN) at the same threshold | Same NaN rule |
| Brier score | Mean of (p - y)^2 over valid cells | Probabilities in [0, 1] |
| Brier skill score | 1 - BS_model / BS_reference; reference = the **same fold's training-group distance-bin climatology** | Positive = better than climatology |
| ECE | Equal-mass bins (15), weighted mean of abs(mean predicted - observed frequency) | Per fold, never pooled across folds |
| Reliability table | Same 15 equal-mass bins, with counts, mean prediction and observed frequency | One per fold |

Implementation note: metrics are computed from counts grouped by distinct score, which makes
pooling exact and lets thresholds be chosen without revisiting pixels. For model outputs over a
whole fold use `StreamingScoreHistogram` (2^20 score bins; ranking metrics exact to ~1e-6 in
score, Brier exact).

## 3. Two summaries, always both

- **Pooled:** all valid cells of all events of the fold treated as one set (benchmark comparability).
- **Event-macro:** compute the metric per event, then the plain mean over events where it is
  defined; report how many events were used (stops a few huge fires from dominating).

Report every P1 test year separately. Pooling the four folds into one number is allowed only as an
extra line, never instead of the per-year table.

## 4. Comparing two models (the Gate 2 / Gate 3 test)

1. Both models are scored on identical events (same fold, same sample list).
2. Per event: difference in AUPRC (model minus reference).
3. **Paired bootstrap over fire-complex groups**, 2,000 replicates, seed 0: each replicate draws
   groups with replacement and keeps all events of each drawn group. Never resample pixels or days:
   neighbouring cells and consecutive days of one fire are strongly correlated, and a pixel
   bootstrap would produce falsely narrow intervals.
4. Report the mean difference, its 95 % interval, groups, events, and events excluded because their
   AUPRC is undefined.
5. Plan v4 6.6: the interval excluding zero is the primary evidence. A module is kept only if its
   interval is above zero on at least two folds; otherwise it is reported as a negative result.

## 5. Thresholds, calibration, early stopping: validation only

| Choice | Fitted on | Enforced by |
|---|---|---|
| F1 threshold | The fold's validation groups, full frames | `select_f1_threshold` refuses any split other than `validation` |
| Probability calibrator (Platt on logit) | The fold's validation groups | `fit_platt` refuses other splits; `FoldCalibrator.transform` refuses another fold |
| Early stopping / checkpoint choice | Validation full-frame AUPRC | Role 2/3 training code; recorded in the run config |
| Baseline parameters (decay scale and amplitude, dilation radius) | Validation groups | `select_decay_scale` refuses other splits |
| Climatology bin probabilities | Training groups only | `fit_climatology` refuses other splits |
| Normalisation statistics | Training groups only | Role 1/2 loader |

One threshold and one calibrator per (protocol, fold); `FoldArtifactStore` has no pooling API and
refuses to overwrite. The test year never influences any of these choices.

## 6. Observability (evaluation-only)

Next-day VNP09GA validity may be used **after prediction** to split events into better- and
poorly-observed strata and report metrics per stratum with coverage. It is information from the
target day, so it must never enter a model input, feature selection or readiness signal.
`fireguard_ml.data.observability.assemble_model_inputs` rejects it by type and by name, and
`test_observability_not_input.py` fails if that guard is removed. Primary benchmark scores are
never altered by observability. Within WSTS the proxy is the share of D+1 cells whose M11, I2 and I1 reflectance
values are present; VNP09GA QA flags (cloud, smoke) are not in the files, so cloudy-but-filled
cells cannot be identified. The values live in `data/evaluation_only/` and a static test fails if
any model-side module refers to them (`test_observability_is_evaluation_only.py`).

## 7. What a model owner hands to Role 4

For each fold, a CSV with the same columns as
`docs/evidence/role4/baseline_per_event_test_scores.csv`:
`fold_test_year, event_id, group_id, baseline (model name), auprc, brier, f1, iou, n_pos, n_valid`,
plus the run config recording: P1 manifest sha256, `configs/evaluation.yaml` sha256, samples sha256,
checkpoint id, channel list (checked by `assert_strict_channels`), and seed.
