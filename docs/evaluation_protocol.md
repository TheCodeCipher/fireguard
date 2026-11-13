# Evaluation protocol (P1, Phase 2 onward)

- Owner: Role 4 | Updated: 2026-11-13 (Phase 3 rules locked)
- Definitions: `docs/metrics_contract.md`. Splits: `docs/validation_protocols.md` section 15.

## Step by step, for every model and every baseline

1. **Load and verify the split.** Read `data/manifests/p1_loyo_fold_manifest.csv`, call
   `validate_manifest`, and check its sha256 against `data/manifests/p1_split_hashes.txt`
   (`run_baselines.py` refuses to run on a mismatch; model code should do the same). The header
   line must say `status=FROZEN` before any result is quoted.
2. **Check channels.** `assert_strict_channels(channel_names, load_register(...))` for strict
   models. GFS is only allowed in a run explicitly labelled benchmark-assumption.
3. **Train** on the fold's `train` groups only. Normalisation statistics from train only.
4. **Select on validation.** Early stopping on validation full-frame AUPRC; then the F1 threshold
   and (if used) the Platt calibrator on validation, one per fold.
5. **Score the test groups once**, on the rows of `wsts_eval_samples.csv` for those events, with
   padding masked out.
6. **Write per-event scores** in the contract format and the run config with all hashes.
7. **Compare** against the strongest soft baseline of the same fold (chosen on validation AUPRC)
   with the paired event-group bootstrap: `scripts/run_model_comparison.py`. The full locked rules
   are in `docs/model_comparison_protocol.md`.

## Gate 2 rule (Plan v4)

M1 must match or beat the strongest soft baseline (distance decay or distance-bin climatology) on
AUPRC under P1, shown per fold with the paired bootstrap interval. If it does not, stop adding
modules and diagnose target, channel timing, observability, class imbalance, padding and
preprocessing first. A documented negative result is acceptable.

Mechanical form (locked 2026-11-13): FAILS if any fold is significantly worse; BEATS if none is
worse and at least two are significantly better; MATCHES otherwise
(`docs/model_comparison_protocol.md` section 5).

## Baselines (implemented, `scripts/run_baselines.py`)

| Baseline | Output | Fitted / chosen on | Reported metrics |
|---|---|---|---|
| Persistence | Today's fire mask | nothing | F1, IoU |
| Dilated persistence | Today's fire grown by r km | r chosen by validation F1 from {0.375, 0.75, 1.5, 3} km | F1, IoU |
| Distance decay | a * exp(-d / s) | s by validation Brier from {0.25 ... 8} km; a in closed form | AUPRC, Brier, BSS, F1, IoU |
| Distance-bin climatology | P(fire tomorrow given distance bin) | training groups only; 11 bins incl. "no fire in frame" | AUPRC, Brier (BSS reference), F1, IoU |
| Downwind-weighted decay | n/a | **disabled**: no wind direction passed the encoding and Level-A audits | n/a |
| Logistic regression / boosting | n/a | optional, not implemented | n/a |

Binary baselines are not given an AUPRC: a 0/1 map has one operating point, so its AUPRC is
degenerate. Distance-decay AUPRC is the same for every s and a (monotone transform), so those
parameters are chosen by Brier, the metric they change.

## Reporting checklist

- [ ] Per-fold tables (pooled and event-macro), never only an average across folds.
- [ ] Event and group counts per fold; events excluded from event-macro AUPRC.
- [ ] 2019 caveat: 74 raw events (vs 156-201), fewest test groups, widest intervals.
- [ ] Manifest status `FROZEN` and all hashes in the results header.
- [ ] Observability-stratified results, if shown, labelled as a diagnostic.
- [ ] Before reporting M1: Role 2's per-event file uses the exact manifest, sample list and config
      hash recorded in `docs/baseline_results.md`.

## Running the baselines

```powershell
python scripts/run_baselines.py --data-root <folder that holds 2018..2021>
```

Stage A reads two active-fire bands per scored sample once (about 10-15 minutes) and caches the
counts in `outputs/role4/` (git-ignored); later runs take seconds. Outputs:
`docs/baseline_results.md`, `docs/evidence/role4/baseline_results.json`,
`docs/evidence/role4/baseline_per_event_test_scores.csv`.
