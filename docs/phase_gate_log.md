# Phase gate log (Role 4 evidence; the team lead records the decisions)

| Date | Gate | Evidence | Status | Decided by |
|---|---|---|---|---|
| 2026-11-13 | G1-D: P1 manifests frozen | `p1_split_hashes.txt` `status=FROZEN`, footprint rule v2, manifest sha256 `34e2909009ad3bb7...`; 0 unreadable TIFFs; `validate_manifest` passes; first frozen hash withdrawn (over-grouping, `fire_complex_groups.md` 3a) | READY FOR LEAD SIGN-OFF | (lead) |
| 2026-11-13 | G1-B: leakage register | `docs/leakage_audit.csv`; strict list of 7 channels; GRIDMET pending EE check 5 (Role 3) | READY FOR LEAD SIGN-OFF | (lead) |
| 2026-11-13 | Baselines on frozen P1 | `docs/baseline_results.md`; decay beats climatology on all 4 folds (paired group bootstrap) | DONE | Role 4 |
| 2026-11-13 | Comparison protocol locked | `docs/model_comparison_protocol.md` (strongest baseline by validation AUPRC; Gate 2 BEATS/MATCHES/FAILS; Gate 3 >= 2 folds) | LOCKED, needs lead sign-off of the two operationalisations | (lead) |
| - | Gate 2: M1 vs strongest baseline | needs Role 2's per-event file + `model_version.json` | NOT RUN | - |
| - | Gate 3: M3, M4, terrain | needs M1-M4 runs | NOT RUN | - |
| - | Explanation layer | not implemented or validated | UNAVAILABLE (do not display) | Role 4 |
| - | Uncertainty layer | none exists; outputs are single probabilities | UNAVAILABLE | Role 4 |
