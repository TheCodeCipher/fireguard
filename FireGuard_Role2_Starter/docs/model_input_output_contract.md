# Model input/output contract — Role 2

## Current documented processed WSTS schema

The Role 1 handoff documents an official processed example with input shape `[1,40,288,224]`, one input observation, and a target of shape `[288,224]`. It describes 40 processed channels: 0–15 VIIRS/vegetation/weather/topography/drought; 16–32 one-hot MODIS land cover; 33–37 GFS forecast variables; 38–39 active-fire time and binary mask. The original inspected TIFF is `304×247`.

Those are **reference facts**, not the strict model input decision. Every channel's timing audit is not complete in the supplied handoff; `docs/leakage_audit.csv` reports unknown/pending entries. Role 4 owns the final strict channel set.

## Stable model interface

| Item | Contract | Status |
|---|---|---|
| Input | `[B,C,H,W]` floating tensor | Implemented; named-channel adapter |
| Current schema adapter | 40 named processed baseline channels | Documented by Role 1; exploratory only |
| Strict input channels | Explicit named subset, in approved order | Pending Role 4; config replaceable |
| Target | `[B,1,H,W]`, binary next-day target | Adapter converts finite values with `>0`; source target semantics documented by Role 1 |
| Source valid mask | `[B,1,H,W]` boolean | Optional adapter input; Role 1 source-aware mask pending |
| Fallback validity | Every selected input channel finite, and target finite | Implemented conservative fallback; not a substitute for source masks |
| Padding | Bottom/right to multiple of 32 | Implemented; preserves original H/W |
| Padded validity | False | Implemented |
| Model output | `[B,1,H_pad,W_pad]` raw logits | Implemented |
| Probability | `sigmoid(logits)` outside model | Contract |
| Evaluation | Unpad; full original frame; valid pixels only | Helper implemented; metric code belongs to Role 4 |
| Loss | BCE-with-logits + soft Dice over valid pixels | Implemented |

## Handoff replacement points

- Channel names/order/subset: `src/fireguard_ml/data/channel_schema.py` and `configs/unet.yaml`.
- Real WSTS loader sample and source mask: call `adapt_sample(...)` from `src/fireguard_ml/data/sample_adapter.py`.
- Split protocol: training orchestration should consume Role 4's manifest/hash; model code intentionally contains no split assignments.
- Data provenance: populate event/date fields in `ModelSample` and log schema/split hash with each run.

## Known preprocessing discrepancy to avoid

The supplied Role 1 reproduction notes report the baseline evaluation loader center-crops 304×247 to 288×224. The locked FireGuard plan instead requires pad-to-32, valid masking, full-frame evaluation and unpadding. This scaffold follows the locked plan (304×247 → 320×256), not the baseline center-crop path. Role 1/4 should reconcile this when the canonical loader is integrated.
