# FireGuard Phase 1 / Phase 2 integrated review status

Prepared 2026-11-13 from the supplied Role 1, 3, 4, and 5 archives. This is a local integrated review copy, not a Git branch or team approval record.

## Current decision

**Phase 1 is not signed off. Phase 2 is not complete. No official M1 training, evaluation campaign, or scale-up has been run.** The lead approved the Phase 2 sequence subject to channel, loader, evaluation, and compute agreement, but the supplied message does not record formal Global Gate 1 approval. Role 2's M1 runner must remain blocked until that approval is explicit.

## What the supplied handoffs unblock

- **Role 2 spatial package:** masked BCE + Dice numerical handling, separate input/target validity masks, bottom/right pad and exact unpad, configuration checks, reproducibility, and fail-closed training handoff guard are retained. A new temporal adapter applies the same pad/mask semantics to `[B,T,C,H,W]`, requiring target validity and valid input coverage on every day. The sample benchmark is still explicitly provisional because the supplied sample has no source masks and uses the cropped 40-channel tensor.
- **Role 3:** M3 stacked-days U-Net, M4 ConvLSTM U-Net, checkpoint metadata, configs and tests are integrated. A supplied real-data shape smoke record shows 5 dated input frames and a 304x247 frame padded to 320x256, with output unpadded to 304x247. That record uses 18 channels and does not establish the newly audited strict schema or source-aware mask behavior. No M3/M4 training was run; the Role 3 config keeps training disabled.
- **Role 4:** event grouping, split validation, channel leakage policy, metrics, calibration, bootstrap, baseline implementations, evaluation config, scripts, manifests and a real-data baseline report are integrated. The supplied baseline report records 10,572 scored samples, zero unreadable events, four LOYO folds, and full-frame natural-prevalence metrics. It is Role 4 evidence; it is not a Role 2 M1 result.
- **Role 5 Phase 2 backend:** the API/replay implementation and migration are integrated from the Role 4 combined snapshot. The later Role 5 ZIPs are byte-identical and identify themselves as a Phase 3/4 overlay; that overlay is not applied here.

## Acceptance checks run on this integrated copy

- ML Ruff: passed.
- ML tests: **143 passed**. scikit-learn was installed in the validation environment so the metric cross-check ran.
- Backend Ruff: passed.
- Backend tests: **34 passed** (one Starlette/httpx deprecation warning).
- SQLite migration: `alembic upgrade head` completed at `b6c2d4e91a20`; `alembic check` reports no new upgrade operations.
- P1 fold manifest hash: verified as `55a3ca18c0c981e8f1671b78e2d23b23a6e30fe84a929100dddcd5f9b39e90d5`. The group, event metadata, footprint CSV, and scored-sample hashes also match the supplied Role 4 hash file.

These checks do not validate PostgreSQL deployment, live FIRMS service behavior, or the actual team Git checkout.

## Remaining Phase 1 blockers and owners

| Requirement | Supplied evidence and current state | Owner / action |
|---|---|---|
| Role 1 loader and canonical full-frame preprocessing | Role 1 ZIP contains schema, reproduction notes, and a compatibility diff, but no callable loader or source-mask-complete export. The sample remains 288x224 after a reported crop from 304x247; crop offsets are not pinned-loader confirmed. | **Role 1:** provide pinned loader invocation/source and confirm crop semantics. Agree whether FireGuard will re-export full frames before model-side pad/unpad. |
| Source-aware nodata masks | No per-source validity masks or target nodata contract are supplied. The ZIP's sample notes state the GeoTIFF mask is not equivalent to per-band observation validity. | **Role 1:** provide masks, invalid-value semantics, and target nodata behavior; then Role 2 integrates and reruns. |
| Aspect and strict channel schema | Role 4's leakage register permits seven source channels (`VIIRS M11/I2/I1`, slope, aspect, elevation, day-D binary active-fire mask) and says aspect must use sin+cos. That implies eight model channels after replacing the scalar aspect channel. The 40-to-41 preprocessing has not been jointly acknowledged by Role 3; the supplied Role 3 handoff still calls the sin/cos choice open. | **Roles 1, 3, 4:** confirm angle convention, units, transformation order and channel names. **Role 2:** implement the agreed 41-channel preprocessing and strict subset/hash after sign-off. |
| Frozen split release integrity | The hash file, generated report and baseline JSON say `FROZEN`; the Role 4 Phase 1 handoff and README still say manifests are provisional / freeze run remains. Five of six hash entries match; the supplied `wsts_event_fire_cells.csv.gz` hash does not (recorded `c853ca0301dad695db70f64ed6765d82763d03d0d0848adf15f049c601c8eca8`, actual `1e5df12700d640d389c8c81591c276b6ae0ad76ab5a3ca525b800abd3d2c6451`). | **Role 4:** reconcile the status documents and compressed-artifact hash, then record receiver acknowledgments. The matching fold manifest hash is wired into configs but not treated as signed Gate 1 evidence yet. |
| Leakage register review | Role 4 explicitly says Role 1–3 review and receiver acknowledgment are pending; 7 strict channels are a consequentially narrow set with no weather, vegetation or land cover. | **Roles 1–3 and team lead:** acknowledge the register and accept or resolve the narrow strict schema at Gate 1. |
| Representative Role 2 benchmark and compute envelope | Prior Role 2 probe is one cropped sample, 40 exploratory channels, finite-only provisional mask, CPU only. It is not representative acceptance evidence. Supplied Role 3 smoke is interface evidence, not an M1 benchmark. | **Role 1/4:** provide agreed mask-complete full-frame sample and strict manifest. **Lead/Role 5:** agree compute constraints. **Role 2:** rerun after Global Gate 1, without scaling beyond the approved plan. |

## Remaining Phase 2 work

- **Role 2:** run the official M1 baseline and save checkpoints, configs, runtime, curves, and per-event scores only after explicit Global Gate 1 approval and the required handoffs. Compare on Role 4's protocol and provide per-event results for paired group bootstrap.
- **Role 3:** model interfaces are integrated; receiver confirmation of the strict channels, aspect encoding and multi-day loader semantics remains open. M3/M4 training is not part of this approval and remains disabled.
- **Role 4:** baseline code/results are supplied; resolve the release discrepancies above and compare M1 against those baselines once trained.
- **Role 5:** local API/replay tests pass, but live FIRMS, PostgreSQL, and integration with the actual current Git checkout remain unverified.
- **Role 6:** no current Role 6 archive was supplied. Frontend/API contract acknowledgment and the live monitor/replay UI work remain open.
- **Team lead:** record Global Gate 1 explicitly before official M1 training begins; later gates still control model expansion and Phase 3 training.

## Repository status

The integrated copy is `work/fireguard-integrated/` in the Codex workspace. It has no `.git` directory and was not pushed or submitted. Use the prepared integration ZIP for review/apply through the team's branch and PR process. Raw WSTS imagery was not included.


Role 2 has also begun the gate-independent Phase 3/4 preparation allowed by the new guide. Its synthetic robustness/export work and gate blockers are tracked separately in `docs/phase3/role2_preparation_status.md`; this does not change the Phase 1/2 approval status above.
