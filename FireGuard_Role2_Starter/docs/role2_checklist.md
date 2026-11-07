# Role 2 checklist

## Phase 1 — do these before Phase 2

- [ ] Confirm team phase and obtain Gate 1 status from the team lead.
- [x] Capture Role 1's documented 40-channel processed schema, sample reference and dates as a replaceable adapter (actual tensor/mask still pending).
- [ ] Get Role 1's real sample tensor, source valid mask and loader reproduction steps.
- [ ] Get Role 4's strict no-GFS channel decision and frozen P1 split manifest/hash; config supports swapping the approved channel names and split path.
- [ ] Coordinate canonical padding utility ownership with Role 1.
- [ ] Create Python 3.11 environment; record PyTorch/CUDA/GPU and VRAM or fallback details.
- [x] Run synthetic padding, model-shape and masked-loss checks (9 passed).
- [x] Run a short synthetic 40-channel shape timing demo; record it as non-WSTS evidence in `docs/compute_benchmark.md`.
- [ ] Replace exploratory 40-channel config selection with approved strict channel names when Role 4 provides them.
- [ ] Run one representative short training benchmark on Role 1's real sample and agreed compute route; record loader time, steps, parameter count, training time, peak memory and hardware.
- [ ] Save evidence and update `compute_benchmark.md` and the model contract.
- [ ] Wait for team Gate 1 PASS/CONDITIONAL PASS before starting full Phase 2 schedule.

## Phase 2 — only after Gate 1

- [ ] Implement/finalize M1 with the strict approved no-GFS channels.
- [ ] Train on Role 4's frozen P1 training groups.
- [ ] Early stop/select only on full-frame validation AUPRC; do not tune on test metrics.
- [ ] Use full-frame validation/test at natural prevalence; unpad and mask invalid/padded cells.
- [ ] Save config, seed, environment/hardware, split hash, code identity, curves, checkpoint identity and per-event outputs.
- [ ] Hand predictions and metadata to Role 4 in the agreed format.
- [ ] Compare against strongest soft baseline before claiming usefulness.
