# Temporal architecture note (Role 3)

Both models see the same strict (no-GFS) channels, the same T days, the same padded full frames and the same splits. Only how time is handled differs.

## M3: stacked-days U-Net (non-recurrent comparator)
The T daily frames are concatenated along the channel axis, so the first convolution sees `T * C` channels and a plain U-Net does everything. Time is only available as "which channel group came from which day", so the model must learn day ordering from the first layer's weights. Cheap, easy to train, and the right control for the question "does recurrence help?"

## M4: shared encoder + ConvLSTM
One encoder with shared weights processes each day separately. The T bottleneck feature maps go through a ConvLSTM, whose gates are convolutions. The hidden state therefore stays a spatial map (`[B, F, h, w]`) rather than a flat vector, so "where" is preserved while "when" is learned by the recurrence. A plain LSTM would need each frame flattened to a vector, which discards the grid the decoder needs. The decoder takes the final hidden state plus skip connections from the last day (day D).

Why skips from the last day: the target is D+1, and day D carries the finest-resolution fire state. Alternatives (fusing skips across days) are possible but add parameters only to M4. This is a design choice to confirm with Role 2.

## How the two differ

| | M3 | M4 |
|---|---|---|
| Where days are combined | first convolution | bottleneck, at 1/32 resolution |
| Weights per day | separate (each day has its own input filters) | shared encoder |
| Knows frame order | only through which filters see which day | by construction; reversing the frames changes the output (tested) |
| Fine-resolution detail from days before D | reaches the decoder through the skips | does not; only day D's skips are used |
| Encoder cost per sample | one pass | T passes |

## Size (measured with the frozen strict list: `C_strict = 7`, `base = 32`, `depth = 5`)

| Model | T = 5 | T = 3 |
|---|---:|---:|
| M3 | 8,363,361 | 8,359,329 |
| M3 with `base = 40` (size-matched control) | 13,062,201 | 13,057,161 |
| M4 | 13,074,913 | 13,074,913 |

M4's size does not depend on T (the encoder is shared across frames); M3's first layer grows with T x C. M4 has 56% more parameters than M3 at the same width, so "M4 beats M3" cannot by itself separate recurrence from capacity. The size-matched M3 (`base = 40`, within 0.1% of M4) is predeclared in `configs/temporal_unet.yaml` (`ablation.m3_size_matched_base`) and is run only if M4 beats M3.

The Phase 2 numbers (C = 35) are superseded; Role 4's audit removed 28 channels.

## Frame and channel order (P3-R3-01)

- Frames: oldest first, newest last; the target is the day after the newest frame. Checked on every sample by `validate_sample` (consecutive dates, target = last + 1, target not among the inputs).
- Channels per frame, in this order: `viirs_m11, viirs_i2, viirs_i1, slope, aspect, elevation, binary_active_fire_mask`. Recorded in the config, in every checkpoint (`channels`) and in every exported forecast record.
- M3 flattens `[T, C]` to `T*C` input planes frame-major: plane `t*C + c` is channel c of frame t.

## Training procedure (implemented, not yet run on WSTS)

`ml/src/fireguard_ml/training/temporal_trainer.py::train_fold`, one call per (model, T, seed, fold):

1. Verify `p1_split_hashes.txt` says FROZEN and that the manifest's SHA-256 matches; validate the manifest; check the channels against the register.
2. Take train / validation events from the manifest and samples from `wsts_eval_samples.csv` (the same rows as M1).
3. Fit the normaliser on training samples only.
4. Train with AdamW, masked BCE + Dice identical to Role 2's `masked_bce_dice_loss` (Role 2's function is imported when installed; otherwise a verbatim copy, tested equal), batch 1 with gradient accumulation 4, AMP on CUDA, seeded shuffling.
5. After each epoch score every validation sample full-frame (pooled AUPRC with Role 4's streaming histogram); early-stop on it.
6. Restore the best epoch; fit the F1 threshold and the Platt calibrator on validation through Role 4's isolation API.
7. Write the checkpoint, normaliser, selection, per-epoch log and run summary (hardware, time, peak memory, hashes).

The test split is never loaded during training; a test wraps the source and asserts that.

## Constraints
- H and W must be multiples of 32 (depth 5). Pad first; crop with `predict(..., orig_hw=...)`.
- GroupNorm is used (not BatchNorm) because full frames force small batches.
- Loss, masks and metrics are not in these modules.
- No GFS channels. See `docs/gfs_branch_gating.md`.

## Retention rule (plan v4)
M4 is kept only if paired event-level results versus M3 on identical folds justify the extra complexity (hypothesis H2). A null result is a valid finding.
