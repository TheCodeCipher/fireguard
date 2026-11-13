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

## Size (measured, `C_strict = 35`, `T = 5`, `base = 32`, `depth = 5`)

| Model | Parameters |
|---|---:|
| M3 | 8,403,681 |
| M4 | 13,082,977 (of which ConvLSTM 4,719,616) |
| M3 with `base = 40` | 13,112,601 |

M4 has 56% more parameters than M3 at the same width. If M4 beats M3 in Phase 3, that alone cannot separate "recurrence helps" from "more parameters help". Proposed control for hypothesis H2: also run M3 at `base = 40`, which matches M4's size to within 0.3%. Decision for Roles 2 and 4 and the lead before Phase 3; it costs one extra M3 run per fold.

## Constraints
- H and W must be multiples of 32 (depth 5). Pad first; crop with `predict(..., orig_hw=...)`.
- GroupNorm is used (not BatchNorm) because full frames force small batches.
- Loss, masks and metrics are not in these modules.
- No GFS channels. See `docs/gfs_branch_gating.md`.

## Retention rule (plan v4)
M4 is kept only if paired event-level results versus M3 on identical folds justify the extra complexity (hypothesis H2). A null result is a valid finding.
