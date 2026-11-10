# Temporal input contract (Role 3)

> DRAFT. Facts marked **verified** come from `docs/wsts_data_dictionary.md` (pinned baseline commit f62dc49, one sample event). Everything else is marked **UNKNOWN** or **proposed**. Needs sign-off from Role 1 (sequence semantics), Role 2 (model contract, padding) and Role 4 (strict channel list, splits).

## 1. Sequence semantics

| Item | Value | Status |
|---|---|---|
| Unit of one frame | one daily raster, 23 original bands, 375 m grid | verified |
| Frames selected per sample | `n_leading_observations + 1` consecutive daily images | verified |
| Inputs | all selected images except the last | verified |
| Target | last band (active fire) of the final image, binarised with `y > 0` | verified |
| Example with 1 leading observation | input 2018-01-06, target 2018-01-07 (next-day S-NPP VIIRS detection) | verified (one sample) |
| Value of `n_leading_observations` used by FireGuard | not fixed in the documents | UNKNOWN, ask Role 1 / lead |
| Frame order inside the tensor for more than 1 day | presumed oldest first, newest last | UNKNOWN, verify by loading a multi-day sample and checking the file dates |
| Target day relative to last input day | exactly +1 day | verified (n = 1) |
| Day boundary | UTC dates from the creation code | UNKNOWN, verify |

## 2. Tensor shapes

Loader output for the inspected sample (verified): input `(T, C, H, W) = (1, 40, 288, 224)` float32, target `(288, 224)` int64 with values 0 and 1.

- `C = 40` is the processed channel count: 23 original bands with land cover expanded to 17 one-hot channels and a binary active-fire channel appended.
- `H, W` are the center crop of the raw raster to multiples of 32 (raw sample 304 x 247 became 288 x 224). Raster size varies by event, so shapes are per event.
- Training mode in the baseline uses random square crops instead (data dictionary section 8.1).

## 3. Channel set

- Processed channels 33 to 37 (forecast precipitation, wind speed, wind direction, temperature, humidity) are GFS. They are **excluded** from strict M1-M4.
- Strict upper bound before Role 4's leakage audit: `C_strict = 40 - 5 = 35`. Role 4 may remove more (for example VNP13A1 channels 3 and 4 if Level A fails). **Use the frozen list from Role 4, never this number.**
- Angle channels (wind direction, aspect): the baseline applies `sin()` only. Whether to add `cos()` is an open decision for Roles 2 and 4, and it changes `C_strict`.

## 4. Interfaces

**M3, stacked-days U-Net (proposed).**
- Input `(B, T * C_strict, H, W)`: the T daily frames concatenated along the channel axis, oldest to newest.
- `in_channels = T * C_strict`.
- Output logits `(B, 1, H, W)`, sigmoid applied outside the model.

**M4, U-Net encoder + ConvLSTM (proposed).**
- Input `(B, T, C_strict, H, W)`.
- A shared per-frame encoder maps each frame to `(B, T, F, h, w)`, where `h, w` are the encoder's reduced spatial size.
- ConvLSTM consumes the T feature maps in order and returns a spatial-temporal map `(B, F', h, w)`.
- The decoder upsamples to logits `(B, 1, H, W)`.
- Do not start large M4 training in Phase 2.

**Both.**
- H and W must be multiples of 32. The padding utility is owned by Role 2; padded pixels never count in loss or metrics (valid mask).
- Both use the same strict channels and the same split manifests as M1.

## 5. Open items to close before Phase 2

1. Role 1: value of `n_leading_observations`, frame order, and UTC versus local day.
2. Role 4: the frozen strict channel list and the sin/cos decision.
3. Role 2: the padding utility and the U-Net input/output contract.
4. Conflict to resolve: the baseline loader applies flips and 90-degree rotations during training (with angle adjustment), while the project rules say no flips or rotations because they break wind and aspect channels. Decide which applies to M3 and M4 training.
