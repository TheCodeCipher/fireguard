# Temporal input contract (Role 3)

> DRAFT, revised 2026-11-11. Status words: **VERIFIED** (seen in the pinned loader code, the creation code or a loaded sample), **ASSUMED**, **UNKNOWN**, **proposed** (a design choice awaiting sign-off). Sources: `docs/wsts_data_dictionary.md` (pinned baseline commit f62dc49, one sample event; Role 1's artifact, not yet committed to this repository); loader `src/dataloader/FireSpreadDataset.py` at f62dc49; creation code `FirePred.py` at f6c5167. Needs sign-off from Role 1 (sequence semantics), Role 2 (model contract, padding) and Role 4 (strict channel list, splits).

## 1. Sequence semantics

| Item | Value | Status |
|---|---|---|
| Unit of one frame | one daily raster, 23 original bands, 375 m grid | VERIFIED |
| Frames selected per sample | `n_leading_observations + 1` consecutive daily images | VERIFIED (loader line 137) |
| Inputs | all selected images except the last | VERIFIED (loader line 157) |
| Target | last band (active fire) of the final image, binarised with `y > 0` | VERIFIED (loader lines 158, 328) |
| Example with 1 leading observation | input 2018-01-06, target 2018-01-07 (next-day S-NPP VIIRS detection) | VERIFIED (one sample) |
| Frame order inside the tensor | oldest first, newest last. The loader sorts the `YYYY-MM-DD.tif` file names and slices them in order. | VERIFIED from code for the GeoTIFF path; still to be confirmed once on a loaded multi-day sample, and for the HDF5 path, by Role 1 |
| Target day relative to last input day | exactly +1 file. This is +1 calendar day only if the event has no missing days. | VERIFIED for n = 1; gap handling UNKNOWN, ask Role 1 |
| Value of `n_leading_observations` for FireGuard | 5 is the working candidate (plan v4 section 4.1). The creation code adds 4 buffer days before each fire so that 5 input days exist from the first fire day. | proposed, needs Role 1 / lead |
| Day boundary | UTC calendar day. The creation code passes date strings without a time zone and filters fire detections on `acq_date`. In the western US that is roughly 16:00-17:00 local time to the same time next day. | VERIFIED from code |
| Do all channels of a frame cover the same 24 hours? | No guarantee. GRIDMET's daily window and the GFS lead hours are open checks in `docs/gfs_channel_audit.md` section 8. | UNKNOWN |

## 2. Tensor shapes

Loader output for the inspected sample (VERIFIED): input `(T, C, H, W) = (1, 40, 288, 224)` float32, target `(288, 224)` int64 with values 0 and 1.

- `C = 40` is the processed channel count: 23 original bands with land cover expanded to 17 one-hot channels and a binary active-fire channel appended.
- `H, W` in that sample are the baseline loader's center crop of the raw raster to multiples of 32 (raw 304 x 247 became 288 x 224). **This conflicts with plan v4**, which requires padding to a multiple of 32 and evaluating the full frame. FireGuard frames must be padded (304 x 247 becomes 320 x 256), not cropped. Owner: Roles 1 and 2.
- Raster size varies by event, so shapes are per event and full-frame batches have batch size 1 unless frames are padded to a common size.
- Training mode in the baseline uses random square crops (data dictionary section 8.1).

## 3. Channel set

- Processed channels 33 to 37 (forecast precipitation, wind speed, wind direction, temperature, humidity) are GFS. They are **excluded** from strict M1-M4.
- Strict upper bound before Role 4's leakage audit: `C_strict = 40 - 5 = 35`. Role 4 may remove more (for example VNP13A1 channels 3 and 4 if Level A fails). **Use the frozen list from Role 4, never this number.** The models take `c_strict` as a constructor argument for that reason.
- Angle channels: the baseline applies `sin()` only, to original band indices 7 (GRIDMET wind direction), 13 (aspect) and 19 (GFS wind direction); these are processed indices 7, 13 and 35 (VERIFIED, loader `utils.py` line 162). Whether to add `cos()` for 7 and 13 is an open decision for Roles 2 and 4, and it changes `C_strict`.
- Static channels (slope, aspect, elevation, land cover) are repeated in every frame by the loader. With `T = 5` the stacked model sees them five times. Whether statics are passed once in a separate tensor (plan v4 section 4.1, `X_static`) is Role 2's interface decision; the contract below assumes they stay inside each frame until that is decided.

## 4. Interfaces

Both models are called the same way so that training and export code does not branch on model type.

| | M3 `StackedDaysUNet` | M4 `ConvLSTMUNet` |
|---|---|---|
| Call input | `(B, T, C_strict, H_pad, W_pad)` float32, frames oldest to newest | same |
| Internally | reshaped to `(B, T * C_strict, H_pad, W_pad)`; first conv has `T * C_strict` input channels | shared encoder on each frame, ConvLSTM over the T bottleneck maps `(B, T, F, H_pad/32, W_pad/32)`, decoder with skips from the newest frame |
| Output | logits `(B, 1, H_pad, W_pad)` | same |
| Probabilities | `model.predict(x, orig_hw=(H, W))` applies sigmoid and removes bottom/right padding | same |

- `H_pad` and `W_pad` must be multiples of 32 (depth 5). The models raise an error otherwise. The padding utility is owned by Role 2; padded pixels never count in loss or metrics (valid mask), and neither loss nor mask handling is inside the model.
- Both use the same strict channels and the same split manifests as M1.
- Checkpoints are written with `fireguard_ml.models.checkpointing.save_checkpoint`, which requires `channels`, `T`, `preprocessing_version`, `config_hash`, `fold` and `split_manifest_hash`, and refuses GFS or forecast channel names in a strict checkpoint.
- Do not start large M4 training in Phase 2.

Status: proposed. Implemented in `ml/src/fireguard_ml/models/temporal_unet.py` and `convlstm.py`, tested in `ml/src/fireguard_ml/tests/`.

## 5. Open items to close before Phase 2 training

1. Role 1: value of `n_leading_observations`, behaviour when an event has a missing day, and frame order on the HDF5 path.
2. Role 4: the frozen strict channel list and the sin/cos decision.
3. Role 2: the padding utility, whether static channels are a separate tensor, and the U-Net input/output contract.
4. Conflict to resolve: the baseline loader applies flips and 90-degree rotations during training (with angle adjustment), while plan v4 turns geometric augmentation off by default. `configs/temporal_unet.yaml` sets it off.
5. Conflict to resolve: the baseline loader center-crops evaluation frames; plan v4 requires pad and full-frame evaluation (section 2 above).
