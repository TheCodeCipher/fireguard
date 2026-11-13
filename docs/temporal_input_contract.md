# Temporal input contract (Role 3)

> Revised 2026-11-13 for Phase 3 (sections 3, 5, 6). Earlier revision 2026-11-11. Status words: **VERIFIED** (seen in the pinned loader code, the creation code or a loaded sample), **ASSUMED**, **UNKNOWN**, **proposed** (a design choice awaiting sign-off). Sources: `docs/data_dictionary.md` on Role 1's branch (pinned baseline commit f62dc49, one sample event); loader `src/dataloader/FireSpreadDataset.py` at f62dc49; creation code `FirePred.py` at f6c5167. Needs sign-off from Role 1 (sequence semantics), Role 2 (model contract, padding) and Role 4 (strict channel list, splits).

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

- Source of truth: Role 4's leakage register `docs/leakage_audit.csv`. Its strict list is 7 processed channels, in loader order: `viirs_m11, viirs_i2, viirs_i1, slope, aspect, elevation, binary_active_fire_mask`. `configs/temporal_unet.yaml` lists the same seven and a test fails if the two ever differ. `c_strict = 7`.
- Excluded: GFS (processed 33-37), GRIDMET (Level A UNKNOWN), PDSI, NDVI/EVI2, year-Y land cover, and `active_fire_time` (encoding problem). The trainer calls `assert_strict_channels` against the register before loading any data, and strict checkpoints refuse forecast channel names.
- Aspect stays a single sin-encoded channel. The sin/cos proposal (8 planes) is not approved; if it is, the register and the config change together in one commit.
- `binary_active_fire_mask` and `aspect` are not standardised; the other five are standardised with train-groups-only statistics (section 6).
- Static channels (slope, aspect, elevation) stay inside every frame, so M3 sees them T times. That matches the loader and costs 4 x 3 extra first-layer input planes at T=5.

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

## 5. Resolved since the Phase 2 draft

| Item | Resolution | Evidence |
|---|---|---|
| Strict channel list | 7 channels from Role 4's register | `docs/leakage_audit.csv`; `test_temporal_config_lists_exactly_the_registers_strict_channels` |
| Crop vs pad | Pad bottom/right to a multiple of 32; padding is invalid in loss and metrics; full frames are scored. The largest WSTS frame (356 x 311) pads to 384 x 320. | `training/temporal_data.py::pad_samples`; `test_padding_is_bottom_right_invalid_and_exactly_reversible` |
| Geometric augmentation | Off (plan v4) | config `geometric_augmentation: false` |
| Missing days | Every one of the 10,572 scored samples in `wsts_eval_samples.csv` (607 events) has target = last input + 1 calendar day, and each event's file dates are gap-free over the scored range. A sample whose dates are not consecutive is rejected by `validate_sample`, never imputed. | check run 2026-11-13 on the manifest; `test_contract_violations_are_rejected` |
| Scored samples | Identical rows for M1, M3, M4 and the T=3 ablation: first target at image index 5. With T=3 the model simply uses the 3 newest of the 5 available input days. | `scored_keys()`; config `ablation` |

Closed 2026-11-13: frame order and target semantics on loaded multi-day samples, by running the pinned loader against Role 1's reader on real events (T=1 and T=5, exact match; Role 1 data dictionary section 12). The HDF5 path is not used. Still open: Role 2 to confirm statics stay inside each frame.

## 6. Phase 3 sample-source interface (implemented by Role 1: `fireguard_ml.data.wsts_dataset.WSTSTemporalSource`)

Role 3 does not load WSTS files. Training and export read samples through this protocol (`ml/src/fireguard_ml/training/temporal_data.py`):

```python
class TemporalSampleSource(Protocol):
    channels: tuple[str, ...]      # exactly the strict list above, in that order
    sequence_length: int           # T
    is_mock: bool                  # False for real WSTS data
    def keys(self) -> list[tuple[str, date]]: ...        # (event_id, target_date)
    def load(self, key) -> TemporalSample: ...
```

`TemporalSample` holds `x [T, C, H, W] float32` (oldest frame first, the processed loader values before standardisation; NaN marks a missing measurement and is allowed in measured channels, never in `binary_active_fire_mask`), `y [H, W]` binary next-day target, `valid [H, W] bool`, the input dates, the target date, the channel names, the 6-term affine transform and CRS of the original raster. `validate_sample` checks every field. Frames are never cropped.

Normalisation (`training/normalization.py`): mean/std per channel over valid, non-missing cells of the fold's training samples only; missing values become 0 after standardisation, i.e. the training mean, which is what the pinned loader's final `nan_to_num` does; `fit_normalizer` raises if a key from a validation or test event is passed. The statistics are saved next to each checkpoint and re-used by the export.

`SyntheticTemporalSource` implements the same protocol with `is_mock = True`. It is used only by tests and dry runs and every artifact it touches is marked `is_mock`.
