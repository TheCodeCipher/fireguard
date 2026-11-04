# WSTS Data Dictionary

## 1. Dataset provenance

- Dataset: WildfireSpreadTS (WSTS)
- Zenodo: https://zenodo.org/records/8006177
- Baseline repository: https://github.com/SebastianGer/WildfireSpreadTS
- Dataset creation repository: https://github.com/SebastianGer/WildfireSpreadTSCreateDataset
- Pinned baseline commit: f62dc49f2c2a6d249c071533ae5340c0d01bfb95
- Dataset license: CC-BY-4.0
- Temporal coverage: 2018–2021
- Extracted files reported: 14,218
- Event directories observed: 607

## 2. Data organization

The extracted dataset is organized by year and fire event:

    extracted/
      2018/
        fire_<event_id>/
          YYYY-MM-DD.tif
      2019/
      2020/
      2021/

The inspected event was `fire_21458798` in 2018 and contained
14 daily TIFF files from 2018-01-01 through 2018-01-14.

## 3. Original GeoTIFF schema: 23 bands

The following names and order are obtained directly from
`FireSpreadDataset.map_channel_index_to_features(only_base=True)`
at the pinned baseline commit.

| Index | Original feature |
|---:|---|
| 0 | VIIRS band M11 |
| 1 | VIIRS band I2 |
| 2 | VIIRS band I1 |
| 3 | NDVI |
| 4 | EVI2 |
| 5 | Total precipitation |
| 6 | Wind speed |
| 7 | Wind direction |
| 8 | Minimum temperature |
| 9 | Maximum temperature |
| 10 | Energy release component |
| 11 | Specific humidity |
| 12 | Slope |
| 13 | Aspect |
| 14 | Elevation |
| 15 | Palmer drought severity index (PDSI) |
| 16 | Landcover class |
| 17 | Forecast: Total precipitation |
| 18 | Forecast: Wind speed |
| 19 | Forecast: Wind direction |
| 20 | Forecast: Temperature |
| 21 | Forecast: Specific humidity |
| 22 | Active fire |

## 4. Processed loader schema: 40 channels

The loader expands original land-cover channel 16 into 17 one-hot
channels and appends a binary active-fire channel.

| Processed index | Feature |
|---:|---|
| 0 | VIIRS band M11 |
| 1 | VIIRS band I2 |
| 2 | VIIRS band I1 |
| 3 | NDVI |
| 4 | EVI2 |
| 5 | Total precipitation |
| 6 | Wind speed |
| 7 | Wind direction |
| 8 | Minimum temperature |
| 9 | Maximum temperature |
| 10 | Energy release component |
| 11 | Specific humidity |
| 12 | Slope |
| 13 | Aspect |
| 14 | Elevation |
| 15 | Palmer drought severity index (PDSI) |
| 16 | Land cover: Evergreen Needleleaf Forests |
| 17 | Land cover: Evergreen Broadleaf Forests |
| 18 | Land cover: Deciduous Needleleaf Forests |
| 19 | Land cover: Deciduous Broadleaf Forests |
| 20 | Land cover: Mixed Forests |
| 21 | Land cover: Closed Shrublands |
| 22 | Land cover: Open Shrublands |
| 23 | Land cover: Woody Savannas |
| 24 | Land cover: Savannas |
| 25 | Land cover: Grasslands |
| 26 | Land cover: Permanent Wetlands |
| 27 | Land cover: Croplands |
| 28 | Land cover: Urban and Built-up Lands |
| 29 | Land cover: Cropland/Natural Vegetation Mosaics |
| 30 | Land cover: Permanent Snow and Ice |
| 31 | Land cover: Barren |
| 32 | Land cover: Water Bodies |
| 33 | Forecast: Total precipitation |
| 34 | Forecast: Wind speed |
| 35 | Forecast: Wind direction |
| 36 | Forecast: Temperature |
| 37 | Forecast: Specific humidity |
| 38 | Active fire |
| 39 | Active fire (binary) |

## 5. Sample raster metadata

Measured from event `fire_21458798`, 2018-01-06 and
2018-01-07.

| Property | Observed value |
|---|---|
| Raster height | 304 pixels |
| Raster width | 247 pixels |
| Bands | 23 |
| Data type | float32 |
| CRS | EPSG:32610 |
| Pixel resolution | 375 m × 375 m |
| GeoTIFF nodata metadata | None |
| Raster mask | All pixels marked valid |

Observed affine transform:

    | 375.00, 0.00, 697500.00 |
    | 0.00, -375.00, 4136625.00 |
    | 0.00, 0.00, 1.00 |

Observed bounds:

    left   = 697500.0
    bottom = 4022625.0
    right  = 790125.0
    top    = 4136625.0

These measurements describe the inspected event only. Do not
assume all events share the same CRS, transform or dimensions
without checking their metadata.

## 6. Temporal sequence and target

The official loader selects `n_leading_observations + 1`
consecutive images.

For TIFF input:

- Input consists of all selected images except the final image.
- Target is the last band of the final image.
- Target is converted to binary using `y > 0`.

For one leading observation:

    Input:  D(t)
    Target: ActiveFire(t+1)

Verified example:

    Input:  2018-01-06.tif
    Target: 2018-01-07.tif

The target raster contained 10 finite positive active-fire pixels.
The official processed loader also returned 10 positive pixels.

The benchmark target is next-day S-NPP VIIRS active-fire detection.
The newly detected cell subset is secondary; a newly detected cell
must not automatically be interpreted as a new ignition.

## 7. Loader tensor example

For the inspected sample:

| Property | Value |
|---|---|
| Input shape | (1, 40, 288, 224) |
| Input dtype | torch.float32 |
| Target shape | (288, 224) |
| Target dtype | torch.int64 |
| Target values | 0 and 1 |
| Input NaNs after preprocessing | 0 |
| Positive target pixels | 10 |

The original raster was 304 × 247. The evaluation loader applies
a center crop to dimensions divisible by 32, producing 288 × 224.

## 8. Missing-value and preprocessing behaviour

Observed in the baseline implementation:

- TIFF active-fire input NaNs are replaced with zero.
- Active-fire detection time is converted from HHMM to hours.
- Target NaNs are replaced with zero.
- Target is binarized using `y > 0`.
- Degree features are transformed with sine.
- Standardization uses predefined statistics for training-year
  combinations.
- Degree features and categorical land cover are not standardized.
- A binary active-fire input mask is appended.
- Land cover is expanded into 17 one-hot channels.
- Remaining input NaNs are replaced with zero.

For the inspected 2018-01-06 TIFF, all 75,088 pixels in original
band 22 were NaN. The GeoTIFF mask nevertheless marked every
pixel valid.

Therefore, raster mask validity and per-band observation validity
must not be treated as equivalent.

Do not apply a universal NaN-to-zero rule to all source channels.
Source-specific validity handling must be determined.

## 9. Angular-feature limitation

The pinned baseline applies sine to angular features but not cosine.
The baseline README records this as a known limitation and notes
that sine and cosine should both be used to avoid losing directional
information.

This limitation is documented, not silently modified here.

## 10. Metadata still requiring confirmation

The following details are not fully established for every channel:

- Exact source product/version for each feature.
- Physical units for each feature.
- Authoritative valid-value ranges.
- Source-specific nodata conventions.
- Native resolution before resampling.
- Whether all events share a common grid and CRS.
- Temporal meaning and forecast initialization for forecast channels.

Do not infer these values from a single sample's min/max.
Obtain them from the dataset paper, creation code, source-product
documentation and further raster inspection.

## 11. Reproducibility

- Baseline commit: f62dc49f2c2a6d249c071533ae5340c0d01bfb95
- Local compatibility patch: `docs/wsts_baseline_local_patch.diff`
- Reproduction notes: `docs/wsts_reproduction_notes.md`
- Sample event: `fire_21458798`
- Sample year: 2018

The local compatibility patch replaces the unavailable `T_co`
import with a `TypeVar` definition. It is a local modification and
is not part of the upstream pinned commit.
