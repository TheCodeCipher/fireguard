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

## 4.1 Temporal and spatial feature groups

The baseline feature mapping distinguishes the following broad groups:

| Group | Processed indices | Features |
|---|---:|---|
| VIIRS reflectance | 0–2 | M11, I2, I1 |
| Vegetation indices | 3–4 | NDVI, EVI2 |
| Weather observations | 5–11 | Precipitation, wind, temperature, ERC, humidity |
| Topography | 12–14 | Slope, aspect, elevation |
| Drought | 15 | PDSI |
| Land cover | 16–32 | 17 one-hot classes |
| Forecast weather | 33–37 | Forecast precipitation, wind, temperature, humidity |
| Active fire | 38–39 | Detection time and binary detection mask |

This grouping is based on the baseline feature mapping. It does not
establish the native temporal resolution or update frequency of every
source product.

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
## 8.1 Training and evaluation preprocessing

The baseline uses different spatial preprocessing depending on the
`is_train` setting.

- Training mode performs random square cropping.
- Ten candidate crops are evaluated.
- Crop selection strongly favors target crops containing active-fire
  pixels, with a smaller preference for crops containing input fire.
- Horizontal and vertical flips and rotations by multiples of 90
  degrees are applied.
- Angular features are adjusted during geometric transformations.
- Evaluation mode applies a center crop whose height and width are
  multiples of 32.

Consequently, training crops are not a representative sample of
natural fire prevalence. Evaluation crops preserve the full-frame
class distribution only within the center-cropped region.

## 8.2 Standardization and missing values

Input features are standardized using predefined means and standard
deviations selected through `stats_years`.

The baseline replaces remaining input NaNs with zero after
standardization. In this context, zero corresponds to the standardized
feature mean, rather than a physical measurement of zero.

The active-fire input channel is handled separately: NaNs are first
replaced with zero, and detection times encoded as HHMM are converted
to hours. A binary active-fire mask is computed before standardization
and appended as an additional input channel.

The HDF5 conversion generator also applies the active-fire NaN and
HHMM preprocessing before writing the converted data.

These are baseline implementation choices. They should not be
interpreted as universal source-product nodata conventions.

## 8.3 Event-wide active-fire audit: `fire_21458798` (2018)

An event-wide audit was completed for all 14 daily GeoTIFFs from 2018-01-01 through 2018-01-14. The raw active-fire band (1-based band 23; zero-based index 22) was read directly from each TIFF.

| Audit measure | Result |
|---|---:|
| Daily TIFFs audited | 14 |
| Pixels per TIFF | 75,088 |
| Total pixel observations | 1,051,232 |
| NaN pixels | 1,051,154 |
| Infinite pixels | 0 |
| Finite pixels | 78 |
| Positive detection pixels | 78 |
| Finite zero pixels | 0 |
| Dates with positive detections | 5 |
| Distinct positive HHMM values across the event | 8 |

Positive detections occurred on:

| Date | Positive pixels | Distinct raw HHMM values |
|---|---:|---|
| 2018-01-01 | 4 | 854 |
| 2018-01-07 | 10 | 2006 |
| 2018-01-10 | 4 | 2048 |
| 2018-01-11 | 34 | 912, 2030 |
| 2018-01-12 | 26 | 848, 1030, 2012 |

The raw positive values are acquisition times encoded as HHMM. For example, 854 represents 08:54. The baseline loader replaces active-fire NaNs with zero, converts HHMM values to integer hours using floor division by 100, and binarizes the target with `y > 0`. It also derives and appends a binary active-fire input mask from the processed input active-fire channel.

The 78 positive pixels are daily rasterized detection-pixel observations across the event. They must not be interpreted as 78 unique fires or ignitions. The eight distinct HHMM values are distinct time values, not eight separate fire events.

These statistics are specific to this inspected event. They are not dataset-wide rates or universal properties of every WSTS event. The audit CSV is `data/manifests/wsts_2018_fire_21458798_active_fire_audit.csv`.

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


### Active-fire detection channel

**Raw channel:** 22 (23rd TIFF band)  
**Processed channel:** 38  
**Additional binary channel:** 39  
**Source:** VIIRS active-fire FeatureCollection  
**Raw representation:** Acquisition time encoded as HHMM integer; NaN values are present where no finite detection value is rasterized.

The dataset-generation pipeline converts the VIIRS acquisition-time attribute to an integer and rasterizes it into the active-fire band.

The baseline loader converts acquisition time from HHMM to integer hours using floor division by 100. It also creates a binary active-fire mask using values greater than zero.

The resulting processed representation contains:
- Channel 38: active-fire acquisition-hour feature.
- Channel 39: binary active-fire detection mask.

The prediction target is generated from the final observation's active-fire band using a positive-value test.

### Active-fire detection channel

**Raw channel:** 22 (23rd TIFF band)  
**Processed channel:** 38  
**Additional binary channel:** 39  
**Source:** VIIRS active-fire FeatureCollection  
**Raw representation:** Acquisition time encoded as HHMM integer; NaN values are present where no finite detection value is rasterized.

The dataset-generation pipeline converts the VIIRS acquisition-time attribute to an integer and rasterizes it into the active-fire band.

The baseline loader converts acquisition time from HHMM to integer hours using floor division by 100. It also creates a binary active-fire mask using values greater than zero.

The resulting processed representation contains:
- Channel 38: active-fire acquisition-hour feature.
- Channel 39: binary active-fire detection mask.

The prediction target is generated from the final observation's active-fire band using a positive-value test.

**Sample inspection:**
For event `2018/fire_21458798`, first TIFF:
- Total pixels: 75,088.
- NaN pixels: 75,084 (99.9947%).
- Finite pixels: 4.
- Positive pixels: 4.
- Unique finite value: 854.
- Infinite pixels: 0.

The value 854 is consistent with acquisition time 08:54 and is converted to hour 8 by the baseline loader.

These statistics describe one TIFF only and must not be generalized to the complete WSTS dataset. NaN should not automatically be interpreted as sensor failure; its meaning depends on the active-fire rasterization and loader conventions.

### Active-fire channel: missing-value and target semantics

The active-fire channel (raw band index 22; 1-based band 23) stores VIIRS active-fire acquisition times in HHMM format. In the inspected event `fire_21458798` (2018-01-01 to 2018-01-14), NaN values represent pixels with no active-fire detection, rather than missing observations.

An event-wide audit of all 14 daily GeoTIFFs found:
- 75,088 pixels per TIFF.
- 1,051,154 NaN pixels across all 14 TIFFs.
- 78 finite positive detection pixels across the event's daily rasters.
- 5 dates with at least one positive detection.
- No finite zero-valued pixels in the audited active-fire bands.

The baseline loader replaces active-fire NaNs with zero, converts HHMM values to integer hours using floor division by 100, and binarizes the target using `y > 0`. It also appends a binary active-fire input mask derived from the processed active-fire input channel.

The 78 positive pixels are daily rasterized detection pixels, not necessarily 78 unique fires or ignitions. These statistics describe the inspected event only and should not be generalized to the full dataset without a broader audit.

### All-pair temporal missing-mask comparison: fire_22863125 (2019)

A temporal missing-mask comparison was performed for NDVI and EVI2 across all 19 daily TIFF files from 2019-05-18 to 2019-06-05.

All unique date pairs were compared independently for each band, producing 171 comparisons per band and 342 comparisons in total.

| Metric | NDVI | EVI2 |
|---|---:|---:|
| Date-pair comparisons | 171 | 171 |
| Identical mask pairs | 55 | 55 |
| Mean IoU | 0.981271 | 0.981271 |
| Minimum IoU | 0.966616 | 0.966616 |
| Maximum IoU | 1.000000 | 1.000000 |

NDVI and EVI2 produced identical temporal comparison statistics. This is consistent with the daily audit, which found identical missing-pixel masks between these two bands on every date.

The high mean IoU indicates strong temporal spatial consistency in missing-data patterns for this event. However, the 116 non-identical date pairs demonstrate that the masks are not completely invariant over time.

These results describe one event and do not establish the cause of missingness or imply that the same pattern occurs across the full WSTS dataset.

The all-pair comparison results are stored in `wsts_2019_fire_22863125_all_pair_temporal_iou_ndvi_evi2.csv`.
### All-pair temporal missing-mask comparison: fire_22863125 (2019)

A temporal missing-mask comparison was performed for NDVI and EVI2 across all 19 daily TIFF files from 2019-05-18 to 2019-06-05.

All unique date pairs were compared independently for each band, producing 171 comparisons per band and 342 comparisons in total.

| Metric | NDVI | EVI2 |
|---|---:|---:|
| Date-pair comparisons | 171 | 171 |
| Identical mask pairs | 55 | 55 |
| Mean IoU | 0.981271 | 0.981271 |
| Minimum IoU | 0.966616 | 0.966616 |
| Maximum IoU | 1.000000 | 1.000000 |

NDVI and EVI2 produced identical temporal comparison statistics. This is consistent with the daily audit, which found identical missing-pixel masks between these two bands on every date.

The high mean IoU indicates strong temporal spatial consistency in missing-data patterns for this event. However, the 116 non-identical date pairs demonstrate that the masks are not completely invariant over time.

These results describe one event and do not establish the cause of missingness or imply that the same pattern occurs across the full WSTS dataset.

The all-pair comparison results are stored in `wsts_2019_fire_22863125_all_pair_temporal_iou_ndvi_evi2.csv` in the external data manifests directory.