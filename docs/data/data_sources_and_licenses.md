# WSTS Data Sources and Licenses

## 1. Purpose and scope

This document records the provenance of the WildfireSpreadTS (WSTS)
dataset and the source collections identified in the WSTS dataset
creation code.

It distinguishes:
- The license of the distributed WSTS dataset.
- The upstream source collections used to construct its features.
- The processing and aggregation operations visible in the creation code.
- Attribution, version and licensing details that still require
  independent verification.

Identification of an Earth Engine collection does not, by itself,
establish the exact product version, license terms or redistribution
permissions applicable to every exported WSTS feature.

## 2. WSTS dataset provenance

| Item | Recorded information |
|---|---|
| Dataset | WildfireSpreadTS (WSTS) |
| Zenodo record | https://zenodo.org/records/8006177 |
| Dataset license | CC-BY-4.0 |
| Baseline repository | https://github.com/SebastianGer/WildfireSpreadTS |
| Dataset creation repository | https://github.com/SebastianGer/WildfireSpreadTSCreateDataset |
| Pinned baseline commit | `f62dc49f2c2a6d249c071533ae5340c0d01bfb95` |
| Temporal coverage inspected | 2018–2021 |

The WSTS dataset is distributed under CC-BY-4.0 according to its
Zenodo record. This dataset-level license must not be interpreted as
a replacement for the terms governing upstream source products.

## 3. Upstream source collections identified in the creation code

The following source identifiers were found in the inspected
`FirePred.py` dataset-generation code.

| Data category | Source identifier in code | WSTS features associated with category |
|---|---|---|
| Elevation and terrain | `USGS/SRTMGL1_003` | Elevation, slope, aspect |
| Land cover | `MODIS/061/MCD12Q1` | Landcover class |
| Daily weather | `IDAHO_EPSCOR/GRIDMET` | Precipitation, wind, temperature, ERC, specific humidity |
| Forecast weather | `NOAA/GFS0P25` | Forecast precipitation, wind, temperature, specific humidity |
| Drought | `GRIDMET/DROUGHT` | Palmer drought severity index (PDSI) |
| VIIRS reflectance | `NOAA/VIIRS/001/VNP09GA` | VIIRS bands M11, I2 and I1 |
| VIIRS vegetation indices | `NOAA/VIIRS/001/VNP13A1` | NDVI and EVI2 |
| VIIRS active fire | `projects/grand-drive-285514/assets/afall` | Active-fire detection time / mask |

These identifiers are reproduced from the inspected creation code.
The exact source product versions, collection status and licensing
terms must be checked against the relevant provider documentation.

## 4. Processing and aggregation observed in the creation code

The following summarizes operations visible in the inspected
`FirePred.py` implementation. It is not an independent validation
of the scientific suitability of each aggregation.

### 4.1 Terrain

- Elevation is sourced from `USGS/SRTMGL1_003`.
- Slope and aspect are derived from the elevation data.

### 4.2 Land cover

- Land cover is sourced from `MODIS/061/MCD12Q1`.
- The code uses the `LC_Type1` band.
- An annual median operation is applied.

### 4.3 Daily weather

The daily GRIDMET source is `IDAHO_EPSCOR/GRIDMET`.

The inspected code uses these source bands:

| WSTS feature | GRIDMET band identifier |
|---|---|
| Total precipitation | `pr` |
| Wind direction | `th` |
| Minimum temperature | `tmmn` |
| Maximum temperature | `tmmx` |
| Energy release component | `erc` |
| Specific humidity | `sph` |
| Wind speed | `vs` |

The code applies a median reducer to the daily collection.

### 4.4 Forecast weather

The forecast source is `NOAA/GFS0P25`.

The inspected code:
- Selects forecasts initialized at midnight.
- Uses forecast hours F01–F24.
- Calculates means for temperature and specific humidity.
- Derives wind speed and direction from u/v wind components.
- Uses date-dependent logic for forecast precipitation.

The exact forecast initialization, valid-time alignment and
precipitation transition behavior require further verification
before these features are used in a new temporal modelling pipeline.

### 4.5 Drought

- Source: `GRIDMET/DROUGHT`.
- Feature: Palmer drought severity index (PDSI).
- The code filters for an image whose time interval covers the
  target date.

### 4.6 VIIRS reflectance and vegetation indices

Reflectance:
- Source: `NOAA/VIIRS/001/VNP09GA`.
- Bands: M11, I2 and I1.
- The inspected code applies a median reducer.

Vegetation indices:
- Source: `NOAA/VIIRS/001/VNP13A1`.
- Features: NDVI and EVI2.
- The inspected code uses a preceding 15-day time window and
  `Reducer.last`.

### 4.7 Active-fire detections

- Source asset: `projects/grand-drive-285514/assets/afall`.
- The code filters detections by date.
- Low-confidence detections (`confidence == 'l'`) are excluded.
- Acquisition time is read from `acq_time`.
- Detection points are buffered by 187.5 m, corresponding to
  half the 375 m output pixel size.
- The code rasterizes acquisition time using `Reducer.last`.
- The output feature is named `active fire`.

The active-fire asset's ownership, original source citation,
license and redistribution conditions have not yet been
independently established in this document.

## 5. Export and spatial processing

The inspected dataset-preparation service indicates:

- Export scale for `FirePred`: 375 m.
- Export CRS is constructed using the configured UTM zone.
- Exported images are converted to float using `toFloat()`.
- The export region is a rectangle derived from event longitude
  and latitude and a configured rectangular size.
- The inspected configuration uses `rectangular_size = 0.5`
  degrees.
- Images are exported to Google Cloud Storage.

These are implementation observations. They do not establish
that all source products have a native resolution of 375 m.
The export scale is not equivalent to native source resolution.

## 6. Licensing and attribution status

| Source / asset | License or attribution status |
|---|---|
| WSTS Zenodo dataset | CC-BY-4.0, as stated by the dataset record |
| SRTM `USGS/SRTMGL1_003` | Provider terms and attribution still to be verified |
| MODIS `MODIS/061/MCD12Q1` | Provider terms and attribution still to be verified |
| GRIDMET `IDAHO_EPSCOR/GRIDMET` | Provider terms and attribution still to be verified |
| NOAA GFS `NOAA/GFS0P25` | Provider terms and attribution still to be verified |
| GRIDMET drought `GRIDMET/DROUGHT` | Provider terms and attribution still to be verified |
| VIIRS `VNP09GA` | Provider terms and attribution still to be verified |
| VIIRS `VNP13A1` | Provider terms and attribution still to be verified |
| Active-fire asset `afall` | Asset ownership, provenance and license still to be verified |

Do not assume that the WSTS CC-BY-4.0 license independently grants
rights to redistribute every upstream source product or the
`afall` asset.

Before redistribution or external publication, verify the applicable
provider terms, required citations, attribution wording and any
restrictions for each source.

## 7. Outstanding verification tasks

- [ ] Record the access date and exact version of each source product.
- [ ] Locate provider documentation for each Earth Engine collection.
- [ ] Verify license and attribution requirements for each source.
- [ ] Establish provenance and licensing of the `afall` asset.
- [ ] Confirm whether source data are redistributed or only derived
      features are included in WSTS.
- [ ] Verify units and scale factors for all exported feature bands.
- [ ] Verify temporal aggregation and forecast valid-time alignment.
- [ ] Record any source-specific nodata and quality-mask handling.

## 8. Reproduction references

- WSTS dataset record:
  https://zenodo.org/records/8006177
- WSTS baseline:
  https://github.com/SebastianGer/WildfireSpreadTS
- WSTS dataset creation code:
  https://github.com/SebastianGer/WildfireSpreadTSCreateDataset

The source identifiers and processing descriptions in this document
are based on inspection of the dataset creation code. They should be
rechecked against the exact source revision used to generate the
distributed WSTS archive.
