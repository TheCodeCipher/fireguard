# FireGuard — WSTS Source Review and Validation Protocol Record

**Role:** Role 4 — Research / Evaluation / Leakage Audit Engineer  
**Phase:** Phase 1  
**Primary task covered:** Task 1 — Read the WSTS paper and official code; Task 2 supporting evidence recorded in Sections 9 and 10  
**Document:** `docs/validation_protocols.md`  
**Document version:** 1.6  
**Initial source-inspection date:** 2026-11-10  
**Updated through:** 2026-11-13  
**Status:** Consolidated source-review record. Public WSTS paper and inspected baseline code have been reviewed in depth. Role 1 geospatial handoff has been reviewed. Archive date-continuity and embedded TIFF metadata have been verified at sample/archive level (Sections 9 and 10). Exact FireGuard-pinned loader revision, canonical channel-level temporal semantics, and Level B operational availability remain pending. This document does **not** claim that all Role 4 Phase 1 tasks or acceptance tests are complete. **Version 1.6 adds Sections 14-16** (P1 protocol, P2 deferral, and the 2026-11-13 status of every dependency and action in Sections 8 and 13); Sections 0-13 are unchanged.

---

## 0. Authority, scope, and evidence conventions

### 0.1 Controlling project documents

This record is governed by:

1. `FireGuard_Final_Project_Plan_v4.md` — authoritative source for project scope, architecture, dataset decisions, temporal semantics, leakage rules, evaluation protocol, and phase gates.
2. `FireGuard_Phase1_Phase2_Rolewise_Execution_Guide.md` — Role 4 task sequence, deliverables, dependencies, and acceptance criteria.

If this document appears to conflict with Plan v4, Plan v4 takes precedence. This document records evidence and status; it does not change locked project scope or architecture.

### 0.2 Scope of this file

This file records the WSTS paper/source review, the independently inspected public WSTS implementation, Role 1's geospatial handoff, archive-level date-continuity and embedded-metadata evidence, metadata observations supplied during the work, and the unresolved dependencies needed for subsequent Role 4 tasks.

It is **not by itself**:
- the complete channel-by-channel leakage register (`docs/leakage_audit.csv`) — that file exists but remains incomplete;
- the related-fire grouping report (`docs/fire_complex_groups.md`);
- the frozen P1 LOYO or validation manifests;
- the split hash record;
- the metrics implementation or final metrics contract;
- proof that Role 4 Phase 1 acceptance tests have passed.

Those are separate Phase 1 artifacts.

### 0.3 Evidence labels

Use the following distinctions throughout:

- **VERIFIED (source/code):** directly supported by a reviewed paper, repository file, command output, or supplied artifact, within the stated scope.
- **REPORTED BY ROLE 1:** stated in the Role 1 handoff; not independently reproduced by Role 4 unless explicitly stated.
- **INDEPENDENT ROLE 4 INSPECTION:** observed in the separately cloned public WSTS repository at the commit recorded below. It is not automatically the FireGuard-pinned implementation.
- **ARCHIVE-LEVEL VERIFIED (Role 4):** observed directly from the released WildfireSpreadTS ZIP archive without extraction (Sections 9 and 10).
- **UNKNOWN / PENDING:** required evidence has not been supplied or verified. Unknown is not PASS.
- **INFERENCE / CANDIDATE:** a derived observation useful for planning, but not a final project decision or validated result.

### 0.4 No-invention and ownership rules

- Do not invent channel names, channel order, product versions, temporal windows, `available_at`, tensor dimensions, CRS transformations, API behavior, model performance, or experiment results.
- The pinned official WSTS loader and Role 1's actual sample/schema handoff are authoritative for FireGuard sample semantics, channel order, dimensions, and target construction.
- Role 4 consumes Role 1's canonical data dictionary; Role 4 does not silently take ownership of creating or redefining it.
- Role 3 owns the detailed GFS/GRIDMET temporal and wind-direction inspections and supplies evidence to Role 4.
- Unknown Level-A channels are excluded from strict-model candidate inputs until resolved.
- Do not commit raw WSTS/HLS/DEM data, credentials, private reports, or large checkpoints to Git.

---

## 1. Primary benchmark sources

| Field | Recorded details | Evidence / status |
|---|---|---|
| Dataset | WildfireSpreadTS (WSTS) | NeurIPS paper and Zenodo record |
| Paper | Sebastian Gerard, Yu Zhao, and Josephine Sullivan, "WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction," NeurIPS 2023 Datasets and Benchmarks Track | Paper and supplementary material reviewed |
| Dataset record | Zenodo record 8006177 | Official dataset record |
| Dataset licence | CC-BY-4.0 as recorded for the dataset in the official dataset/baseline materials | Preserve distinction between dataset licence and code-repository licence |
| Dataset period | January 2018 to October 2021 | Paper abstract |
| Retained WSTS event records | 607 | Paper |
| Daily images | 13,607 | Paper |
| Base input channels | 23 | Paper and public baseline source |
| Geographic coverage | Contiguous United States; retained events are concentrated in the western United States | Paper/supplementary material |
| Active-fire grid resolution | 375 m | Paper |
| Primary benchmark target | Next-day binarized VIIRS active-fire detection map | Paper and loader documentation |
| Target product | Suomi-NPP VIIRS Active Fire product (VNP14IMG) | Paper |
| Public baseline repository | `https://github.com/SebastianGer/WildfireSpreadTS` | Independently cloned and inspected |
| Dataset-creation repository | `https://github.com/SebastianGer/WildfireSpreadTSCreateDataset` | README notice and specific commit inspected; see Sections 5 and 6 |
| Dataset archive record | `https://zenodo.org/records/8006177` | Official Zenodo record |

### 1.1 Event selection and retained sample

The paper describes selecting GlobFire events larger than 1,000 hectares for 2018–2021. Image sequences were extracted around event areas and included four days before and four days after the GlobFire event dates. Events without VIIRS active-fire detections during the relevant period, together with a small number affected by data-format issues, were discarded. The paper reports 607 retained events and 948 discarded events.

The paper attributes much of the filtering to differences between the MODIS burned-area product used by GlobFire and the VIIRS active-fire product used to construct the active-fire maps. A burned-area event and a satellite active-fire detection sequence are not identical observations.

**Role 4 implication:** 607 WSTS event records must not be treated as 607 independent fire complexes. Related-fire grouping is a separate Role 4 task and must follow Plan v4. The event count alone does not establish group membership.

### 1.2 Year-wise event counts

| Year | Events |
|---:|---:|
| 2018 | 176 |
| 2019 | 74 |
| 2020 | 201 |
| 2021 | 156 |
| **Total** | **607** |

The 2019 event count is smaller than the other yearly counts. P1 fold reports must show per-year event and group counts, and interpretation must acknowledge possible lower statistical power for the 2019 held-out fold.

### 1.3 Supplementary documentation evidence

The NeurIPS supplementary dataset documentation provides details beyond the main paper that directly affect the leakage audit and evaluation design:

| Field | Recorded detail | Relevance to Role 4 |
|---|---|---|
| Daily aggregation window | Active-fire detections aggregated within **24-hour windows** inside an area derived from the GlobFire database entry | The window start/end convention must be verified against the pinned loader; do not assume midnight-aligned or a specific time zone from the paper alone |
| Confidence filtering | Low-confidence detections removed; **medium and high confidence kept** | Target label definition includes medium-confidence detections, not only high. This affects the label-noise audit scope |
| GFS wind transformation | GFS wind originally **u and v components**; transformed to **overall wind speed and wind direction** to match GRIDMET features | If the stored direction is lossy, the original u/v may exist in creation code or raw data. Role 3 must verify whether reconstruction is possible from released files |
| GFS temporal aggregation | GFS features originally **hourly**; aggregated to **daily resolution** | The aggregation method (e.g., mean, max, specific hour) is not specified in the supplementary text. Pinned creation code must be inspected |
| Non-daily feature handling | For features updated less often than daily, the **most recent value** is taken; subsequent days often contain the same values | This applies to VNP13A1 (16-day composite), MODIS land cover (yearly), and potentially others. A "daily" tensor index does not mean a daily observation |
| SRTM vintage | SRTMGL1 v003 collected in **year 2000** | Static elevation is not temporally aligned with 2018–2021 events. This is expected but must be documented as a known static-layer limitation |
| Geographic scope | Dataset contains fires in the **western USA**; some source products are only available in the USA | Confirms the geographic limitation already noted in Section 2 and Section 4 |
| Event date accuracy caveat | GlobFire dates used to determine the time range may **not be completely accurate**; 4 days before/after were included and often contained active fires (about 3/8 days on average) | The "forecast origin" inherited from WSTS sample construction carries this uncertainty. Do not treat the dataset's date labels as verified fire-start or fire-end dates |

**Evidence status:** `VERIFIED` from supplementary documentation. This does not replace the pinned-loader verification required for exact window boundaries.

---

## 2. Benchmark target and temporal description

The WSTS benchmark predicts active-fire detections at a 24-hour horizon. The paper describes aggregating VIIRS active-fire detections into 24-hour windows beginning at midnight. When multiple detections occur in a pixel during a window, the active-fire map records the latest detection time. Low-confidence detections are removed. The next day's active-fire map is binarized to form the prediction target.

The paper describes daily alignment of other data sources to this temporal resolution. However, the input channels do not all originate from products with daily native temporal support. For example:
- VNP13A1 vegetation indices are derived from a 16-day composite product.
- VNP09GA surface reflectance is described as a daily product.
- Forecast products have initialization, creation/publication, and valid-time semantics that must be audited separately.

### 2.1 Two independent leakage levels

**Level A — no look-ahead (`no_lookahead`):** Does the source observation/composite window contain information from or after the target-window start? This is a temporal-information test.

**Level B — operational availability (`operationally_available`):** Was the source actually published, processed, and accessible by the forecast origin (`available_at <= t0`)? This is an operational data-readiness test.

These fields are independent. A channel may pass Level A but fail or remain unknown at Level B. A Level-A PASS must never be described as proof of operational availability.

The detailed per-channel evidence and decisions belong in `docs/leakage_audit.csv`.

### 2.2 Sample-level temporal evidence currently available

Role 1's handoff identifies one verified loader sample:

| Field | Handoff value |
|---|---|
| Event | `fire_21458798` |
| Manifest year | 2018 |
| Input date | 2018-01-06 |
| Target date | 2018-01-07 |

This is one sample-level example. It does not by itself prove dataset-wide consecutive-day spacing, absence of missing dates, or identical semantics for every event and every loader revision.

Section 9 adds archive-level date-continuity evidence: across all 607 event directories and 13,607 TIFF files, no gaps, missing days, or duplicate dates were found between each event's first and last observed date. This supports calendar-date continuity of the archived image sequences. It does not establish exact sensing/composite windows, forecast issue times, or operational availability.

The independent public-code inspection in Section 5 establishes that the inspected loader selects the next stored image position as the target. The inspected functions do not validate calendar-day continuity themselves. Combined with Section 9, the loader + archive evidence jointly supports a positional-and-calendar-continuous sequence for the released archive. This remains subject to reconciliation with Role 1's pinned loader and any local patches.

### 2.3 Additional detail on target aggregation and confidence

The paper and supplementary documentation together establish:

- The target is a **binarized next-day active-fire map** at 375 m.
- Detections are aggregated within 24-hour windows **inside the GlobFire-derived event area**.
- **Low-confidence detections are removed; medium and high confidence are retained.**
- When multiple detections occur in a pixel within the window, the **latest detection time** is recorded in the active-fire map.

**Implication for Role 4:** The phrase "active fire" in this benchmark means "medium- or high-confidence VIIRS active-fire detection within the aggregation window and event area," not "any thermal anomaly" and not "verified physical fire perimeter." This wording must be used consistently in the metrics contract and any replay UI labelling.

---

## 3. Dataset characteristics and limitations relevant to evaluation

| Limitation | What the source supports | Relevance to FireGuard |
|---|---|---|
| Noisy active-fire labels | The paper identifies noise from smoke, clouds, and inaccuracies in active-fire detection. | Treat the target as an active-fire detection proxy, not complete ground truth for physical fire spread. |
| Incomplete observability | WSTS does not provide complete cloud/observation validity information for interpreting every next-day negative cell. | Next-day VNP09GA observability is evaluation-only under Plan v4. Do not turn unobserved cells into confirmed negatives or feed next-day observability into model inputs. |
| Severe class imbalance | The paper reports approximately 0.1% active-fire pixels. | AUPRC is primary under the project protocol; accuracy alone is not an adequate performance summary. |
| Geographic concentration | Retained events are concentrated in the western United States. | Geographic generalization is limited. P2 geographic holdout remains distinct from P1 LOYO. |
| Uneven year counts | 2019 has 74 events, compared with 176, 201, and 156 in the other years. | Report per-fold counts and uncertainty; note lower statistical power where applicable. |
| Variable frame dimensions | The paper reports dimensions ranging from 304 × 207 to 356 × 308. | Preserve the WSTS grid and follow Plan v4 padding/unpadding and full-frame evaluation requirements. Do not infer fixed dimensions from one sample. |
| Mixed temporal support | Daily sequences include sources with different native temporal support, including 16-day vegetation-index composites. | A daily array index does not prove a daily source observation. Audit each channel's sensing/composite window. |
| Event definition mismatch | GlobFire burned-area events and VIIRS active-fire detections are different products and can disagree. | Do not equate WSTS event records with independently verified fire perimeters or final fire complexes. |

---

## 4. Public repository provenance and source-review method

### 4.1 Independently inspected baseline repository

| Item | Role 4 inspection record |
|---|---|
| Repository | `https://github.com/SebastianGer/WildfireSpreadTS.git` |
| Branch | `main` |
| Inspected commit | `f62dc49f2c2a6d249c071533ae5340c0d01bfb95` |
| Clone method | `git clone --depth 1` |
| Working tree | Clean at time of inspection |
| Inspection date | 2026-11-10 |
| FireGuard / Role 1 pinned revision | **UNKNOWN — not confirmed by this inspection** |

The inspected commit is provenance for Role 4's independent public-code review only. It must not be represented as the exact revision used by FireGuard. Role 1 must supply its pinned commit and any local patches before the two implementations can be reconciled.

### 4.2 Public README and repository notices

The public repository README was reviewed for project description, data structure, dataset usage, loader entry points, HDF5 preparation, post-publication correction notice, and angle-feature warning.

Findings recorded from the README:
- The loader/DataModule entry points include `src/dataloader/FireSpreadDataset.py` and `src/dataloader/FireSpreadDataModule.py`.
- The README documents conversion of GeoTIFF data to HDF5 for faster training and states that the HDF5 representation can take approximately twice the storage space. This is a storage/performance note, not a requirement to extract or convert the archive for this source-review document.
- The README identifies commit `ab3c8f35c5ec8c52c306a4488eaeb71a5a13d0de` as a dataset-class bug fix. The README describes slightly higher performance in initial experiments while overall result trends remained broadly similar.
- A February 2026 README update warns that applying sine without cosine to angle features can lose directional information.
- The dataset-creation repository README contains a February 2026 notice concerning a wind-direction computation correction.

These README notices do not replace inspection of the exact FireGuard-pinned code, raw value ranges, or product metadata.

### 4.3 Dataset-creation repository

- Repository: `https://github.com/SebastianGer/WildfireSpreadTSCreateDataset`
- Purpose: code for computing/creating WSTS using Google Earth Engine.
- Exact commit inspected by Role 4 during later leakage-audit work: `f6c5167279a4051f68d0cca55495f4693598096e` (recorded in `docs/leakage_audit.csv` evidence rows).
- README-level wind-direction correction notice reviewed.
- Full source revision reconciliation with Role 1's pinned revision remains pending.
- Licence: inspect the checked-out `LICENSE` before recording a code-licence conclusion. Do not infer the code licence from the dataset licence or vice versa.

---

## 5. Detailed inspection of the independently cloned public baseline code

**Scope warning:** all code findings in this section refer to the independently inspected public repository commit `f62dc49f2c2a6d249c071533ae5340c0d01bfb95`, unless a separate Role 1 handoff is explicitly identified. These findings must be reconciled with Role 1's pinned loader before being treated as canonical FireGuard behavior.

### 5.1 `src/dataloader/FireSpreadDataset.py` — sample selection and target construction

The inspected `load_imgs` logic:
- starts from an event image index (`in_fire_index`) and applies `skip_initial_samples`;
- computes `end_index = in_fire_index + n_leading_observations + 1`;
- for HDF5, reads the selected slice from `f["data"][in_fire_index:end_index]`;
- separates input observations from the final selected image;
- uses the active-fire band of the final selected image as the target;
- for the TIFF path, uses `imgs[:-1]` as inputs and the final image's final active-fire band as the target.

The dataset docstring describes the target as the next day's binary active-fire mask. The inspected implementation selects the **next stored image position**. The reviewed functions do not themselves verify exact calendar-date spacing between every adjacent pair; Section 9 provides archive-level evidence that the released image sequences do have consecutive dates.

**Conclusion:** code supports "next selected/stored image" target construction. "Exactly next calendar day for every sample" is archive-level supported by Section 9, but the exact forecast issue-time and sensing/composite windows remain **UNKNOWN**.

### 5.2 Input count and test adjustment

The inspected constructor computes:

`skip_initial_samples = n_leading_observations_test_adjustment - n_leading_observations`

For the inspected monotemporal config:
- `n_leading_observations = 1`
- `n_leading_observations_test_adjustment = 5`
- resulting `skip_initial_samples = 4`

This skips the first four candidate starting positions for the configured test adjustment. It is a **positional sample skip**, not an increase in input sequence length: the model still receives one leading observation in this configuration. It does **not** mean that the model receives five input days.

Only the named config `cfgs/data_monotemporal_full_features.yaml` was inspected for these values. Do not generalize the values `5`, `1`, or the resulting skip of `4` to multi-observation configurations, other WSTS configs, or the FireGuard configuration.

### 5.3 Image ordering and temporal integrity checks

The inspected `FireSpreadDataset.py` contains three relevant helper functions in the supplied PowerShell source excerpt:

- `find_image_index_from_dataset_index()` — lines 80–120;
- `read_list_of_images()` — lines 211–245;
- `compute_datapoints_per_fire()` — lines 247–270.

The loader's ordering and sample-count behavior is positional. The inspected helper functions do not establish calendar-day continuity by themselves; Section 9 adds independent archive-level evidence that the actual released sequences are consecutive.

#### 5.3.1 `read_list_of_images()` — lines 211–245

The supplied PowerShell output shows the following behavior in the inspected public source:

**TIFF path**
- obtains the fire-directory paths for a year;
- sorts the fire-directory list using `fires_in_year.sort()`;
- for each fire directory, obtains its TIFF image paths;
- sorts the TIFF paths within that fire using `fire_img_paths.sort()`;
- derives the fire name using `fire_dir_path.split("/")[-2]`;
- when a fire directory contains zero TIFF images, emits `warnings.warn(...)` rather than treating that condition as a fatal error in this function.

**HDF5 path**
- obtains the HDF5 event-file paths;
- sorts the HDF5 file list using `fires_in_year.sort()`;
- derives the fire name using `Path(fire_hdf5).stem`.

**Interpretation and limitations**
- Lexicographic sorting is the ordering operation observed in the function. It is not an explicit date-continuity validation.
- The TIFF and HDF5 paths derive the event name using different expressions.
- The zero-TIFF behavior is a warning at this function boundary. Whether a zero-image event is subsequently filtered, causes an error, or affects downstream indexing must be verified from the complete pinned loader path.
- The function's sort order does not by itself prove that all filenames are valid, unique, chronological, or exactly one calendar day apart. Section 9 provides archive-level evidence that the released filenames do produce consecutive dates after sorting.

#### 5.3.2 `compute_datapoints_per_fire()` — lines 247–270

The supplied PowerShell output shows that the inspected function derives the number of candidate samples per event from the number of stored images and the configured input/skip values.

The recorded formulas are:

```python
# TIFF path
n_fire_imgs = len(fire_imgs) - self.skip_initial_samples

# HDF5 path
n_fire_imgs = len(f["data"]) - self.skip_initial_samples
```

> **Merge note:** the first supplied file ends inside this code block at a `# Both paths` comment, and the second supplied file resumes with the explanatory text below. The "both paths" formula lines (including the `datapoints_in_fire` computation) were present in neither supplied file and are intentionally **not reconstructed** here. Recover them from the inspected revision (or the original PowerShell output) before using this subsection as a test oracle.

The source comment explains the positional logic: if there are two days of observations and a one-day lead, the first image is used as the observation and the second image supplies the target; the event therefore contributes one prediction sample.

The displayed PowerShell excerpt reaches the `if datapoints_in_fire <= 0:` branch at lines 269–270, but does not show enough of the branch body to establish its exact action. Accordingly, the precise small-event handling (raise, skip, warn, or another action) is **UNKNOWN** from the supplied excerpt and must not be guessed. Inspect the complete function in the pinned revision before documenting that outcome.

This count is based on stored image positions. It does not itself validate calendar dates, missing days, or duplicate dates. Section 9 provides archive-level evidence that no gaps, missing dates, or duplicate dates were observed.

#### 5.3.3 `find_image_index_from_dataset_index()` — lines 80–120

The supplied PowerShell output shows that the inspected function maps a global dataset index to an event and an event-local starting index using the previously computed per-event sample counts.

Recorded behavior:
- a negative `target_id` is normalized using `target_id = self.length + target_id`;
- an index outside the valid dataset range results in a `RuntimeError`;
- the function walks the per-event `datapoints_per_fire` counts to identify which event contains the requested global sample index;
- the return signature is `(fire_year, fire_name, in_fire_index)`.

The mapping therefore depends on the event ordering and per-event counts produced by `read_list_of_images()` and `compute_datapoints_per_fire()`. This function performs positional lookup; no chronological-date check is established by the supplied excerpt.

The exact error message and all boundary conditions should be rechecked against the complete pinned function before being used as a test oracle.

#### 5.3.4 `load_imgs()` — lines 121–162

The inspected `load_imgs()` logic computes the end of the selected image slice as:

```python
end_index = in_fire_index + self.n_leading_observations + 1
```

For the HDF5 path, the selected images are read using:

```python
imgs = f["data"][in_fire_index:end_index]
```

The TIFF path selects the corresponding sequence from the event's ordered image list. The selected observations are used as input and the final selected image supplies the target active-fire mask, as described in Section 5.1.

This formula explains why the selected slice contains `n_leading_observations` input image(s) plus one target image. It remains a positional selection. Section 9 confirms archive-level calendar-date continuity; exact sensing windows remain pending.

#### 5.3.5 Date-order and integrity conclusions

The inspected source extracts dates from TIFF filenames and sorts paths lexicographically. The HDF5 event files are sorted, while image order inside an HDF5 event dataset is inherited from the TIFF ordering used during conversion.

The reviewed functions:
- do not assert unique dates within every event;
- do not assert monotonic parsed dates after loading;
- do not assert exact 24-hour differences between consecutive images;
- do not detect every missing calendar date;
- do not establish that duplicate date files are impossible;
- do not compare each stored image position with a validated calendar-day offset.

Section 9 provides the independent archive-level verification these functions do not perform. The following remain pending:
- Whether Role 1's pinned implementation performs the same archive validation.
- Whether FireGuard's loader retains the same per-event count formula after any filtering/grouping.
- Reconciliation between the independently inspected public commit and Role 1's pinned commit.

### 5.4 Active-fire, target, and time preprocessing observed in the TIFF path

The inspected TIFF path includes the following behavior:
- active-fire NaN values are replaced with zero;
- target NaN values are replaced with zero;
- VIIRS active-fire detection time (HHMM) is converted to hour using integer floor division by 100;
- the target is binarized using `y = (y > 0).long()`;
- the binary active-fire input mask is computed and appended after standardization;
- the binary mask is therefore separate from the standardized continuous channels.

These are code-path observations from the inspected public revision. They do not prove that every FireGuard data path or Role 1 local patch behaves identically.

**Interpretation caution:** HHMM converted to integer hour loses minute-level precision. It must not be treated as a continuous ordinary scalar without an approved representation. Plan v4 requires the actual time-window and mask semantics to be verified; if a time-of-day feature is used, the approved encoding and valid-fire-pixel mask must be documented and tested.

### 5.5 Normalization and feature exclusions observed in `src/dataloader/utils.py`

The inspected helper code:
- computes means and standard deviations using the indicated training years;
- computes statistics from unaugmented, unstandardized data;
- excludes degree features and categorical land-cover from standardization;
- identifies degree-feature indices as `[7, 13, 19]`;
- adds index `16` (land-cover) to the non-standardized feature list.

The exact mapping from these indices to the FireGuard canonical channel schema remains subject to Role 1's pinned channel dictionary. Do not infer channel identity from index values alone.

### 5.6 Degree-feature transformation and directional-information warning

The inspected `FireSpreadDataset.py` applies sine after converting degree features to radians. The README warns that sine-only encoding can lose directional information because it is non-injective (different angles can share the same sine value). The README recommends sine plus cosine to preserve full-circle angular information.

This is an audit finding, not permission to change the locked FireGuard architecture or to assume that a stored angle has full-circle information. Before using a direction-dependent feature or baseline:
- verify the raw angle range and encoding;
- verify whether quadrant information is preserved;
- check whether raw vector components exist;
- use Role 3's documented direction audit;
- keep downwind-dependent methods disabled unless the required evidence passes.

### 5.7 Feature mapping exposed by the inspected public source

The inspected public `FireSpreadDataset.py` feature mapping exposes the following 23 base feature labels in the `only_base` path. This table documents the public source inspected by Role 4; it is not a substitute for Role 1's canonical FireGuard data dictionary and must not be treated as a confirmed FireGuard tensor order until the pinned revision is reconciled.

| Zero-based index in inspected public source | Feature label in inspected source |
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
| 16 | Land-cover class |
| 17 | Forecast: Total precipitation |
| 18 | Forecast: Wind speed |
| 19 | Forecast: Wind direction |
| 20 | Forecast: Temperature |
| 21 | Forecast: Specific humidity |
| 22 | Active fire |

The inspected source also defines 17 land-cover classes. In the non-`only_base` path, the public code expands the land-cover class into 17 channels and appends a binary active-fire channel. The resulting processed representation is 40 channels: the first 16 base features, 17 land-cover one-hot channels, the six base features from indices 17–22, and one binary active-fire channel. The `only_base=True` path returns 23 base features.

Do not conflate:
- the paper's 23 base input channels;
- the processed 40-channel full-feature representation in the inspected public code;
- the exact FireGuard model input channel list, which must be derived from Role 1's pinned loader and the strict leakage decisions.

Section 10 independently confirms, from the embedded `GDAL_METADATA` tag of the inspected 2018 TIFF, that the released sample descriptions follow this 23-channel order.

### 5.8 Augmentation behavior observed in the public source

The inspected training augmentation code includes:
- random square cropping with candidate crops scored to prioritize target fire pixels and then input fire pixels;
- horizontal flips with angle adjustment `360 - angle`;
- vertical flips with angle adjustment `(180 - angle) % 360`;
- random rotations by multiples of 90 degrees with angle adjustment `(angle - 90 * rotate) % 360`.

Plan v4 says geometric augmentation is disabled by default until directional transformations are correct and unit-tested. Role 4 does not authorize enabling these augmentations merely because the public baseline code contains them. FireGuard's implementation must follow the locked plan and the relevant tests.

### 5.9 `cfgs/data_monotemporal_full_features.yaml`

The inspected config contains:

```yaml
data_dir: DATA_DIR_PATH
batch_size: 64
n_leading_observations: 1
crop_side_length: 128
load_from_hdf5: true
num_workers: 8
remove_duplicate_features: false
features_to_keep: null
n_leading_observations_test_adjustment: 5
```

This is one inspected public configuration. It is not proof that FireGuard uses the same sequence length, crop, loader mode, or test adjustment.

### 5.10 `src/preprocess/CreateHDF5Dataset.py` and HDF5 conversion

The complete inspected `CreateHDF5Dataset.py` excerpt:
- iterates over years `[2018, 2019, 2020, 2021]`;
- instantiates `FireSpreadDataset` with `n_leading_observations=1`, `load_from_hdf5=False`, `is_train=True`, and `stats_years=(2018, 2019)` for the conversion process;
- obtains event data from the HDF5 generator;
- writes a per-event HDF5 `data` dataset and attributes including `year`, `fire_name`, `img_dates`, and `lnglat`.

The inspected generator:
- traverses year/event inventories;
- reads TIFFs and collects `lnglat` from the first TIFF's dataset metadata;
- extracts dates from TIFF filenames;
- yields event/date/location/data information;
- applies active-fire NaN handling and the inspected HHMM-to-hour conversion.

The conversion code preserves image dates in HDF5 attributes, but the inspected code does not perform a calendar-continuity assertion before writing the event sequence. Section 9 provides archive-level evidence that the released sequences are consecutive.

The README's statement that HDF5 may require approximately twice the storage of the GeoTIFF archive is a planning/storage note. HDF5 conversion is not required merely to finish Role 4's paper/code source review.

### 5.11 Original WSTS DataModule folds are not FireGuard P1 folds

The inspected public `FireSpreadDataModule.py` uses original year-based benchmark split logic, described in the source review as 12 year-based folds. These are not the FireGuard P1 protocol.

FireGuard P1 is mandatory leave-one-year-out with event-group-safe train/validation/test assignment. Do not reuse the original benchmark folds as FireGuard's frozen P1 manifests.

### 5.12 Source line-range traceability

The following line ranges are the ranges recorded from the PowerShell source outputs supplied during the review. They refer to the independently inspected public `FireSpreadDataset.py` at commit `f62dc49f2c2a6d249c071533ae5340c0d01bfb95`; line numbers can change in another revision.

| File | Inspected line range | Function / behavior documented |
|---|---|---|
| `src/dataloader/FireSpreadDataset.py` | 18–80 | Constructor/configuration state, including leading observations and test adjustment |
| `src/dataloader/FireSpreadDataset.py` | 80–120 | `find_image_index_from_dataset_index()`; global-to-event-local index mapping |
| `src/dataloader/FireSpreadDataset.py` | 121–162 | `load_imgs()`; input/target image slice selection |
| `src/dataloader/FireSpreadDataset.py` | 211–245 | `read_list_of_images()`; TIFF/HDF5 path discovery, sorting, fire-name derivation, empty-directory warning |
| `src/dataloader/FireSpreadDataset.py` | 247–270 | `compute_datapoints_per_fire()`; per-event candidate sample-count formula and small-event branch |
| `src/dataloader/FireSpreadDataset.py` | 280–294 | Degree-feature transformation / sine encoding in the inspected path |
| `src/dataloader/FireSpreadDataset.py` | 296–348 | TIFF active-fire and target preprocessing, HHMM conversion, standardization and binary-mask handling |
| `src/dataloader/FireSpreadDataset.py` | 337–345 | Additional inspected excerpt covering degree-feature / binary-mask processing; overlaps the broader preprocessing range above |
| `src/dataloader/FireSpreadDataset.py` | 365–432 | Training augmentation and angle adjustments for flips/rotations |
| `src/dataloader/FireSpreadDataset.py` | 549–580 | `map_channel_index_to_features()` / 23 base feature labels |
| `src/dataloader/FireSpreadDataset.py` | 611–644 | HDF5 generator; event traversal, date extraction, location metadata and yielded data |
| `src/dataloader/utils.py` | 6–11 | Normalization helper setup / training-statistics behavior inspected |
| `src/dataloader/utils.py` | 140–170 | Feature indices excluded from standardization and degree-feature helper |
| `src/preprocess/CreateHDF5Dataset.py` | 1–48 | HDF5 conversion configuration and per-event dataset/attribute writing |

**Traceability limitation:** these line ranges document the supplied public-source inspection output. They are not guaranteed to match Role 1's pinned revision. Reconfirm them against the exact pinned commit before using them as code-review line references in a final implementation audit.

### 5.13 Source-review conclusion

The independent public source review now covers:
- sample selection and next-stored-image target construction (`load_imgs()`);
- event-directory and HDF5 file ordering/name behavior (`read_list_of_images()`);
- per-event candidate sample-count formula and small-event branch (`compute_datapoints_per_fire()`);
- global dataset-index to event-local-index mapping (`find_image_index_from_dataset_index()`);
- input count and test adjustment;
- filename ordering and date parsing;
- absence of calendar-continuity checks inside the inspected functions (supplemented by Section 9 archive evidence);
- active-fire and target preprocessing;
- HHMM conversion;
- normalization and feature exclusions;
- angle transformation and directional warning;
- 23 base feature labels and 40-channel expanded representation (corroborated by Section 10 embedded metadata);
- augmentation transformations;
- one inspected monotemporal configuration;
- HDF5 conversion and stored metadata;
- original benchmark year-based split behavior.

These findings materially extend the earlier partial review. They remain tied to the independently inspected public commit until Role 1's pinned revision and local patches are supplied and compared.

---

## 6. Role 1 geospatial handoff review

### 6.1 Handoff identity and provenance

- **Artifact:** `Role_4_Geospatial_Handoff_WSTS.pdf`
- **Direction:** Role 1 — WSTS / Geospatial Data Engineer → Role 4
- **Prepared:** 5 November 2026
- **Branch reported:** `Geospatial-Data-Engineer`
- **Latest checkpoint reported by Role 1:** `a303778` — "Update WSTS Phase 1 documentation"
- **Remote reported:** `origin/Geospatial-Data-Engineer`
- **Working tree:** reported clean by Role 1.

This handoff reports metadata scope and requests Role 4's requirements. It explicitly does not claim that event grouping, spatial split design, or split validation is complete.

### 6.2 Role 1 reported inventory facts

| Item | Role 1 handoff value | Scope / qualification |
|---|---|---|
| Event metadata records | 607 | One metadata record per WSTS event |
| Year coverage | 2018–2021 | Event inventory |
| Expected / found TIFFs | 13,607 / 13,607 | Handoff reports zero missing and zero unexpected TIFF paths |
| Missing / unexpected TIFF paths | 0 / 0 | Reported inventory check |
| Metadata scope | First TIFF inspected for each of 607 events | Not every daily TIFF |
| CRS | EPSG:32610 | For inspected first TIFFs |
| Resolution | 375 m × 375 m | For inspected first TIFFs |
| Sample manifest example | `fire_21458798`, 2018 | One verified loader example |
| Sample input / target | 2018-01-06 / 2018-01-07 | One sample only |
| Split assignments | None | Role 1 had not assigned spatial or temporal splits |

The inventory count is Role 1-reported evidence. It is not a claim that Role 4 independently opened and validated all 13,607 rasters, though Sections 9 and 10 add archive-level and cross-year sample evidence respectively.

### 6.3 Coordinate and geometry interpretation

The handoff states:
- `center_x` and `center_y` are projected bounding-box centre coordinates in metres under EPSG:32610.
- They are not longitude/latitude.
- The event metadata contains left, bottom, right, top bounds and an affine transform for the first TIFF inspected for each event.
- `metadata_scope` is `first_tiff_of_event`.
- Daily grid consistency within each event has not been verified.

Therefore:
- do not interpret projected metre coordinates as longitude/latitude;
- do not use a bbox centre as if it were a fire-perimeter centroid;
- do not assume all daily rasters in an event have identical dimensions, transforms, or bounds;
- do not infer complete fire footprints or perimeters from first-TIFF bounding boxes.

### 6.4 Role 1 handoff limitations

The PDF explicitly states:
- The 607 event records do not define independent fire complexes or spatial groups.
- A projected bounding-box centre is only a summary point, not a substitute for the full event footprint/perimeter.
- First-TIFF inspection does not establish daily raster grid consistency.
- Longitude/latitude must not be inferred from EPSG:32610 coordinates without an explicit CRS transformation.
- No spatial or temporal split assignments were made.
- The single sample manifest record is not dataset-wide loader validation.

### 6.5 Role 4 metadata-file observations supplied during the work

The supplied `wsts_event_metadata.csv` was reported as having shape 607 × 24, with no nulls in the inspected columns. The columns were:

`year`, `event_id`, `tiff_count`, `first_date`, `last_date`, `width`, `height`, `band_count`, `crs`, `resolution_x`, `resolution_y`, `left`, `bottom`, `right`, `top`, `transform`, `center_x`, `center_y`, `metadata_source_file`, `metadata_scope`, `coordinate_units`, `center_coordinate_type`, `metadata_source_path`, `longitude_latitude_available`.

Reported checks and results:

| Check | Reported result | Interpretation / limit |
|---|---|---|
| Row count | 607 | Matches event-record count |
| Nulls | None in inspected CSV | Does not validate unseen raster pixels |
| CRS | EPSG:32610 for all 607 rows | Metadata is first TIFF per event |
| Band count | 23 for all 607 rows | First-TIFF metadata scope |
| Resolution | 375 m × 375 m for all 607 rows | First-TIFF metadata scope |
| Date parsing | Valid in inspected metadata | Does not establish no missing dates between first and last (Section 9 provides independent archive-level continuity evidence) |
| `first_date > last_date` | None reported | Basic date-range consistency only |
| TIFF count sum | 13,607 | Matches reported inventory |
| Minimum / maximum TIFFs per event | 9 / 94 | Event sequence lengths vary |
| Metadata scope | `first_tiff_of_event` | Daily grid consistency remains unverified |

Reported year-wise event and TIFF totals:

| Manifest year | Events | TIFFs |
|---:|---:|---:|
| 2018 | 176 | 3,773 |
| 2019 | 74 | 1,425 |
| 2020 | 201 | 4,292 |
| 2021 | 156 | 4,117 |
| **Total** | **607** | **13,607** |

**Year-label caution:** the metadata includes an event with manifest year 2018 (`fire_21458836`) whose first date is 2017-12-29 and last date is 2018-02-01. Do not silently reassign the event year based only on its first date. Preserve the authoritative manifest year and document the cross-year date span.

### 6.6 Bounding-box candidate observations

The metadata analysis reported:
- 4,113 candidate event-pair bounding-box overlaps.
- 115 candidate pairs with bbox-centre distance at most 30 km and date-interval gap at most 14 days.

These are candidate-pair observations only. They are not final related-fire groups, verified perimeter overlaps, or a validated grouping result. Bounding boxes may overlap even when actual fire perimeters do not. Centre-distance screening can miss geometry-specific relationships and must not replace the Plan v4 grouping rule.

No group IDs, group-size distribution, largest-group share, or final single-linkage/complete-linkage decision is established by these candidate counts.

### 6.7 Handoff artifacts and outstanding inputs

Role 1 identifies these repository artifacts:
- `data/manifests/wsts_event_metadata.csv` — event metadata (607 records).
- `data/manifests/wsts_sample_manifest.csv` — one verified sample record.

The `wsts_sample_manifest.csv` file is referenced in the handoff PDF but was not independently opened or inspected in this review. The one sample example recorded above is therefore attributed to the PDF's report, not to a direct Role 4 inspection of the CSV.

Still required from Role 1:
- exact pinned WSTS commit used by FireGuard;
- exact loader file/function and local patches;
- canonical 23-channel names and order as exposed by the pinned loader;
- sample input/target temporal semantics from the pinned FireGuard path;
- event-level date continuity / duplicate-date audit (Section 9 provides an independent Role 4 archive-level check; reconciliation with Role 1's own validation remains pending);
- daily raster grid consistency results if required for spatial grouping;
- clarification of geometry available (bbox, polygon/perimeter, or both), coordinate convention, and grid validation.

---

## 7. Archive and storage investigation record

This section records operational investigation so the storage work is not repeated unnecessarily. It is not a scientific validation result and does not change project data requirements.

### 7.1 WSTS archive

The WSL raw-data directory was observed to contain:
- `WildfireSpreadTS.zip` (displayed as 46 GB by `ls -lah`);
- several `wget-log` files;
- no extracted `.tif` or `.hdf5` paths were returned by the contemporaneous `find` command.

The ZIP listing showed top-level year folders: `2018`, `2019`, `2020`, `2021`.

The archive listing showed dated TIFFs nested under event folders. The displayed order of entries within an event was not chronological; this is consistent with the need for the loader to sort filenames before positional sample construction.

Previously recorded archive metadata:
- expected byte size: 48,359,369,821;
- previously verified MD5: `dc1a04e63ccc70037b277d585b8fe761`.

The MD5 result was verified in an earlier session. It was not rerun in the latest storage-check sequence, so if current integrity proof is required, rerun the checksum and save the output.

### 7.2 Storage observations

A Windows PowerShell `Get-PSDrive -PSProvider FileSystem` output reported:

| Drive | Free space at time of check |
|---|---:|
| C: | 13.32 GB |
| D: | 0.50 GB |
| E: | 17.54 GB |
| F: | 414.89 GB |
| G: | 373.67 GB |

WSL `df -h ~` reported `/dev/sdd` with:
- filesystem size 1007 GB;
- used 93 GB;
- available 863 GB;
- usage 10%.

**Important interpretation:** WSL's virtual filesystem free-space report is not proof that the physical Windows drive containing the virtual disk has the same amount of free physical capacity. The physical location of the WSL virtual disk had not yet been established by the attempted `ext4.vhdx` searches. Do not assume the virtual capacity is separate physical storage.

### 7.3 Extraction decision

No full extraction is required merely to finish the paper/code source-review record. Role 1 has already reported the expected/found TIFF inventory count. Dataset-dependent checks (especially per-event date continuity and daily-grid consistency) still need authoritative evidence, but the appropriate next step is to request the targeted audit from Role 1 or perform selective inspection only if required. Sections 9 and 10 provide targeted, no-extraction verifications that do not require full unpacking.

Do not re-download the archive. Do not extract it onto a drive without confirming physical free space and the location of the WSL virtual disk. Do not commit the raw archive, extracted TIFFs, or large HDF5 files to Git.

---

## 8. Role 4 handoffs and dependencies

| Required evidence | Owner | Role 4 use | Current status |
|---|---|---|---|
| Exact pinned WSTS repository commit, loader file/function, and local patch list | Role 1 | Establish the implementation actually used by FireGuard | Pending |
| Canonical channel names and order | Role 1 | Populate the leakage register without guessing | Pending |
| Input sequence and target-day semantics for FireGuard's pinned loader | Role 1 | Confirm forecast origin and target window | Partial: one sample example; archive-level date continuity verified (Section 9) |
| Event-level missing/duplicate-date and daily continuity audit | Role 1 | Verify that next stored image is the next calendar day | Role 4 archive-level check done (Section 9); Role 1 independent validation still pending |
| Per-event daily grid consistency, CRS, dimensions, transform, nodata, quality mask | Role 1 | Validate spatial metadata and grouping inputs | Partial: first TIFF per event; four cross-year sample-level (Section 10) |
| Event footprints/perimeters or agreed geometry source | Role 1 | Apply Plan v4 related-fire grouping rule | Not supplied in the PDF |
| GFS initialization/creation time, valid time, availability and raw direction audit | Role 3 | Level-A/Level-B decisions; downwind method gate | Pending (Role 3 draft audit referenced in `docs/leakage_audit.csv`) |
| GRIDMET time-window, publication latency and wind-direction evidence | Role 3 | Level-A/Level-B decisions | Pending (Role 3 draft audit referenced in `docs/leakage_audit.csv`) |
| VNP13A1 composite window/acquisition support | Role 1 / Role 3 according to source evidence ownership | Decide lag/drop/strict-input eligibility | Pending |
| VNP09GA daily window and availability | Role 1 / Role 3 according to source evidence ownership | Confirm evaluation-only observability semantics | Pending |
| VIIRS HHMM and active-fire mask window | Role 1, with Role 3 temporal coordination | Confirm input-window semantics and safe encoding | Partial public-code behavior reviewed; archive-level date-continuity verified; pinned-schema confirmation pending |
| Static-layer product vintage | Role 1 | Check for post-event/burn-scar leakage | Pending |

Role 4 must not recreate Role 1's canonical data dictionary or claim whole-dataset validation from one sample or first-TIFF metadata. If evidence is unavailable, retain `UNKNOWN`, document the dependency, and exclude unresolved channels from strict-model candidate inputs as required by Plan v4.

---

## 9. Archive date-continuity verification

*Role 4 Phase 1 — Task 2 supporting evidence*

The released WildfireSpreadTS ZIP archive was inspected directly. The filename-based audit produced the following results:

| Check | Result |
|---|---:|
| Event directories identified | 607 |
| TIFF files identified | 13,607 |
| Events with gaps between observed dates | 0 |
| Missing calendar days between each event's first and last observed dates | 0 |
| Events with duplicate dates | 0 |

The check parsed dates from TIFF filenames, grouped files by event, sorted the dates, and checked adjacent date differences.

**Interpretation and limitations**

This supports the conclusion that the archived daily image sequences have consecutive dates between each event's first and last observed dates.

- It supports the calendar-date continuity of the WSTS input/target sequence.
- It does not establish the exact forecast issue time or timezone.
- It does not establish the source products' sensing/composite windows, historical publication times, or operational availability.
- It does not independently establish Level A no-lookahead or Level B operational availability for any input channel.

Accordingly, this result does not change any unresolved `no_lookahead` or `operationally_available` value in `docs/leakage_audit.csv`. Those classifications remain `UNKNOWN` where evidence is absent, and unresolved strict-model candidates remain `EXCLUDE`.

This archive-level verification supplements the loader source-review findings in Section 5. Together they support: (a) the loader selects the next stored image position, and (b) the released archive's stored images are on consecutive calendar dates per event.

---

## 10. Embedded TIFF metadata verification

*Role 4 Phase 1 — Task 2 supporting evidence*

The released WSTS ZIP archive was inspected without extracting the TIFF files. Python's built-in `zipfile` and `struct` modules were used to read TIFF headers and Image File Directory (IFD) entries directly from archived files. No additional TIFF or GDAL packages were required.

### 10.1 Cross-year sample results

One TIFF was inspected from each dataset year (2018–2021).

| Property | Observed result |
|---|---|
| Coordinate reference system | EPSG:32610 — WGS 84 / UTM zone 10N |
| Pixel scale | 375 m × 375 m |
| Pixel interpretation | PixelIsArea |
| Samples per pixel | 23 |
| Bits per sample | 32 for all 23 samples |
| Sample format | IEEE floating point for all 23 samples |
| `GDAL_NODATA` tag | Absent in all four inspected samples |

The `ModelPixelScale` tag was `(375, 375, 0)`. `ModelTiepoint` values differed between samples, as expected for different raster locations.

### 10.2 Embedded channel descriptions

The `GDAL_METADATA` tag of the inspected 2018 TIFF contained 23 sample descriptions. These descriptions correspond to the following channel sequence:
- Samples 0–2: M11, I2, I1.
- Samples 3–4: NDVI_last, EVI2_last.
- Samples 5–11: GRIDMET weather variables.
- Samples 12–14: slope, aspect, elevation.
- Sample 15: PDSI.
- Sample 16: LC_Type1.
- Samples 17–21: GFS forecast variables.
- Sample 22: active fire.

This independently corroborates the embedded channel labels and ordering in the inspected 2018 TIFF. It is not an archive-wide verification of every TIFF's band descriptions.

### 10.3 Interpretation and limitations

- The CRS and 375 m output pixel spacing were consistent across the four inspected yearly samples. These are sample-level findings, not an independent audit of every archived TIFF.
- The 375 m output pixel spacing must not be interpreted as the native resolution of every source product.
- The absence of the `GDAL_NODATA` tag does not establish the actual nodata convention. Nodata semantics remain `UNKNOWN`.
- Source-product resampling methods and quality-mask semantics were not established by this inspection.
- Embedded spatial metadata and channel descriptions do not establish sensing times, composite support, forecast issue times, historical publication times, or operational availability.
- This inspection does not change any Level A or Level B leakage classification or any input decision in `docs/leakage_audit.csv`.

The cross-year sample findings and their limitations have been recorded in the Role 4 leakage register. Archive-wide spatial and channel verification remains subject to the appropriate Role 1 evidence.

---

## 11. Task 1 completion checklist and status

### 11.1 Evidence collected

- [x] Review the WSTS paper and supplementary dataset documentation.
- [x] Record the benchmark target and 24-hour prediction horizon from the benchmark description.
- [x] Record event years, event count, daily-image count, input-channel count, and spatial resolution.
- [x] Record the paper's event-selection process and relevant dataset limitations.
- [x] Review the public baseline repository README and identify loader/DataModule entry points.
- [x] Record the documented dataset-class fix and angle-feature warning.
- [x] Independently clone the public baseline repository and record inspected commit, branch, clone method, and working-tree status.
- [x] Review `utils.py` for normalization and angle-feature helper behavior.
- [x] Review `FireSpreadDataModule.py` for original benchmark year-based split behavior.
- [x] Record the dataset-creation repository's README-level wind-direction correction notice.
- [x] Inspect the relevant `FireSpreadDataset.py` sections for sample selection, input/target construction, TIFF/HDF5 path behavior, NaN handling, HHMM conversion, normalization, feature mapping, and augmentation.
- [x] Record `read_list_of_images()` sorting, event-name derivation, and zero-TIFF warning behavior from the supplied PowerShell output.
- [x] Record `compute_datapoints_per_fire()` per-event sample-count formulas and flag the truncated small-event branch as unresolved.
- [x] Record `find_image_index_from_dataset_index()` negative-index normalization, out-of-range `RuntimeError`, event lookup walk, and return signature from the supplied PowerShell output.
- [x] Inspect the complete `CreateHDF5Dataset.py` conversion logic and the relevant HDF5 generator behavior.
- [x] Record the inspected public config values and correctly distinguish test-position adjustment from input sequence length.
- [x] Record that the inspected public loader selects the next stored image; archive-level date-continuity verified in Section 9.
- [x] Review Role 1's WSTS geospatial handoff PDF.
- [x] Record Role 1's reported 607-event / 13,607-TIFF inventory and its first-TIFF-only metadata scope.
- [x] Record the one sample input-target example and the limitations of that sample.
- [x] Record the supplied event-metadata CSV schema and reported summary checks with scope qualifications.
- [x] Record the archive/storage investigation and the decision not to perform unnecessary full extraction.
- [x] Archive-level date-continuity verification recorded (Section 9).
- [x] Embedded TIFF metadata verification recorded (Section 10).

### 11.2 Pending evidence / closure conditions

- [ ] Obtain Role 1's exact pinned commit, loader entry point, and local patch list; compare them with the independently inspected public revision.
- [ ] Obtain and review Role 1's canonical channel names/order and sample manifest from the pinned loader.
- [ ] Reconcile input sequence, forecast origin, daily frame window, and D+1 target window against the pinned loader and sample metadata.
- [ ] Obtain Role 1's independent date-continuity / duplicate-date audit for comparison with the Role 4 archive check in Section 9.
- [ ] Inspect the complete pinned `compute_datapoints_per_fire()` small-event branch and record the exact handling when `datapoints_in_fire <= 0`.
- [ ] Verify whether FireGuard's loader uses the same per-event sample-count formula after any filtering or grouping.
- [ ] Obtain dataset-wide daily-grid consistency evidence; first-TIFF metadata plus four cross-year samples are not a full archive-wide audit.
- [ ] Obtain Role 3's GFS and GRIDMET temporal/availability and wind-direction evidence for subsequent leakage decisions.
- [ ] Resolve the remaining source-product timing/version and static-layer-vintage dependencies through Roles 1 and 3.

### 11.3 Task 1 status

**Task 1 status: SOURCE REVIEW SUBSTANTIALLY COMPLETED; HANDOFF RECONCILIATION PENDING.**

The paper review and detailed inspection of the independently cloned public baseline source have been recorded. Role 1's geospatial handoff has been reviewed and documented with its scope limitations. Archive-level date-continuity (Section 9) and embedded metadata (Section 10) have been verified. The exact FireGuard-pinned loader revision, canonical schema reconciliation, and full Level A/Level B channel evidence remain pending.

Do not mark Task 1 fully closed until the pinned implementation and the relevant sample/temporal semantics have been reconciled. This is a Task 1 status only; it is not a Role 4 Phase 1 acceptance decision or Global Gate 1 decision.

---

## 12. Boundaries for the remaining Role 4 Phase 1 tasks

The following items are protocol boundaries so that later artifacts remain consistent with Plan v4. Their inclusion does not mean they have already been implemented or passed.

### 12.1 Leakage register (`docs/leakage_audit.csv`)

The register must contain at least:
- `channel_name`
- `source_product`
- `product_version`
- `sensing_or_composite_window`
- `forecast_creation_time`
- `forecast_valid_time`
- `available_at`
- `no_lookahead`
- `operationally_available`
- `evidence`
- `input_decision`
- native resolution, CRS, resampling, nodata, and quality-mask notes.

Use `PASS`, `FAIL`, or `UNKNOWN` independently for Level A and Level B. Do not populate the canonical channel rows by guessing from the public source feature list. Use Role 1's pinned data dictionary and Role 3's temporal evidence. Unknown Level-A candidates are excluded from strict-model inputs until resolved.

The existing `docs/leakage_audit.csv` already includes the required columns plus `raw_channel_index`, `processed_channel_index`, `channel_role`, and `audit_status`. It covers 23 raw channel entries plus derived `binary_active_fire_mask`. Not-yet-complete items are detailed in Section 13.

### 12.2 Mandatory channel checks to coordinate

The project-plan checks include:
- Daily windows and forecast origin; confirm actual sample semantics.
- GFS issue/creation/availability and valid time; do not assume a 00Z run was available at 00Z.
- GFS wind-direction encoding and raw range; determine whether direction is full-circle or quadrant-limited/lossy.
- GRIDMET daily window and publication/access latency.
- VNP13A1 16-day composite window and acquisition support; lag/drop if the composite overlaps the target start.
- VNP09GA daily window and availability; next-day observability is evaluation-only.
- VIIRS active-fire mask and HHMM timing; do not treat HHMM as an ordinary continuous scalar.
- Static-layer product vintage; check that layers do not encode post-event burn scars or later land-cover changes.
- Angle-variable raw ranges; use circular encodings only where the original data preserves full-circle information.
- Preprocessing isolation; fit normalization, class sampling, feature selection, thresholds, and calibration using training/validation only, never the test year.

### 12.3 Strict model and optional benchmark-assumption branch

Plan v4 requires strict M1–M4 to exclude GFS by default. GFS/FiLM may only be treated as a separately labelled benchmark-assumption branch if the relevant Level-A audit passes and provenance is documented. Do not silently merge GFS into strict inputs. Do not enable direction-dependent methods unless Role 3 verifies suitable full-circle wind direction.

The current `docs/leakage_audit.csv` correctly marks all GFS channels as `MODEL_INPUT_EXCLUDED_STRICT` with `input_decision = EXCLUDE`. This aligns with Plan v4.

### 12.4 Observability

The WSTS target does not provide complete cloud observability for next-day negatives. Next-day VNP09GA reflectance validity/QA may be used only for a supplementary evaluation-only observability analysis. It must not enter model input tensors, feature selection, or pre-origin readiness signals. An automated configuration/schema test must prevent this leakage. Do not relabel unobserved cells as true negatives or claim perfect ground truth.

### 12.5 Related-fire grouping and P1/P2 split protocol

The later Role 4 tasks must follow the grouping and split rules in Plan v4 and the role-wise execution guide:
- group related events based on the approved footprint-overlap or distance/time rule;
- inspect single-linkage chaining;
- report group-size distribution and largest-group share by year;
- if any group exceeds 20% of a year's events, inspect/tighten the rule or use a complete-linkage alternative and document the 50 km / 30 day sensitivity check;
- keep mandatory P1 leave-one-year-out separate from P2 geographic holdout;
- select validation only as whole event groups from the remaining training years, targeting 15–20% where feasible and documenting group-count limitations;
- no group/day/tile/pixel from the same group may cross train/validation/test within a fold;
- freeze and hash manifests only after grouping and validation strategy are settled;
- P2 may be explicitly deferred if schedule pressure requires.

Do not create or freeze split manifests from the current bbox candidate-pair counts. First obtain the geometry and grouping evidence required by Plan v4.

### 12.6 Metrics and threshold/calibration protocol

The Role 4 metrics contract must specify:
- AUPRC as primary;
- F1 and IoU as secondary;
- Brier score/Brier skill score and fold-specific calibration;
- pooled and event-macro reporting;
- full-frame evaluation at natural prevalence, excluding padding and invalid pixels;
- paired bootstrap over event/groups, never individual pixels;
- continuous soft-score baselines (distance-to-fire decay and training-fold distance-bin climatology);
- binary persistence/dilated persistence for F1/IoU, not as the only continuous-score AUPRC baseline;
- downwind-weighted baseline only if valid full-circle direction is verified;
- threshold selection, early stopping, and calibration only on that fold's event-level validation groups; never tune on the held-out test year.

These are protocol requirements for the separate `docs/metrics_contract.md` and implementation tests; they are not results.

---

## 13. Immediate next actions (Phase 1 continuation)

The following sequence completes Role 4 Phase 1, in the order required by the role-wise execution guide. Do not skip ahead. Each step ends with a required artifact and evidence.

### 13.1 Finish the leakage audit register (Task 2 population)

**Objective:** Complete `docs/leakage_audit.csv` to the extent permitted by available evidence.

Required steps:
1. Preserve the current CSV structure (already correct: full header plus 23 raw + 1 derived row).
2. For each row, update `no_lookahead` and `operationally_available` only when Role 1 or Role 3 evidence justifies it. Do not promote `UNKNOWN` to `PASS` without named evidence.
3. For channels where Role 1 supplies pinned-channel semantics, update `sensing_or_composite_window`, `forecast_creation_time`, `forecast_valid_time`, and `available_at` accordingly, and add the Role 1 artifact to the `evidence` field.
4. For GFS channels, incorporate Role 3's finalized `docs/gfs_channel_audit.md` (currently referenced as a draft). Update `no_lookahead` only if Role 3 confirms an issue-time audit passes; keep `input_decision = EXCLUDE` and `channel_role = MODEL_INPUT_EXCLUDED_STRICT` unless a benchmark-assumption branch is explicitly approved.
5. Add a second document, `docs/leakage_audit_notes.md`, summarizing:
   - which rows are now `PASS`, `FAIL`, or `UNKNOWN` and why;
   - which channels remain `EXCLUDE` and why;
   - which channels could enter the strict model once Level A passes;
   - the exact reference artifacts used for each change.

**Acceptance criteria:**
- Every row has a non-empty `evidence` cell and one of `PASS`/`FAIL`/`UNKNOWN` in each timing column.
- No `PASS` value appears without an explicit named evidence artifact.
- All strict-model candidates with Level-A `UNKNOWN` remain `EXCLUDE`.
- CSV parses cleanly and matches the header exactly.

### 13.2 Write `tests/test_leakage_audit_schema.py`

**Objective:** Automate the schema check so CI can catch malformed rows.

Steps:
1. Load `docs/leakage_audit.csv` with `csv.DictReader`.
2. Assert the header matches the required columns exactly.
3. Assert every row's `no_lookahead` and `operationally_available` values are in `{"PASS", "FAIL", "UNKNOWN", ""}`.
4. Assert every `input_decision` value is in the allowed set (`STRICT`, `BENCHMARK_ASSUMPTION`, `EXCLUDE`, or a documented equivalent).
5. Run in CI on every PR.

**Acceptance criteria:** the test passes on the current CSV and fails if a malformed value is introduced.

### 13.3 Fire-complex grouping rule and synthetic test (Task 3)

**Objective:** Implement the grouping rule and diagnostic plan, not yet applied to real WSTS events.

Required artifacts:
- `docs/fire_complex_groups.md` documenting the Plan v4 rule exactly (footprint overlap OR centroid ≤30 km AND date gap ≤14 days), the single-linkage chaining risk, the 20% largest-group warning, and the 50 km / 30 day sensitivity check.
- `src/fireguard/data/fire_complex_groups.py` with a pure function that accepts event IDs, geometry (or bbox), and date intervals and returns group IDs.
- `tests/test_fire_complex_groups.py` with synthetic events covering:
  - exact overlap;
  - 29 km / 13-day gap (should group);
  - 31 km / 13-day gap (should not group);
  - 29 km / 15-day gap (should not group);
  - a chain of events that single-linkage would join, to confirm the diagnostic exposes it.

**Acceptance criteria:** all synthetic tests pass; no real WSTS grouping is claimed yet.

### 13.4 P1 LOYO split schema and synthetic split test (Task 4)

**Objective:** Prepare the P1 manifest schema and validation logic, again using synthetic data.

Required artifacts:
- `data/manifests/p1_loyo_fold_schema.csv` (header only): `fold_test_year, split, group_id, event_id, anchor_year`.
- `tests/test_event_splits.py` with synthetic group IDs:
  - test-year groups all in test;
  - no group appears in more than one split within a fold;
  - validation groups come only from training years.
- `docs/validation_protocols.md` updated with a P1 section (can be appended) documenting:
  - test = all groups anchored to the test year;
  - validation = 15–20% of remaining groups (whole groups);
  - train = the rest;
  - cross-year groups anchored to their earliest event start date;
  - P2 documented separately as deferrable.

**Acceptance criteria:** synthetic split test passes; real manifest remains empty with an explicit "blocked on Role 1 event metadata + completed grouping diagnostic" note.

### 13.5 Metrics contract and metrics stubs (Task 5)

**Objective:** Lock definitions and skeleton code so Phase 2 cannot improvise.

Required artifacts:
- `docs/metrics_contract.md` covering AUPRC (primary), F1/IoU, Brier/Brier Skill Score, fold-specific calibration, pooled and event-macro reporting, full-frame natural-prevalence evaluation, and paired bootstrap over event/groups.
- `src/fireguard/evaluation/metrics.py` with function signatures and docstrings:
  - `auprc(y_true, y_score, valid_mask)`
  - `f1_iou(y_true, y_pred, valid_mask)`
  - `brier_score(y_true, y_prob, valid_mask)`
  - `brier_skill_score(y_true, y_prob, y_ref, valid_mask)`
  - `calibration_ece(y_true, y_prob, bins)`
  - `bootstrap_event_metric(metric_fn, per_event_scores, n_boot)`
- `tests/test_metrics_masks.py` with synthetic arrays confirming padded/invalid cells are excluded from every metric.

**Acceptance criteria:** all synthetic metric tests pass; no real dataset performance numbers are reported.

### 13.6 Observability guard (Task 6)

**Objective:** Guarantee next-day VNP09GA cannot enter the model input.

Required artifacts:
- `src/fireguard/data/observability.py` returning a dict tagged `evaluation_only: True`.
- `tests/test_observability_not_input.py` constructing a mock input builder and asserting it rejects the observability key.
- `docs/metrics_contract.md` updated with a one-line statement that observability is post-prediction stratification only.

**Acceptance criteria:** the leakage test fails if observability data is passed to model input assembly.

### 13.7 Role 4 Phase 1 hand-off report

**Objective:** Close Role 4 Phase 1 with the required handoff artifact.

Required artifact: `docs/handoffs/role4_phase1_handoff.md` containing:
- tasks completed with paths and evidence;
- tests run with results;
- dataset and source decisions (verified / assumed / unknown / excluded);
- interface hand-offs and confirmation of receipt;
- open issues and blockers;
- Role-4 Phase-1 acceptance: PASS / FAIL / BLOCKED;
- Global Gate 1 impact: NONE / BLOCKING / CARRY-OVER.

**Acceptance criteria:** the hand-off references every artifact above, and unresolved items are explicitly mapped to the Role 1 and Role 3 dependencies in Section 8.

### 13.8 What must remain blocked

The following cannot be completed by Role 4 alone and must remain `UNKNOWN`/`EXCLUDE` until their owners supply evidence:

| Item | Owner | Why it blocks |
|---|---|---|
| Pinned WSTS loader commit + local patches | Role 1 | Determines whether FireGuard's loader matches the inspected public commit |
| Canonical channel names/order | Role 1 | Required to finalize the leakage register and model input contract |
| GFS issue/valid-time evidence and wind-direction classification | Role 3 | Determines Level-A for GFS and whether downwind baselines are permitted |
| GRIDMET time-window and publication latency | Role 3 | Determines Level-A and Level-B for GRIDMET channels |
| VNP13A1 composite window evidence | Role 1 / Role 3 | Determines Level-A for NDVI/EVI2 |
| VNP09GA observability timing | Role 1 / Role 3 | Determines evaluation-only semantics |
| Static-layer product vintage | Role 1 | Determines leakage risk for slope/aspect/elevation/land-cover |
| Event perimeter/geometry source | Role 1 | Required before running fire-complex grouping on real events |

None of these may be silently assumed. If evidence does not arrive before the Phase 1 gate, the corresponding channels remain `EXCLUDE` and the role-wise execution guide's Gate 1 decision is recorded as CONDITIONAL PASS or FAIL accordingly.

---

## 14. P1 leave-one-year-out protocol (frozen design)

*Added 2026-11-13. Implements execution guide Role 4 Phase 1 tasks 6-8; code in
`ml/src/fireguard_ml/data/event_splits.py`, builder `scripts/build_groups_and_p1_splits.py`,
tests `test_event_splits.py`.*

| Element | Rule |
|---|---|
| Unit of assignment | Fire-complex group (`docs/fire_complex_groups.md`); a group is never split across train, validation and test within a fold |
| Folds | Four: test years 2018, 2019, 2020, 2021 |
| Test | Every group whose anchor year is the test year, with all its events |
| Anchor year | Manifest year of the group's earliest-starting event (cross-year groups stay whole in that fold) |
| Validation | Whole groups from the other three anchor years: round(0.175 x n) groups drawn per year with a fixed seed, so validation spans every training year; target 15-20 % of non-test groups |
| Train | All remaining groups |
| Seed | 0 (string-seeded per test year and anchor year for cross-version reproducibility) |
| Integrity checks | `validate_manifest`: three non-empty splits per fold; each event exactly once per fold; no group in two splits; test = exactly the anchor-year groups; validation and train never from the test year; every event tested in exactly one fold |
| Files | `data/manifests/p1_loyo_fold_manifest.csv` (`fold_test_year, split, group_id, event_id, event_year, anchor_year`), `data/manifests/fire_complex_groups.csv`, `data/manifests/wsts_eval_samples.csv` |
| Hashes | `data/manifests/p1_split_hashes.txt`: SHA-256 of each file with CRLF normalised to LF (the repository's `.gitattributes` does not pin CSVs to LF, so a Windows checkout could otherwise change the bytes). The header line records `status`, grouping input, rule, linkage, validation target and seed |
| Freeze condition | `status=FROZEN` only when real footprints were used and no TIFF was unreadable; otherwise `PROVISIONAL`, and no result may be quoted |

**Provisional fold sizes** (frame-centre grouping, before footprints; groups / events):

| Test year | Train | Validation | Test | Validation share of non-test groups |
|---:|---:|---:|---:|---:|
| 2018 | 293 / 351 | 63 / 80 | 150 / 176 | 17.7 % |
| 2019 | 363 / 441 | 77 / 92 | 66 / 74 | 17.5 % |
| 2020 | 280 / 339 | 60 / 67 | 166 / 201 | 17.6 % |
| 2021 | 315 / 365 | 67 / 86 | 124 / 156 | 17.5 % |

2019 has 74 raw events against 156-201 in the other years, so its test fold has the fewest groups
and the widest confidence intervals. Results must be shown per year with this caveat.

## 15. P2 western-US geographic holdout: deferred

P2 (hold out every group intersecting one western-US ecoregion; separate group-level validation)
is **explicitly deferred**, as Plan v4 permits ("the first major item to drop under schedule
pressure"; it may roll into Phase 2 without blocking Gate 1). Reasons: P1 must be frozen first;
the ecoregion choice and boundary data are not yet agreed; and the grouping code it would reuse is
now in place, so P2 can be added later without changing P1. If it is resumed, it gets its own
manifest and hash file and is never combined with P1 folds.

## 16. Status update, 2026-11-13

### 16.1 Section 8 dependencies

| Required evidence | Status on 2026-11-13 |
|---|---|
| Pinned WSTS commit, loader file, local patches (Role 1) | Still pending; Role 1's material is on their branch and has not been reconciled here |
| Canonical channel names and order (Role 1) | Cross-checked through Role 3's audit (`FirePred.py` export order) and Section 10's embedded metadata; formal Role 1 confirmation pending |
| Event-level date continuity (Role 1) | Role 4 archive check (Section 9) stands; `scripts/build_groups_and_p1_splits.py` re-asserts consecutive dates for every event from Role 1's metadata |
| Daily grid consistency (Role 1) | **Received**: `wsts_event_grid_consistency.csv` / `wsts_raster_grid_audit.csv`. 595 of 607 events consistent; the other 12 are inconsistent only because 20 TIFFs (all on a Google Drive `G:\My Drive` path) could not be opened. All 13,587 readable TIFFs are EPSG:32610, 375 m, 23 float32 bands, no nodata tag |
| Event footprints / geometry (Role 1) | Not supplied (`fire_footprint_proxy_status = NOT_DERIVED`); **Role 4 now derives them** (`scripts/derive_event_footprints.py`) |
| GFS timing and direction (Role 3) | **Received, final** (`docs/gfs_channel_audit.md`, 2026-11-12): direction QUADRANT_LIMITED, precipitation DISABLE; Level A still UNKNOWN pending EE checks 1-2 |
| GRIDMET window and latency (Role 3) | Partly resolved: documented windows recorded; Level B FAIL from publication practice; Level A pending EE check 5 |
| VNP13A1 composite window | **Resolved: Level A FAIL** from the creation code window (see `docs/leakage_audit_notes.md`) |
| VNP09GA daily window | **Resolved for day-D inputs: Level A PASS**; next-day VNP09GA remains evaluation-only |
| Static-layer vintage | **Resolved**: SRTM (2000) PASS; MODIS land cover (own year) FAIL |

### 16.2 Section 13 actions

| Action | Status | Artifact |
|---|---|---|
| 13.1 Finish leakage register | Done (decisions with evidence; review by Roles 1-3 pending) | `docs/leakage_audit.csv`, `docs/leakage_audit_notes.md` |
| 13.2 Register schema test | Done | `test_leakage_audit_schema.py` |
| 13.3 Grouping rule and synthetic tests | Done, and run on real metadata (provisional) | `docs/fire_complex_groups.md`, `fire_complex_groups.py`, `test_fire_complex_groups.py` |
| 13.4 P1 schema and split tests | Done; manifests built and hashed as PROVISIONAL | Section 14, `test_event_splits.py` |
| 13.5 Metrics contract and code | Done, fully implemented (not stubs) | `docs/metrics_contract.md`, `evaluation/metrics.py`, `test_metrics_masks.py` |
| 13.6 Observability guard | Done | `data/observability.py`, `test_observability_not_input.py` |
| 13.7 Phase 1 hand-off | Done | `docs/handoff/role4_phase1_handoff.md` |

Path note: the guide's `src/fireguard/...` and top-level `tests/` become `ml/src/fireguard_ml/...`
and `ml/src/fireguard_ml/tests/` in this repository, because `backend/` already owns the
`fireguard` package name and the ML test suite lives inside the package.

## 17. References

1. Gerard, S., Zhao, Y., & Sullivan, J. (2023). *WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction.* NeurIPS 2023 Datasets and Benchmarks Track. Paper PDF: https://papers.nips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf
2. Gerard, S., Zhao, Y., & Sullivan, J. (2023). *WildfireSpreadTS supplementary material / dataset documentation.* Supplementary PDF: https://proceedings.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Supplemental-Datasets_and_Benchmarks.pdf
3. WildfireSpreadTS dataset. Zenodo record 8006177. https://zenodo.org/records/8006177
4. SebastianGer. WildfireSpreadTS official baseline and loader repository. https://github.com/SebastianGer/WildfireSpreadTS
5. SebastianGer. WildfireSpreadTSCreateDataset official dataset-creation repository. https://github.com/SebastianGer/WildfireSpreadTSCreateDataset
6. Official loader implementation: `FireSpreadDataset.py`. https://github.com/SebastianGer/WildfireSpreadTS/blob/main/src/dataloader/FireSpreadDataset.py
7. Role 1 → Role 4. *WSTS Geospatial Data Handoff*, prepared 5 November 2026. Supplied project handoff PDF; checkpoint `a303778` reported by Role 1.
8. Role 3 → Role 4. `docs/gfs_channel_audit.md` (draft, generated 2026-11-11). Referenced as evidence in `docs/leakage_audit.csv`; finalization pending.
9. Role 4. `docs/leakage_audit.csv`. Current channel-by-channel leakage register; populated to the extent permitted by available Role 1 and Role 3 evidence.
10. Role 4. Sections 9 and 10 of this document. Archive-level date-continuity verification and embedded TIFF metadata verification.
