# FireGuard — Phase 1 & Phase 2 Role-wise Execution Guide

**Companion document to:** `FireGuard_Final_Project_Plan_v4.md`  
**Document version:** 1.0  
**Prepared:** 31 October 2026  
**Applies to:** Six-person FireGuard team  
**Coverage:** Phase 1 (Weeks 1–2) and Phase 2 (Weeks 3–4) only

> **Authority rule:** `FireGuard_Final_Project_Plan_v4.md` is the source of truth for project scope, architecture, dataset decisions, evaluation protocols, product behaviour and phase gates. This guide translates that plan into role-specific work, deliverables, dependencies and tests. If this guide appears to conflict with v4, follow v4 and raise the conflict before changing implementation.

---

# 0. How every teammate must use this guide with an LLM

Each teammate should upload **both** files to their chosen LLM:

1. `FireGuard_Final_Project_Plan_v4.md`
2. `FireGuard_Phase1_Phase2_Rolewise_Execution_Guide.md`

Then paste the prompt below. The LLM must identify the teammate's role before giving technical instructions. This prevents six people from accidentally implementing the same component or making incompatible design decisions.

## Copy-paste prompt for every teammate

```text
You are assisting one member of the FireGuard project team.

I have uploaded two controlling documents:
1. FireGuard_Final_Project_Plan_v4.md — the locked project plan and source of truth.
2. FireGuard_Phase1_Phase2_Rolewise_Execution_Guide.md — the detailed role-wise task list, deliverables, dependencies and acceptance tests for Phases 1 and 2.

IMPORTANT: Before giving me any project task, code, or implementation instructions, ask me exactly:
“Which FireGuard role are you assigned: Role 1, 2, 3, 4, 5, or 6? Please also provide your name if you want it included in generated files.”

Wait for my answer. Do not assume my role from previous context.

After I select a role:
- Read the corresponding role section in the execution guide and the matching responsibilities in Project Plan v4.
- Work ONLY on my assigned role's tasks for the current phase. Do not implement another member's work or silently take ownership of another role's deliverables.
- You may explain dependencies and request interface details from another role, but identify them as hand-offs, not as tasks I own.
- Start with Phase 1. Do not begin Phase 2 until I confirm that my Phase 1 acceptance tests have passed and the team has passed the global Gate 1.
- Give instructions in the order listed in my role section. Use the specified dataset sources, versions, data semantics, spatial grid, strict no-GFS default and replay-only product boundaries in v4.
- Do not invent channel names, channel order, temporal semantics, data availability, dataset contents, API behaviour, model performance or experiment results. Inspect the actual repository, data, metadata and documentation. Mark unresolved information as UNKNOWN and explain how to verify it.
- Do not change the locked project scope or architecture. Propose a change separately with rationale, affected roles, test impact and approval required.
- Never ask me to download or commit raw WSTS/HLS/DEM data, credentials, FIRMS keys, private user reports or large checkpoints into GitHub.
- For every task, tell me: (1) objective, (2) exact steps, (3) files to create/update, (4) commands/code/tests, (5) expected evidence, and (6) acceptance criteria.
- If a task depends on another role, state the required input, the owner, the expected format and the hand-off test. Do not fabricate that input.
- Maintain a checklist of my tasks. At the end of each task, ask me to share the command output, test result or artifact needed to verify completion.
- At the end of the phase, produce a concise hand-off report listing completed tasks, artifact paths, tests and results, unresolved issues, dependencies and pass/fail status against the phase acceptance criteria.

My role selection is: [WAIT FOR MY ANSWER BEFORE PROCEEDING]
```

## Working rules for all six members

1. **One role, one owner.** Each person owns the deliverables listed under their role. Cross-review is encouraged; duplicate implementation is not.
2. **No silent scope changes.** The strict default models M1–M4 exclude GFS. GFS/FiLM is only a separately labelled benchmark-assumption experiment after the audit. HLS-in-model and GNN are conditional experiments. The v1 forecasting application is historical replay-only.
3. **No unverified claims.** Use `VERIFIED`, `ASSUMED`, `UNKNOWN`, or `LEAKY/EXCLUDED` for data/timing decisions, with evidence.
4. **No raw data in Git.** Commit source code, documentation, small synthetic fixtures, manifests and checksums only. Store downloaded data and large model artifacts outside the repository.
5. **Every task ends with evidence.** A task is not complete because code was written. It is complete when the stated acceptance test passes and the evidence is saved.
6. **Use interfaces agreed by the owners.** Do not guess tensor shapes, API schemas, channel order, coordinate transforms or data formats.
7. **Phase completion is gated.** A person can finish their assigned checklist, but the whole team advances only when the global phase gate passes.
8. **Record negative findings.** If a channel is leaky, HLS coverage is insufficient, an API source is unavailable, or a model fails to beat a baseline, record the result. Do not hide it by changing the test set or metric.

---

# 1. Locked project facts everyone must understand

## 1.1 What FireGuard v1 actually does

FireGuard v1 is a **research prototype and historical decision-support application** with two clearly separated products:

- **Live Monitor:** displays global NASA FIRMS fire observations only. It does not display a FireGuard-generated live forecast.
- **Forecast & Replay Lab:** selects a historical WildfireSpreadTS (WSTS) fire event and forecast origin, retrieves a precomputed **out-of-fold (OOF)** prediction generated by a model that did not train on that event, and allows the user to reveal the next-day target after seeing the forecast.

The model predicts the next-day Suomi-NPP VIIRS active-fire detection map on the WSTS 375 m grid for already-burning large wildfire events. “Newly detected cells” are a secondary analysis. The model does not claim to predict exact physical fire perimeters or new ignitions.

## 1.2 Default architecture and research discipline

- M1: strict 2D U-Net spatial baseline.
- M2: U-Net + WSTS static terrain and MODIS land-cover context.
- M3: stacked-days U-Net temporal comparator.
- M4: U-Net encoder + ConvLSTM temporal model.
- GFS/FiLM: separate benchmark-assumption experiment only after Level-A no-look-ahead and channel checks; not in strict default M1–M4.
- GNN: optional coarsened-graph experiment only if time/compute permits.
- HLS: feasibility study by default; joins the trained model only if the v4 coverage/time/cost rule passes.
- The backend retrieves precomputed OOF artifacts; it does not load PyTorch or run inference.

## 1.3 Core dataset register

| Dataset / source | Owner(s) | Phase 1–2 use | Status in v1 |
|---|---|---|---|
| **WildfireSpreadTS (WSTS)** — Zenodo record 8006177 and official baseline code | Role 1; Roles 2–4 consume documented outputs | Main training/validation/test benchmark; reproduce sample; inspect HDF5/loader; prepare tensors and labels | Mandatory |
| **WSTS S-NPP VIIRS active-fire target (VNP14IMG as described by WSTS)** | Roles 1 and 4 | Primary next-day benchmark target; target semantics and split evaluation | Mandatory |
| **WSTS VIIRS reflectance VNP09GA channels** | Roles 1 and 4 | Input-channel audit; next-day validity may be used for evaluation-only observability, never model input | Mandatory audit; next-day use evaluation-only |
| **WSTS VNP13A1 NDVI/EVI2** | Roles 1 and 4; Role 3 assists timing audit | Inspect 16-day composite window and possible look-ahead/burn-scar contamination | Include only if Level A passes; otherwise lag/exclude |
| **WSTS GRIDMET daily weather/drought/ ERC variables** | Roles 1, 3 and 4 | Input preprocessing and channel timing/latency audit | Strict-model candidate only after Level-A check |
| **WSTS GFS forecast variables** | Role 3; Role 4 records audit; Role 1 provides channel extraction | Raw ranges, forecast run/valid time and direction encoding; optional FiLM test | Excluded from strict default M1–M4; benchmark-assumption branch only |
| **WSTS SRTM topography + MODIS MCD12Q1 land cover** | Role 1; Role 2 uses; Role 4 evaluates | Static channels for M2 terrain + land-cover ablation | Core benchmark-provided static context, subject to channel audit |
| **NASA FIRMS VIIRS_SNPP_NRT / VIIRS_SNPP_SP** | Role 6 + Role 5 | NRT S-NPP map layer; standard-processing S-NPP for historical comparison where appropriate | Mandatory observation layer / comparison source |
| **NASA FIRMS VIIRS_NOAA20 / VIIRS_NOAA21** | Role 6 + Role 5 | Additional labelled satellite context on Live Monitor | Observation-only; never merge invisibly into WSTS target |
| **NASA HLS S30 v2.0** | Role 6 + Role 1 | Historical coverage audit for WSTS events; current 2026 sample supports product EDA only | Conditional; not assumed to be model input |
| **MTBS / USGS Landsat Collection 2 Level-3 Burned Area** | Role 4, with Role 1 data support | Event-level label-quality audit if time/data coverage permits | Optional; not daily labels |
| **LANDFIRE fuel/canopy products** | Role 1 if requested; Role 4 evaluates | Optional pre-event enrichment | Not required; WSTS does not contain this dedicated fuel/canopy layer |
| **ERA5 / ERA5-Land** | Role 3, Role 4 review | Historical/oracle weather analysis only | Not a live/operational forecast input |
| **Google Earth Engine (GEE)** | Roles 1 and 4; Role 3 backup | Inspect/rebuild WSTS source layers and verify GFS/composite timing where needed | Account/access verification in Phase 1 |

### Important dataset ownership boundaries

- **Role 1 owns WSTS loading and the source-channel data dictionary.** Other members consume its documented loader outputs; they should not independently create competing channel-order definitions.
- **Role 3 owns raw GFS channel-range and wind-direction encoding analysis.** Role 4 owns the leakage register and records Role 3's evidence.
- **Role 4 owns splits, validation isolation, metric definitions and the statistical evaluation framework.** Other roles must use the frozen manifests and must not make their own random train/test split.
- **Role 5 owns backend contracts and server-side storage/API implementation.** Role 6 owns frontend behaviour and collaborates on FIRMS source/visual integration.
- **Role 6 owns the FIRMS frontend/ingestion hand-off and HLS historical coverage audit with Role 1.** Role 1 assists with AOI/event metadata and raster/grid alignment.
- **The current 2026 HLS Sample 02 is not a matched WSTS training input.** Its EDA is useful product analysis only. Do not join it to 2018–2021 benchmark events.

## 1.4 Official dataset/source locations

- WSTS dataset: https://zenodo.org/records/8006177
- WSTS benchmark paper: https://papers.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf
- WSTS baseline code: https://github.com/SebastianGer/WildfireSpreadTS
- WSTS dataset-creation code: https://github.com/SebastianGer/WildfireSpreadTSCreateDataset
- NASA FIRMS active fire: https://firms.modaps.eosdis.nasa.gov/active_fire/
- NASA FIRMS Area API: https://firms.modaps.eosdis.nasa.gov/api/area
- NASA FIRMS data availability API: https://firms.modaps.eosdis.nasa.gov/api/data_availability/
- NASA HLS S30 data access: https://hls.gsfc.nasa.gov/data-access-and-tools/
- NASA HLS User Guide: https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf
- NASA HLS known issues: https://lpdaac.usgs.gov/documents/2435/HLS_v2.0_S30_known_issues_April2026.pdf
- NOAA GFS Earth Engine catalogue: https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25
- Google Earth Engine: https://earthengine.google.com/
- MTBS: https://www.mtbs.gov/
- USGS Landsat Collection 2 Level-3 Burned Area: https://www.usgs.gov/landsat-missions/landsat-collection-2-level-3-burned-area-science-product
- LANDFIRE: https://landfire.gov/data
- ERA5: https://www.ecmwf.int/en/forecasts/datasets/era5-hourly-data-single-levels-1940-present

**FIRMS API note:** sources have conflicted on the documented `DAY_RANGE` maximum. The implementation must use chunks of **no more than five days** until the current official NASA API contract is checked at implementation time. Re-check the live documentation then. Keep the MAP_KEY server-side and out of frontend code.

---

# 2. Shared repository and hand-off contract

## 2.1 Repository starts in Phase 1

Role 5 creates the repository in Week 1. All members contribute through feature branches and pull requests.

Expected top-level structure (the locked v4 plan is the full reference):

```text
FireGuard/
├── README.md
├── docs/
│   ├── project_plan_v4.md
│   ├── phase1_phase2_execution_guide.md
│   ├── data_dictionary.md
│   ├── data_sources_and_licenses.md
│   ├── leakage_audit.csv
│   ├── fire_complex_groups.md
│   ├── validation_protocols.md
│   └── api_contract.md
├── configs/
├── data/
│   ├── README.md
│   └── manifests/
├── notebooks/
├── src/fireguard/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── baselines/
│   ├── training/
│   ├── evaluation/
│   ├── replay/
│   └── api/
├── frontend/
├── tests/
├── outputs/                 # small reports/plots only; no raw datasets
└── .github/workflows/ci.yml
```

## 2.2 Shared engineering standards

- Use Python 3.11 for the agreed Python environment unless a tested dependency conflict requires a documented change.
- Use PyTorch for offline model development. Use FastAPI for the backend; React + TypeScript + Vite + MapLibre GL JS for the frontend; PostgreSQL for metadata.
- Keep model training and prediction generation offline. Backend serves static prediction artifacts only.
- Pin package versions in `pyproject.toml` or lock files. No unpinned environment that only works on one member's laptop.
- Use type hints, docstrings and small functions in shared source modules.
- Use deterministic seeds where supported; log unavoidable nondeterminism.
- Every run must save configuration, seed, split manifest hash, code commit, environment, GPU, metrics and checkpoint identity.
- Use `.env` locally and `.env.example` in Git. Never commit keys/passwords.
- CI should run lint/basic import/unit tests and a mocked API test. CI must not download WSTS/HLS or require a GPU.
- Large data/model files stay outside GitHub. Store download instructions, source/version, checksums and manifests in the repo.

## 2.3 Required hand-off contracts

| Producer | Consumer | Hand-off artifact | Required contract |
|---|---|---|---|
| Role 1 | Roles 2 and 3 | WSTS sample/data loader + data dictionary | Explicit channel names/order, dtype, units, shape, time steps, target definition, nodata and valid mask |
| Role 1 | Role 4 | WSTS event/sample metadata + channel evidence | Event IDs, years, event dates, footprint metadata, source products, channel time windows |
| Role 3 | Role 4 | GFS direction/timing audit evidence | Raw value distribution, min/max/percentiles, source code/metadata, Level-A conclusion, direction decision |
| Role 4 | Roles 1–3 | Frozen P1 split manifests and metrics contract | Train/validation/test event-group IDs per fold, natural-prevalence full-frame evaluation, hashes |
| Role 2 | Roles 3 and 4 | U-Net input/output interface + checkpoint/metrics | Expected tensor shape, selected channels, logits shape, mask behaviour, training config |
| Role 4 | Role 2 | Baseline scores and fold/validation protocol | Frozen split IDs, metric implementation, full-frame validation protocol |
| Role 5 | Role 6 | OpenAPI/API contract and mock response | Endpoint paths, request/response schemas, error/status behaviour, pagination/cache semantics |
| Role 6 | Role 5 | FIRMS map requirements | Selected sources/platforms, filters, attribution, freshness/data-age fields |
| Role 6 + Role 1 | Team | HLS historical coverage audit | Event coverage, acquisition-to-origin gap, clear-land share, time/effort estimate, go/no-go |

If an interface is not agreed, create a short contract first. Do not code against a guessed interface.

---

# 3. Phase 1 — Feasibility, data audit and repository foundation (Weeks 1–2)

## 3.1 Phase 1 shared objective

By the end of Phase 1, the team must prove that one WSTS sample can be reproduced, the strict model's input channels are explicitly audited for no look-ahead, the event grouping and P1 validation/test split are frozen, pad/mask handling is tested, a representative U-Net run is timed, and the application repository/API foundations exist.

Phase 1 is **not** a requirement to finish every optional data audit. HLS full historical coverage, MTBS matching and full FIRMS comparison may roll into Phase 2 as defined in v4.

## 3.2 Role 1 — WSTS / Geospatial Data Engineer

### Datasets and resources assigned

**Mandatory:**
- WildfireSpreadTS Zenodo archive (`https://zenodo.org/records/8006177`).
- Official WSTS loader/baseline repository (`https://github.com/SebastianGer/WildfireSpreadTS`).
- WSTS data-creation repository (`https://github.com/SebastianGer/WildfireSpreadTSCreateDataset`).
- The WSTS-provided channels sourced from S-NPP VIIRS active fire, VNP09GA reflectance, VNP13A1 vegetation indices, GRIDMET/weather/drought, GFS forecast features, MODIS MCD12Q1 land cover and SRTM topography. Exact channel names/order must come from the pinned loader, not this summary.

**Support / evaluation-only:**
- VNP09GA next-day validity/QA metadata may be prepared for Role 4's observability diagnostic, but it must not enter model input.
- NASA HLS S30 v2.0 is handled with Role 6 for historical coverage feasibility. The current 2026 sample is product EDA only.

### Phase 1 tasks

1. Read the WSTS paper, Zenodo record, baseline code and data-creation repository. Record URLs and date/version accessed.
2. Clone the WSTS baseline repository and pin the exact commit hash used. Record local modifications separately; do not edit upstream files without documenting the diff.
3. Check storage before downloading. The v4 planning estimate is 150–200 GB free for archive, extraction/HDF5 conversion and intermediates; measure actual disk usage and report it. Do not assume this estimate is exact.
4. Download the WSTS archive from the official record. Verify download completion, archive integrity and available checksum/metadata. Do not commit raw data to GitHub.
5. Inspect repository instructions for archive extraction and HDF5 conversion. Convert only as needed and record conversion command, elapsed time, disk usage and any errors.
6. Load one sample using the official WSTS loader. Reproduce the actual channel schema and sequence semantics. Record:
   - event ID and year;
   - input frame count and order;
   - target frame/day definition;
   - actual tensor shape and dtype;
   - each of the 23 input channel names/order as exposed by the loader;
   - per-channel source, units/range, nodata/validity handling and native resolution where documented;
   - original H/W before padding;
   - active-fire label count / positive-pixel fraction.
7. Confirm that the benchmark target is next-day S-NPP VIIRS active-fire detection. Record that the “newly detected cell” subset is secondary and is not necessarily a new ignition.
8. Build `docs/data_dictionary.md` using the real loader schema. Do not guess a channel index or map a channel based only on a plot.
9. Build a sample-level manifest containing source file, event ID, sample origin, input sequence, target day, shapes, checksum if practical and loader commit hash.
10. Coordinate with Role 4 on event metadata/footprints and grouping. Supply the authoritative event table and clarify coordinate conventions/grid transform.
11. Assist Role 6 with the HLS historical coverage audit if needed: provide event footprints, forecast origins and event years. Do not make HLS join decisions independently.

### Phase 1 artifacts to commit

- `docs/data_dictionary.md`
- `data/manifests/wsts_sample_manifest.csv` (metadata only)
- `data/manifests/wsts_event_metadata.csv` (metadata only; no raw rasters)
- `docs/wsts_reproduction_notes.md`
- `docs/data_sources_and_licenses.md` contribution
- `notebooks/wsts_eda.ipynb` or a reproducible script for sample inspection

### Phase 1 acceptance tests — Role 1

- [ ] A fresh environment can follow the documented WSTS loader steps to load the chosen sample.
- [ ] The sample's actual channel count is verified as 23 where the selected released representation specifies that; any loader/version discrepancy is explicitly reconciled.
- [ ] The input sequence, target day, event ID, year, tensor shape and dtype are printed and saved.
- [ ] The channel list is obtained from the pinned loader and matches the produced tensor channel dimension.
- [ ] Data dictionary contains all model-candidate channels, source product and known time semantics; unresolved entries are `UNKNOWN`, not guessed.
- [ ] Sample manifest records the exact loader commit and data source/version.
- [ ] No raw dataset or large HDF5 file is committed to Git.
- [ ] Role 4 receives the event metadata/footprint hand-off.

### Phase 1 is complete for Role 1 when

All acceptance tests above pass and the hand-off to Roles 2, 3 and 4 is acknowledged. The global Gate 1 still applies.

---

## 3.3 Role 2 — Spatial Deep-Learning Engineer (CNN / U-Net)

### Datasets and resources assigned

- WSTS sample tensors and masks provided by Role 1; do not create an independent loader or change channel order.
- WSTS benchmark target from Role 1.
- Frozen P1 split manifests and metric protocol from Role 4.
- PyTorch, CUDA/GPU, Python 3.11 and project configs.

### Phase 1 tasks

1. Set up the model-development environment with the shared dependency file. Record Python, PyTorch, CUDA and GPU versions.
2. Confirm local GPU availability using `torch.cuda.is_available()` and record GPU model/VRAM. If unavailable, test the agreed Colab/Kaggle/university fallback and note limits.
3. Receive a real WSTS sample from Role 1. Do not choose input channels independently; use the strict no-GFS input candidate after Role 4's Level-A audit.
4. Define and document the proposed model interface:
   - input: batch of dynamic frames/channels in the agreed order;
   - optional static channels in the M2 interface only;
   - valid-pixel/padding mask;
   - output: one logit per target grid cell;
   - loss: BCE + Dice initial comparison, applied only to valid unpadded pixels.
5. Implement and unit-test pad-to-multiple-of-32 and exact unpadding for sample H/W. Record original and padded dimensions. Do not crop full-frame evaluation samples.
6. Implement a small U-Net smoke-test model (not the final tuned model) to verify forward shape and masked loss with synthetic or one small sample batch.
7. Time one representative short U-Net training run as required by Gate 1. Record batch size, number of steps/epochs, model parameter count, data-loader time, training time, peak memory and hardware. The timing must be reproducible enough to estimate the Phase 2 run budget.
8. Set geometric augmentation off by default. Do not add flips/rotations unless all wind/aspect directional channels can be transformed and the transformations are unit-tested.

### Phase 1 artifacts to commit

- `configs/unet.yaml` initial configuration
- `src/fireguard/models/unet.py` smoke-test implementation or model skeleton
- `src/fireguard/data/padding.py` or agreed shared padding utility (coordinate with Role 1)
- `tests/test_padding_masks.py`
- `docs/compute_benchmark.md`
- `docs/model_input_output_contract.md`

### Phase 1 acceptance tests — Role 2

- [ ] Environment and GPU/fallback are documented.
- [ ] U-Net forward pass returns the same spatial dimensions as the padded target.
- [ ] Padded pixels are excluded from masked loss; a unit test proves changing padded target values does not change the loss.
- [ ] Unpadding restores the original H/W exactly.
- [ ] A small real-data or representative smoke training run completes and is timed.
- [ ] Training time, memory, batch size and hardware are saved in `compute_benchmark.md`.
- [ ] Model input channel order is taken from Role 1's data dictionary and Role 4's strict-model decision.

### Phase 1 is complete for Role 2 when

The forward/padding/mask tests pass and the run-time benchmark/fallback decision is documented. Do not begin the full Phase 2 training schedule before Gate 1 passes.

---

## 3.4 Role 3 — Temporal / Weather / GNN Engineer

### Datasets and resources assigned

- WSTS HDF5/channel data and exact loader schema from Role 1.
- WSTS GFS forecast channels — raw range/timing/encoding audit.
- WSTS GRIDMET wind/weather/drought channels — assist with direction/time semantics.
- WSTS data-creation repository and NOAA GFS Earth Engine catalogue.
- GEE access as backup to Roles 1 and 4.

### Phase 1 tasks

1. Inspect the raw HDF5 channel arrays and the official data-creation code. Work from actual channel indices/names supplied by Role 1, not assumed indices.
2. Identify the exact GFS wind-direction channel and compute min, max, percentiles, histogram and representative unique/sample values. Report NaN/fill behaviour.
3. Determine whether the direction is full-circle (0–360 or equivalent) or quadrant-limited (for example near −90 to +90). If quadrant information is lost, do not attempt to recover it by simply applying sine/cosine.
4. Inspect whether raw GFS u/v components exist in the released files or data-creation source. If not, document that direction cannot be reconstructed from the stored angle alone.
5. Trace GFS forecast initialization/creation time and valid time through WSTS creation code and available Earth Engine metadata. Record Level A (no-look-ahead) and Level B (operational availability) separately. Do not assume a 00Z run was available at 00Z.
6. Assist in checking GRIDMET wind-direction range and time window. Send the evidence to Role 4, who owns the leakage register.
7. Document sequence length, daily ordering and expected temporal model interface after Role 1 confirms sample semantics.
8. Do not add GFS to strict M1–M4. Do not enable wind-direction-dependent baseline/FiLM until direction and timing pass the required checks.

### Phase 1 artifacts to commit

- `docs/gfs_channel_audit.md`
- `outputs/gfs_wind_direction_histogram.png` (small plot only)
- `outputs/gfs_channel_range_summary.csv`
- `docs/temporal_input_contract.md`
- Evidence links/screenshots/code references for GFS source timing

### Phase 1 acceptance tests — Role 3

- [ ] GFS wind-direction channel identity is tied to the actual WSTS loader/HDF5 schema.
- [ ] Raw min/max/percentiles and histogram are saved.
- [ ] Direction representation is classified as `FULL_CIRCLE`, `QUADRANT_LIMITED`, `UNKNOWN` or `NOT_PRESENT` with evidence.
- [ ] GFS initialization/creation and valid-time evidence is documented; unknown values remain `UNKNOWN`.
- [ ] The recommendation for downwind baselines and GFS/FiLM is explicit: `ALLOW`, `DISABLE` or `CONDITIONAL`.
- [ ] Role 4 receives the audit evidence before freezing its channel decisions.
- [ ] Strict model config contains no GFS channels.

### Phase 1 is complete for Role 3 when

The GFS direction and timing audit is saved and communicated. A failed/unknown direction check is an acceptable result; silently using an invalid direction is not.

---

## 3.5 Role 4 — Research / Evaluation / Leakage Audit Engineer

### Datasets and resources assigned

- WSTS event metadata, targets, source channels and sample manifests from Role 1.
- WSTS paper, Zenodo and official loader/data-creation code.
- GRIDMET/GFS/VNP13A1/VNP09GA channel semantics and evidence from Roles 1 and 3.
- VNP09GA next-day validity/QA strictly for evaluation-only observability.
- MTBS / Landsat Level-3 Burned Area only as optional event-level label QA; not daily target labels.

### Phase 1 tasks

1. Read the WSTS paper and official code. Record the benchmark target, event years, event counts and limitations from the source.
2. Create `docs/leakage_audit.csv` with at least these fields: channel name; source product/version; sensing/composite window; forecast creation/valid time; `available_at`; `no_lookahead`; `operationally_available`; evidence; input decision; resolution/CRS/nodata/quality notes.
3. Treat Level A and Level B as independent fields. Use `PASS`, `FAIL` or `UNKNOWN`; do not treat unknown as pass. Strict-model candidate channels with Level-A `FAIL` or `UNKNOWN` are excluded until resolved.
4. Coordinate with Role 1 and Role 3 to audit daily window/target semantics, GRIDMET, GFS, VNP13A1 16-day composite windows, VNP09GA daily window, VIIRS HHMM and static-layer vintage.
5. Define a related-fire grouping algorithm based on v4: events are grouped if footprints overlap, or are within the specified distance/time rule. Inspect single-linkage chaining. Report group-size distribution and largest-group share by year. If any group exceeds 20% of a year's events, tighten the rule or use a complete-linkage alternative and document the 50 km / 30 day sensitivity check.
6. Define P1 leave-one-year-out protocol separately from P2 geographic holdout. P1 is mandatory. P2 is separately designed and may be dropped if schedule pressure requires.
7. For each P1 test year Y, select validation **whole event groups** only from the remaining training years. No day, tile or pixel from the same group may cross train/validation/test. Use the v4 target validation proportion (15–20% where feasible) and document group-count limitations.
8. Freeze and hash split manifests only after the grouping and validation strategy are settled. Include train/validation/test group IDs and event IDs for every fold.
9. Define metrics and code contracts: AUPRC primary; F1/IoU secondary; Brier/BSS and fold-specific calibration; pooled and event-macro reporting; paired bootstrap over event/groups, never over pixels.
10. Specify soft baselines: distance-to-fire decay and training-fold distance-bin climatology. Binary persistence/dilation are retained for F1/IoU; binary masks alone are not meaningful continuous score baselines for AUPRC.
11. Define downwind-weighted baseline only if Role 3 verifies valid full-circle wind direction. Otherwise mark disabled.
12. Define the next-day VNP09GA observability analysis as evaluation-only. Add a configuration/schema test that prevents observability inputs from entering model input tensors or feature selection.
13. Define F1 threshold selection, early stopping and calibration to use only the fold's event-level validation groups at natural prevalence/full frames. Never fit any threshold/calibrator on the test year.

### Phase 1 artifacts to commit

- `docs/leakage_audit.csv`
- `docs/validation_protocols.md`
- `docs/fire_complex_groups.md`
- `data/manifests/p1_loyo_fold_manifest.csv`
- `data/manifests/p1_group_validation_manifest.csv` or one combined manifest with explicit split columns
- `data/manifests/p1_split_hashes.txt`
- `docs/metrics_contract.md`
- `tests/test_event_splits.py`
- `tests/test_no_future_data.py`
- `tests/test_observability_not_input.py`

### Phase 1 acceptance tests — Role 4

- [ ] Every strict-model candidate channel has Level-A and Level-B status with evidence; unknown strict channels are excluded.
- [ ] The event grouping report includes method, distance/time thresholds, group-size distribution and largest-group share by year.
- [ ] The >20% largest-group warning rule is evaluated; any tightening/complete-linkage fallback is documented.
- [ ] P1 LOYO splits are event-group safe and validation groups are selected only from training years.
- [ ] P1 manifests are frozen and hashed; no event group appears across train/validation/test within a fold.
- [ ] P2 protocol is documented separately or explicitly deferred.
- [ ] VNP09GA next-day observability is not included in input features; a test prevents this.
- [ ] Metric definitions, full-frame natural-prevalence evaluation and event-level paired bootstrap are documented.
- [ ] Role 1–3 channel/time evidence is reflected accurately in the register.

### Phase 1 is complete for Role 4 when

The leakage register, event grouping, P1 validation/test manifests and metric contract are frozen, hashed and reviewed by Roles 1–3. P2 can be deferred if necessary.

---

## 3.6 Role 5 — Backend / MLOps Engineer

### Datasets and resources assigned

- NASA FIRMS API endpoints and source metadata for observation map.
- WSTS event catalogue metadata from Role 1 / Role 4 for eventual Replay Lab (metadata only in Phase 1–2).
- No raw WSTS tensors or model inference are needed in the backend during these phases.

### Phase 1 tasks

1. Create the GitHub repository in Week 1. Add the v4 project plan and this execution guide to `docs/`.
2. Create branch policy: `main`, `dev`, and feature branches. Require PR review and CI passing before merge.
3. Create README, `.gitignore`, `.env.example`, source/license register, `pyproject.toml` or backend dependency file, and basic documentation.
4. Add GitHub Actions CI for backend lint/import tests and a small mocked API test. CI must not download WSTS or need GPU access.
5. Create FastAPI application with `/api/v1/health` and a mocked FIRMS availability/fires route or response fixture.
6. Draft API contract with Role 6: query params, source/platform labels, acquisition time, FRP/confidence fields, freshness/data-age, cache/error states and pagination if required.
7. Confirm how `MAP_KEY` will be stored as a server-side environment secret. It must never appear in frontend environment variables, JavaScript bundles, committed files or public logs.
8. Inspect the FIRMS Area API and data-availability contract with Role 6. Keep request chunks at no more than five days while official documentation conflict remains unresolved; re-check official docs immediately before implementation.
9. Draft minimal PostgreSQL schema/migrations for `fire_observations`, `wsts_events`, `replay_origins`, `oof_predictions`, `replay_metrics` and `data_assets`. Only build tables needed in the current phase; no user-report tables until that feature is approved.
10. Agree static artifact storage conventions with Role 4: local/shared filesystem for v1 PNG/GeoTIFF/JSON files; no MinIO/TiTiler/Redis unless a demonstrated need appears.
11. Add secret scanning / `.env` ignore checks and ensure no raw datasets are staged.

### Phase 1 artifacts to commit

- Repository skeleton and CI workflow
- `README.md`, `.gitignore`, `.env.example`, `pyproject.toml` or backend dependency lock
- `docs/api_contract.md`
- `docs/data_sources_and_licenses.md` starter
- `src/fireguard/api/main.py` and mocked routes
- `tests/test_api.py`
- `docs/backend_storage_plan.md`

### Phase 1 acceptance tests — Role 5

- [ ] Repository exists and all teammates have the correct access.
- [ ] CI runs successfully on a clean checkout without external datasets or secrets.
- [ ] FastAPI health endpoint returns the documented response.
- [ ] Mock FIRMS endpoint returns stable sample data matching the API contract.
- [ ] MAP_KEY is server-side only; `.env` is ignored and `.env.example` contains names/placeholders only.
- [ ] No raw WSTS/HLS/DEM data, keys or large checkpoints are in Git history.
- [ ] API contract and storage conventions are reviewed by Role 6 and Role 4.

### Phase 1 is complete for Role 5 when

The repo, CI, mocked API, API contract and secret/data handling checks all pass.

---

## 3.7 Role 6 — Frontend / Product / FIRMS / HLS Feasibility Engineer

### Datasets and resources assigned

**Mandatory application observations:**
- NASA FIRMS `VIIRS_SNPP_NRT` for the labelled current S-NPP observation layer.
- NASA FIRMS `VIIRS_SNPP_SP` for standard-processing historical S-NPP observations where appropriate.
- `VIIRS_NOAA20` and `VIIRS_NOAA21` as separate labelled context layers; do not merge these with S-NPP model labels.
- FIRMS data availability endpoint to inspect product/platform availability.

**Conditional HLS audit:**
- NASA HLS S30 v2.0 historical products for WSTS events dated 2018–2021.
- Current 2026 HLS Sample 02 only as the team's product EDA example, not as a WSTS model-training input.

### Phase 1 tasks

1. Work with Role 5 to inspect the current FIRMS API request/response and platform/source identifiers. Agree the API contract before building the UI.
2. Build a small FIRMS client or ingestion component that calls the backend proxy, not the browser directly with a secret key. Role 5 owns backend routing/cache; Role 6 owns client/source-selection and UI integration.
3. Separate S-NPP NRT, S-NPP standard processing, NOAA-20 and NOAA-21 in source labels and filters. The map must not make these look like a single homogeneous model target.
4. Check the FIRMS data-availability endpoint before requesting large ranges. Keep client queries to no more than five-day chunks until the official contract is verified.
5. Draft frontend wireframes for Overview, Live Monitor and Forecast & Replay Lab. In Phase 1 only the Live Monitor/wireframes need to work; replay uses mock artifacts later.
6. With Role 1, run the HLS historical coverage audit over the WSTS event list/footprints and forecast origins:
   - search S30 scenes for 2018–2021 event AOIs;
   - measure whether a scene exists within 14 days before forecast origin;
   - estimate clear-land/valid coverage over the event AOI using Fmask;
   - report event coverage and percentage meeting the 50% clear-land threshold;
   - estimate person-week effort for retrieval, cloud mask, projection/alignment and features.
7. Apply v4 HLS go/no-go rule: HLS is included in the model only if ≥70% candidate events have an S30 scene within 14 days before origin, ≥50% clear land over the event AOI, and pipeline effort is roughly one person-week or less. Otherwise HLS remains an EDA/feasibility report only.
8. Confirm your current HLS Sample 02's geography/tile so the team can assess whether it resembles the WSTS western-US fire region. Do not generalize one tile's statistics.
9. Do not start the optional public report-upload feature in Phase 1. Do not create forecast overlays from FIRMS observations.

### Phase 1 artifacts to commit

- `docs/firms_source_mapping.md`
- `docs/hls_historical_coverage_audit.md` and coverage CSV (may roll into Phase 2 if not finished)
- Frontend wireframes/screenshots
- `frontend/` skeleton with route/navigation structure if practical
- FIRMS sample response fixtures (small and non-sensitive)

### Phase 1 acceptance tests — Role 6

- [ ] FIRMS platform/source identifiers are understood and separately labelled.
- [ ] The browser calls Role 5's backend; no FIRMS key is shipped to frontend.
- [ ] Data availability and ≤5-day request chunking are represented in the client contract.
- [ ] Wireframes reflect the v4 observation-only Live Monitor and historical replay distinction.
- [ ] HLS audit is complete or explicitly marked as a non-blocking Phase 2 carry-over with a named remaining task.
- [ ] The current 2026 HLS sample is not presented as matched WSTS training data.
- [ ] Role 5 signs off the FIRMS/API hand-off.

### Phase 1 is complete for Role 6 when

The FIRMS source mapping/API hand-off and wireframes are ready, and HLS coverage audit is complete or formally carried into Phase 2 with no false claim that it is already feasible.

---

## 3.8 Global Gate 1 — team-wide decision to enter Phase 2

**Gate 1 deadline:** end of Week 2. The following five items are the minimum blocking requirements from v4.

| Gate item | Owner(s) | Required evidence | Pass condition |
|---|---|---|---|
| G1-A: WSTS reproduction | Role 1; reviewed by Roles 2–4 | Sample output, channel schema, target/time semantics, commit hash | One WSTS sample loads end-to-end using pinned loader; shape, 23-channel representation (or discrepancy reconciliation), sequence and target are documented |
| G1-B: strict-model leakage register | Role 4; evidence from Roles 1 and 3 | `leakage_audit.csv` | Every proposed strict input has Level-A and Level-B statuses; unknown/failed Level-A channels are excluded |
| G1-C: GFS direction audit | Role 3; reviewed by Role 4 | Raw range/percentiles/histogram and code/source trace | Direction representation is classified; direction-dependent baseline remains disabled unless full-circle integrity is demonstrated |
| G1-D: grouping and P1 manifests | Role 4; event data from Role 1 | Group diagnostics, frozen/hashes manifests | Group-size distribution and max yearly share recorded; P1 LOYO with group-safe validation/test split frozen and hashed |
| G1-E: padding + timed run + compute | Role 2; support from Roles 1 and 5 | Unit-test results and compute report | Pad-to-32/unpad and valid-mask test pass; one representative U-Net run timed; GPU or fallback route confirmed |

### Gate 1 items that may roll into Phase 2

These do not block Phase 2 if explicitly assigned and tracked:

- Full HLS historical coverage/go-no-go audit.
- MTBS/Landsat event-level matching.
- FIRMS-versus-WSTS external comparison.
- Final upstream licence/source-condition review (but no redistribution until complete).
- P2 geographic holdout if it threatens P1 completion.
- Optional reports page or external data enrichment.

**Gate 1 decision:**
- `PASS`: all five minimum requirements pass; Phase 2 may begin.
- `CONDITIONAL PASS`: only a non-blocking item remains, with named owner and Phase 2 due date.
- `FAIL`: any minimum requirement missing/failed; do not scale model training. Fix the blocking item first.

The team lead records the decision, date, evidence links and open carry-over tasks in `docs/phase_gate_log.md`.

---

# 4. Phase 2 — Soft baselines and spatial CNN (Weeks 3–4)

## 4.1 Phase 2 shared objective

Demonstrate a credible benchmark comparison between soft/non-learning baselines and a strict no-GFS 2D U-Net. Phase 2 is not the time to add every possible architecture. The goal is to establish whether the spatial CNN is useful under the frozen P1 evaluation protocol before expanding to temporal or optional branches.

## 4.2 Role 1 — WSTS / Geospatial Data Engineer

### Phase 2 tasks

1. Turn the Phase 1 sample reproduction into a reusable, documented WSTS loader for the approved split manifests.
2. Implement deterministic sample lookup by event ID, origin/day and split; make no random split inside the loader.
3. Return a documented sample object containing (as applicable): dynamic inputs, static inputs, target, valid-pixel mask, original H/W, padded H/W, event ID, year, origin and target-window metadata.
4. Implement padding to multiples of 32 and preserve original dimensions for unpadding. Coordinate with Role 2 so only one shared implementation owns padding.
5. Implement nodata/validity handling exactly as defined by the WSTS loader/source. Do not replace missing values with zero without a mask.
6. Ensure training-only positive/fire-centred crop sampling can be requested by the training pipeline. The validation/test data loader must return full event frames at natural prevalence.
7. Keep next-day VNP09GA observability in a separate evaluation-only loader/namespace. Add a guard so the training input builder cannot access it.
8. Add data loading tests for channel order, target alignment, no sample leakage across manifests, pad/unpad geometry and deterministic retrieval.
9. Finish the HLS go/no-go audit with Role 6 if it rolled over. If criteria fail or coverage is not proven, write “HLS not included in v1 model” and do not build a complex join.
10. Prepare the WSTS event metadata needed for Role 5's event catalogue, without exposing test targets before reveal.

### Phase 2 artifacts

- `src/fireguard/data/wsts_loader.py`
- `src/fireguard/data/quality_masks.py` / valid mask utilities
- `src/fireguard/data/padding.py` (or agreed shared implementation)
- `src/fireguard/data/observability.py` (evaluation-only path)
- `tests/test_channel_schema.py`
- `tests/test_padding_masks.py`
- `tests/test_observability_not_input.py`
- Updated `docs/data_dictionary.md` and loader usage documentation

### Phase 2 acceptance tests — Role 1

- [ ] Loader retrieves samples deterministically by event/origin and uses only frozen manifests.
- [ ] Input channels match the pinned dictionary and strict-model channel list.
- [ ] Target is benchmark-aligned next-day S-NPP active-fire mask; new-cell target is secondary only.
- [ ] Pad/unpad preserves original spatial extent and padding mask is correct.
- [ ] Full-frame validation/test loader preserves natural class prevalence; no fire-centred crops are used for evaluation.
- [ ] Evaluation-only observability cannot enter model inputs.
- [ ] Tests catch channel-order, target-shift and mask/padding errors.
- [ ] HLS decision is explicit: passed go/no-go with evidence or not included.

### Phase 2 is complete for Role 1 when

The reusable loader and its tests pass, and Roles 2–4 can reproduce the same batch from a manifest ID.

---

## 4.3 Role 2 — Spatial Deep-Learning Engineer (CNN / U-Net)

### Phase 2 tasks

1. Implement the M1 2D U-Net described in v4. Input channel selection must come from the strict no-GFS channel list approved after Gate 1.
2. Use the agreed input/output contract and produce one logit per grid cell. Do not use next-day VNP09GA or target-derived data.
3. Implement BCE + Dice loss initially. The loss must use the valid-pixel mask and exclude padded/invalid cells. Any focal-loss experiment is optional and should not replace the initial baseline without a controlled comparison.
4. Use positive/fire-centred crops only for training optimization if needed. Validation uses full event frames at natural prevalence. Test frames are never cropped for evaluation.
5. Train M1 on the P1 training groups, use only the fold's validation groups for early stopping, and select the checkpoint by full-frame validation AUPRC.
6. Do not use test-year metrics to tune hyperparameters, choose checkpoints, set thresholds or calibrate probabilities.
7. Save prediction logits/probabilities and per-event results for the validation/test sets as required by Role 4. Test predictions must remain clearly labelled and must not be used for further tuning.
8. Log training/validation loss and validation AUPRC per epoch; record optimizer, learning rate, scheduler, batch size, epochs, crop strategy, class sampling, seed, model parameters, hardware, runtime and config hash.
9. Compare at least one sensible loss/sampling configuration only if the experiment budget permits. Keep runs within the timed budget established in Phase 1.
10. Export a stable model interface: input schema version, checkpoint, expected tensor shape, output logits shape and preprocessing version.

### Phase 2 artifacts

- `src/fireguard/models/unet.py`
- `src/fireguard/training/losses.py`
- `src/fireguard/training/train_unet.py` or shared training entry point
- `configs/unet.yaml`
- `outputs/experiments/m1/` (outside Git for large checkpoints; commit small metrics/config summaries)
- `docs/m1_training_report.md`
- `tests/test_unet_output_shape.py`
- `tests/test_masked_loss.py`

### Phase 2 acceptance tests — Role 2

- [ ] M1 uses the strict approved no-GFS input set.
- [ ] Output logits match target spatial size after unpadding.
- [ ] Padded/invalid cells are excluded from loss and metrics.
- [ ] Full-frame validation is run at natural prevalence.
- [ ] Early stopping/checkpoint selection uses validation full-frame AUPRC only.
- [ ] Train/validation/test event IDs match the frozen Role 4 manifest.
- [ ] Training logs, curves, hyperparameters, hardware, seed and config hash are saved.
- [ ] Model outputs are available to Role 4 in the agreed format.
- [ ] No claim of success is made until M1 is compared against the strongest soft baseline.

### Phase 2 is complete for Role 2 when

M1 training is reproducible, validation/test outputs and metadata are handed to Role 4, and the model interface is documented. Passing the global Gate 2 still depends on comparison with baselines.

---

## 4.4 Role 3 — Temporal / Weather / GNN Engineer

### Phase 2 tasks

1. Finalize temporal frame ordering with Role 1 based on the actual WSTS loader and daily-window definitions.
2. Build the **M3 stacked-days U-Net input/architecture interface** as the non-recurrent temporal comparator. Use exactly the strict no-GFS channels for fair comparison.
3. Build a tested ConvLSTM module/interface for M4: shared per-frame encoder features in, spatial-temporal feature map out. Confirm tensor dimensions with synthetic input and one real sample batch.
4. Do not start large M4 training in Phase 2 unless the M1 baseline is stable and the project lead explicitly approves unused compute. Full temporal experiments belong to Phase 3.
5. Document GFS/FiLM as an isolated optional branch. Do not silently include GFS channels in strict M1/M2/M3/M4 configs.
6. If the GFS timing and wind-direction audit passes Level A, prepare a separate `gfs_benchmark_assumption.yaml` design; Level B must remain separately reported. If audit fails/unknown, disable the branch.
7. Implement downwind-weighted baseline support only if Role 3's direction audit verified a full-circle direction or reconstructable u/v. Otherwise document why it is disabled and do not guess a direction.
8. Do not build the GNN yet unless the model pipeline, baseline and compute budget are stable. GNN is a later optional experiment.
9. Write a short architecture note explaining why ConvLSTM preserves spatial feature maps and how it differs from stacked-days U-Net.

### Phase 2 artifacts

- `src/fireguard/models/temporal_unet.py` or agreed M3 config/module
- `src/fireguard/models/convlstm.py` module skeleton/interface
- `configs/temporal_unet.yaml`
- `docs/temporal_architecture_note.md`
- `tests/test_convlstm_shapes.py`
- Optional GFS branch config only if Level-A audit permits

### Phase 2 acceptance tests — Role 3

- [ ] Temporal frame order matches the approved WSTS loader semantics.
- [ ] M3 receives the same strict channels and data splits as M1.
- [ ] ConvLSTM forward pass works on synthetic and representative real feature-map shapes.
- [ ] ConvLSTM output preserves intended spatial dimensions for the decoder.
- [ ] Strict configs contain no GFS channels.
- [ ] GFS/FiLM remains separate and labelled benchmark-assumption; no Level-B/operational claim unless verified.
- [ ] Downwind baseline is either supported by verified direction data or explicitly disabled.
- [ ] Optional GNN has not displaced the required Phase 2 M1/baseline work.

### Phase 2 is complete for Role 3 when

The stacked-days and ConvLSTM interfaces are tested and documented, and all conditional branches remain gated. Full architecture experiments continue in Phase 3.

---

## 4.5 Role 4 — Research / Evaluation / Explainability Engineer

### Phase 2 tasks

1. Verify that the frozen P1 manifests load exactly and that the event/group/validation split isolation tests pass before any baseline result is computed.
2. Implement binary persistence and binary dilated-persistence baselines. Use these for F1/IoU at a declared threshold, not as the only continuous-score AUPRC baseline.
3. Implement a continuous distance-to-fire decay score. Define the distance transform, scale/decay parameter selection and valid-mask handling. Fit/choose parameters on training/validation only, never test.
4. Implement a training-fold distance-bin climatology `P(next-day active fire | distance-to-current-fire bin)` and output a continuous probability/score map. This provides a soft-score reference and a suitable climatology comparator for Brier Skill Score where defined by the v4 protocol.
5. Implement downwind-weighted distance decay only if Role 3's direction audit permits. If not, mark disabled and explain in results.
6. Implement logistic regression/gradient boosting on neighbourhood/weather features only if time permits after required baselines. This is optional.
7. Implement AUPRC, F1, IoU, Brier score/Brier skill score and reliability functions with valid-pixel masks.
8. Produce both pooled metrics and event-macro metrics. For model comparisons, compute per-event metric differences and paired bootstrap over event groups; never bootstrap individual pixels.
9. Define F1 thresholds using only each fold's validation event groups. Store a separate threshold/calibrator per fold if calibration is performed. Never pool calibrators across folds or tune on the held-out test year.
10. Implement validation result tables and plots for the full-frame, natural-prevalence validation data.
11. Prepare an evaluation-only observability report interface. Next-day VNP09GA/reflectance validity may stratify evaluation; write an automated schema/config test that fails if it appears in model input.
12. Record separate P1 results by test year and note smaller 2019 event count/statistical-power limitation.
13. Before reporting M1 performance, ensure Role 2 outputs correspond to exact fold manifests, target masks and config hash.

### Phase 2 artifacts

- `src/fireguard/baselines/persistence.py`
- `src/fireguard/baselines/distance_decay.py`
- `src/fireguard/evaluation/metrics.py`
- `src/fireguard/evaluation/bootstrap.py`
- `src/fireguard/evaluation/calibration.py` initial fold-specific scaffold
- `configs/evaluation.yaml`
- `docs/baseline_results.md`
- `docs/evaluation_protocol.md`
- `tests/test_event_splits.py`
- `tests/test_metrics_masks.py`
- `tests/test_validation_calibration_isolation.py`

### Phase 2 acceptance tests — Role 4

- [ ] P1 split manifest is used unchanged by every experiment.
- [ ] Binary baselines are evaluated with F1/IoU and soft baselines provide continuous scores for AUPRC/Brier.
- [ ] Distance-bin climatology is fitted using training-fold data only.
- [ ] Validation thresholds/calibrators are fit only on validation groups at natural prevalence.
- [ ] Test year/group never influences threshold, checkpoint, parameter or calibrator selection.
- [ ] Full-frame metrics ignore padding/invalid pixels.
- [ ] Pooled and event-macro metrics are both reported.
- [ ] Bootstrap resampling unit is event/group, not pixel.
- [ ] Observability is evaluation-only and blocked from model inputs.
- [ ] Results include per-year/per-fold tables and a low-power caveat for the smaller 2019 fold.

### Phase 2 is complete for Role 4 when

The baselines, metrics, validation isolation and event-level statistical reporting are ready to compare with M1. The final Gate 2 decision is made by the team using these results.

---

## 4.6 Role 5 — Backend / MLOps Engineer

### Phase 2 tasks

1. Turn the Phase 1 mock API into a stable local backend with versioned `/api/v1` routes.
2. Implement `/api/v1/health` and a FIRMS availability proxy route with proper input validation, timeouts and meaningful errors.
3. Implement FIRMS observation query route using Role 6's agreed source/platform filters. Ensure the key stays server-side. Cache by source, platform, date and spatial query; document cache expiry.
4. Implement a small static WSTS event catalogue endpoint using metadata supplied by Roles 1 and 4. It must not expose target rasters in the initial event/input response.
5. Implement mocked replay-origin listing and static artifact retrieval contract, even if actual OOF artifacts are not available until Phase 4. Use clearly marked mock artifacts only.
6. Implement minimal PostgreSQL migrations or a documented local development schema for observation/event/replay metadata. Do not add PostGIS unless bbox queries genuinely require it.
7. Add API tests for request validation, source separation, cache key separation, API failure behaviour and no-secret exposure.
8. Add a storage interface for static files (PNG/GeoTIFF/JSON); backend returns URLs/metadata but does not compute predictions.
9. Ensure backend dependency tree does not include PyTorch as a runtime requirement. Offline ML packages may live in a separate training environment/package group.
10. Work with Role 6 to integrate the frontend development server and test CORS/API errors.

### Phase 2 artifacts

- `src/fireguard/api/main.py`
- `src/fireguard/api/schemas.py`
- `src/fireguard/api/routes/health.py`
- `src/fireguard/api/routes/firms.py`
- `src/fireguard/api/routes/events.py`
- `src/fireguard/api/routes/replays.py` (mock/static contract)
- Database migrations/schema docs
- `tests/test_api.py`
- `docs/api_contract.md` updated from actual implementation

### Phase 2 acceptance tests — Role 5

- [ ] Health route works on a clean local start.
- [ ] FIRMS API key is never exposed to browser/JS response/config/log.
- [ ] FIRMS satellite/platform/source filters remain distinct.
- [ ] Cache keys include all parameters that affect data; chunks remain ≤5 days until official contract is rechecked.
- [ ] API handles upstream timeout, invalid query and unavailable data gracefully.
- [ ] Event catalogue returns only approved metadata and does not leak next-day target.
- [ ] Replay endpoint serves only static/mock artifacts and has no model inference path.
- [ ] API tests pass on CI.
- [ ] Backend runtime does not need PyTorch or GPU.

### Phase 2 is complete for Role 5 when

The local backend API and its tests are stable, and Role 6 can integrate using the published OpenAPI contract.

---

## 4.7 Role 6 — Frontend / Product / FIRMS / HLS Feasibility Engineer

### Datasets and resources assigned

- NASA FIRMS: `VIIRS_SNPP_NRT`, `VIIRS_SNPP_SP`, `VIIRS_NOAA20`, `VIIRS_NOAA21` (separate labelled sources).
- HLS S30 v2.0 historical scenes for 2018–2021 WSTS event AOIs, jointly with Role 1.
- WSTS event metadata and event grouping/IDs from Roles 1 and 4 for frontend catalogue mocks only.

### Phase 2 tasks

1. Set up React + TypeScript + Vite and the agreed styling/map stack (Tailwind + MapLibre GL JS; charting with Recharts/ECharts only as needed).
2. Build the navigation shell for Overview, Live Monitor, Forecast & Replay Lab and Methodology. In Phase 2 the primary functional page is Live Monitor; Forecast & Replay Lab may use a clearly labelled mock catalogue/forecast asset until Phase 4.
3. Implement the Overview page with project summary, dataset status and links/navigation. Do not show fabricated model-performance values.
4. Implement Live Monitor with a global map, source/platform filters, acquisition-time/data-age display, detection details and refresh/loading/error states.
5. Show S-NPP NRT separately from S-NPP standard processing and NOAA-20/21. Include an explicit source label in marker details or layer legend.
6. Display FIRMS points using coordinates as provided by the API. Do not label the data as FireGuard predictions or infer physical perimeter from point detections.
7. Add filters for date/time window, satellite/platform and confidence/product source where supported by the API contract. Ensure a filter changes the API query and visible markers, not just its visual state.
8. Add observation detail panel: latitude, longitude, acquisition date/time, satellite/platform, confidence, FRP if supplied, processing source and data age.
9. Add a Methodology/Data Sources page explaining that the map is an observation layer and forecasting is historical replay-only in v1.
10. Complete the HLS historical coverage audit if not completed in Phase 1. Apply the strict go/no-go rule from v4. Do not add HLS to M1 without documented pass and team approval.
11. Add responsive loading, no-data, API-error and stale-data states. Show data freshness rather than imply an always-current feed.
12. Prepare Phase 4 Replay Lab interaction design, but do not build the reveal/target path before the backend/API contract and OOF artifact structure are ready.

### Phase 2 artifacts

- `frontend/src/` application and route/navigation shell
- `frontend/src/pages/Overview.tsx`
- `frontend/src/pages/LiveMonitor.tsx`
- Map and FIRMS layer components
- Source/platform filter and observation detail panel
- `docs/frontend_wireframes.md`
- `docs/hls_historical_coverage_audit.md` and CSV (if completed)
- Frontend build/lint tests where configured

### Phase 2 acceptance tests — Role 6

- [ ] Frontend builds from a clean checkout and starts with documented commands.
- [ ] Live Monitor fetches observations through Role 5's backend, not directly using a key.
- [ ] Satellite/platform source labels remain visible and separate.
- [ ] Filters genuinely update API request and map features.
- [ ] Observation detail panel shows only fields supplied by the source; missing FRP/confidence is not invented.
- [ ] Loading, empty, stale and error states are implemented.
- [ ] Overview and Methodology pages do not display fabricated metrics or imply global forecast validation.
- [ ] HLS is marked `IN MODEL`, `FEASIBILITY ONLY` or `NOT INCLUDED` based on documented gate evidence.

### Phase 2 is complete for Role 6 when

The observation-only Live Monitor works end-to-end with the backend and the HLS decision is documented. Forecast & Replay Lab full integration is Phase 4.

---

# 5. Global Gate 2 — team-wide decision to enter Phase 3

**Gate 2 deadline:** end of Week 4.

The purpose of Gate 2 is to decide whether the spatial CNN adds value beyond simple soft baselines. If it does not, the team must diagnose before adding ConvLSTM/GNN/other modules.

## Required Gate 2 evidence

1. Frozen P1 split manifests and event grouping were used without changes.
2. Binary persistence/dilated persistence results are available for F1/IoU.
3. Continuous soft baselines (distance-to-fire decay and distance-bin climatology) are available for AUPRC and Brier/BSS comparison.
4. Downwind-weighted baseline is included only if valid wind direction passed its audit; otherwise the exclusion is documented.
5. M1 U-Net was trained on strict no-GFS inputs using the benchmark target.
6. M1 validation/checkpoint selection used only fold-specific validation groups and full-frame AUPRC.
7. Test data were not used for tuning, threshold selection or calibration.
8. Metrics include pooled and event-macro AUPRC/F1/IoU, and paired bootstrap over events/groups.
9. M1 per-event and fold-level comparison against the strongest soft baseline is documented.
10. Training curves, full hyperparameter config, seeds, hardware, checkpoint and config hash are saved.

## Gate 2 pass condition

**M1 must match or beat the strongest soft baseline on AUPRC under the pre-registered P1 protocol, with per-event differences and bootstrap interval shown.**

If Gate 2 fails:

- Do not proceed to a larger architecture as a way to hide the failure.
- Inspect target alignment, input time windows, channel leakage decisions, observability limits, class imbalance, crop policy, valid masks and padding.
- Re-run only the justified correction and document it.
- A transparent negative result is acceptable; changing the test set/metric after seeing results is not.

## Phase 2 decision record

Create/update `docs/phase_gate_log.md` with:

- Gate 2 status: `PASS / FAIL / CONDITIONAL`;
- date and attendees;
- exact P1 manifest hashes;
- best soft baseline and M1 metric table;
- bootstrap method and confidence interval;
- issues and corrective actions;
- approval to enter Phase 3 or decision to diagnose/reduce scope.

---

# 6. Team-wide weekly check-ins and sign-off format

At the end of every week, each role owner submits a short update using this exact structure:

```text
ROLE: Role X — [Role name]
PHASE/WEEK: Phase 1 or 2 — Week X
STATUS: NOT STARTED / IN PROGRESS / BLOCKED / READY FOR REVIEW / PASSED

COMPLETED TASK IDS:
- [IDs]

ARTIFACTS:
- [repo paths / report paths / run IDs]

TESTS RUN:
- [command/test name]

TEST RESULTS:
- PASS/FAIL + evidence

DATA / ASSUMPTIONS VERIFIED:
- [what was verified and source]

BLOCKERS / DEPENDENCIES:
- [role owner, required input, date needed]

DECISIONS NEEDED:
- [if any]

NEXT TASKS:
- [IDs]
```

No one should mark a task `PASSED` without the evidence and acceptance tests in their role section.

---

# 7. Role-to-role dependency map

| Task | Responsible owner | Required collaborator | Rule |
|---|---|---|---|
| WSTS load and 23-channel schema | Role 1 | Role 2, Role 3, Role 4 | One canonical loader/schema only |
| GFS wind direction and timestamp audit | Role 3 | Role 1 provides channel identity; Role 4 records decision | No downwind use until verified |
| Leakage register and P1 splits | Role 4 | Roles 1 and 3 provide source evidence | All roles consume frozen manifests |
| U-Net input shape and padding | Role 2 | Role 1 loader; Role 4 masks/metrics | Shared padding implementation, no duplicate logic |
| Soft baselines and metric implementation | Role 4 | Role 1 data loader | Parameters fit on train/validation only |
| FIRMS API contract/proxy | Role 5 | Role 6 defines source/platform and UI requirements | Key server-side; API is observation-only |
| HLS historical coverage | Role 6 | Role 1 provides event AOIs; Role 4 provides forecast origins | HLS enters model only on v4 go/no-go pass |
| Frontend Live Monitor | Role 6 | Role 5 API | No client-side FIRMS secret |
| Replay backend and OOF artifact retrieval | Role 5 | Roles 1, 2, 4; Role 6 UI | Full implementation Phase 4; mock contract Phase 2 |

---

# 8. What must NOT happen in Phases 1 and 2

- Do not build/train the GNN before the M1 baseline and main pipeline are stable.
- Do not make GFS the default weather input.
- Do not use post-origin VNP09GA observability as a model input.
- Do not use next-day targets in the input sequence or normalization/feature selection.
- Do not use HLS Sample 02 (2026) as a WSTS 2018–2021 training input.
- Do not assume the HLS sample's vegetation statistics represent all fire-prone regions.
- Do not use ERA5 future/reanalysis values as operational forecast-time inputs.
- Do not merge S-NPP, NOAA-20 and NOAA-21 FIRMS products into one unlabeled target/observation layer.
- Do not use binary persistence as the only AUPRC baseline; create continuous soft-score baselines.
- Do not tune thresholds/calibrators/hyperparameters on test-year or test-event labels.
- Do not randomly split tiles, days or pixels from the same event group across train/validation/test.
- Do not crop validation/test full frames to make metrics easier.
- Do not count padded pixels in losses or metrics.
- Do not apply flips/rotations to directional channels unless vector/aspect transformations are correct and tested.
- Do not serve a forecast for an event the selected model trained on. Replay requires OOF predictions from the fold that held that event group out.
- Do not put PyTorch/GPU inference in the Phase 1–2 backend.
- Do not create public safety alerts, evacuation recommendations or claim globally validated forecast performance.
- Do not upload raw datasets, large HDF5 files, model checkpoints, FIRMS keys, Earthdata credentials or private reports to GitHub.

---

# 9. Definition of “Phase 1 and Phase 2 complete”

## Phase 1 is complete only when

- Every role has submitted its Phase 1 hand-off report and evidence.
- All five global Gate 1 criteria pass.
- Any permitted carry-over item has a named owner, due date and explicit non-blocking status.
- The team lead has recorded the Gate 1 decision in `docs/phase_gate_log.md`.

## Phase 2 is complete only when

- All six role-specific Phase 2 acceptance checklists have been reviewed.
- The WSTS loader supplies reproducible full-frame validation/test samples.
- Soft baselines and strict M1 U-Net results are available under the frozen P1 protocol.
- Fold-specific validation, threshold and calibration isolation is demonstrated.
- Pooled and event-macro metrics plus paired event/group bootstrap are reported.
- The backend, FIRMS Live Monitor and mock replay contracts work end-to-end.
- All ten Gate 2 evidence items are available.
- The team has recorded Gate 2 as `PASS`, `FAIL` or `CONDITIONAL` with a written reason.

**Only after Gate 2 should the team scale into Phase 3 temporal and multimodal experiments.**

---

# Appendix A — Suggested task IDs for issue tracking

Use these IDs in GitHub Issues/Projects so tasks can be assigned and reviewed without ambiguity.

| Task ID | Owner | Phase | Task |
|---|---|---|---|
| P1-R1-01 | Role 1 | 1 | Pin WSTS loader/repository commit and inspect source docs |
| P1-R1-02 | Role 1 | 1 | Download/check archive, record storage and conversion |
| P1-R1-03 | Role 1 | 1 | Reproduce one WSTS sample and target |
| P1-R1-04 | Role 1 | 1 | Build 23-channel data dictionary and sample manifest |
| P1-R2-01 | Role 2 | 1 | Configure PyTorch/GPU or fallback |
| P1-R2-02 | Role 2 | 1 | U-Net forward/padding/mask smoke test |
| P1-R2-03 | Role 2 | 1 | Time representative U-Net run |
| P1-R3-01 | Role 3 | 1 | Inspect GFS raw ranges and direction encoding |
| P1-R3-02 | Role 3 | 1 | Trace GFS issue/valid-time and GRIDMET timing evidence |
| P1-R3-03 | Role 3 | 1 | Document strict temporal input interface |
| P1-R4-01 | Role 4 | 1 | Create two-level leakage audit |
| P1-R4-02 | Role 4 | 1 | Define fire-complex grouping and size diagnostic |
| P1-R4-03 | Role 4 | 1 | Freeze P1 LOYO + group-safe validation manifests |
| P1-R4-04 | Role 4 | 1 | Define metrics/observability/threshold protocol |
| P1-R5-01 | Role 5 | 1 | Create repo, branch policy, CI and secret rules |
| P1-R5-02 | Role 5 | 1 | Create FastAPI mock and API contract |
| P1-R5-03 | Role 5 | 1 | Draft PostgreSQL/storage plan |
| P1-R6-01 | Role 6 | 1 | FIRMS source/platform and availability audit |
| P1-R6-02 | Role 6 | 1 | FIRMS frontend/backend hand-off and wireframes |
| P1-R6-03 | Role 6 | 1 | HLS historical coverage go/no-go with Role 1 |
| P2-R1-01 | Role 1 | 2 | Implement reusable WSTS loader |
| P2-R1-02 | Role 1 | 2 | Add pad/mask/full-frame and evaluation-only observability tests |
| P2-R1-03 | Role 1 | 2 | Finalize HLS go/no-go if carried over |
| P2-R2-01 | Role 2 | 2 | Implement strict M1 U-Net |
| P2-R2-02 | Role 2 | 2 | Implement masked BCE + Dice training |
| P2-R2-03 | Role 2 | 2 | Train M1, save curves/checkpoints/config and outputs |
| P2-R3-01 | Role 3 | 2 | Implement stacked-days U-Net interface |
| P2-R3-02 | Role 3 | 2 | Implement/test ConvLSTM module interface |
| P2-R3-03 | Role 3 | 2 | Gate GFS/downwind branches; document no-GFS default |
| P2-R4-01 | Role 4 | 2 | Implement binary persistence/dilation baselines |
| P2-R4-02 | Role 4 | 2 | Implement continuous distance-decay/climatology baselines |
| P2-R4-03 | Role 4 | 2 | Implement full-frame metrics and event bootstrap |
| P2-R4-04 | Role 4 | 2 | Produce baseline-vs-M1 evaluation report |
| P2-R5-01 | Role 5 | 2 | Implement health/FIRMS/event/replay mock API routes |
| P2-R5-02 | Role 5 | 2 | Add DB metadata schema, caching and API tests |
| P2-R6-01 | Role 6 | 2 | Build React application/navigation shell |
| P2-R6-02 | Role 6 | 2 | Implement observation-only FIRMS Live Monitor |
| P2-R6-03 | Role 6 | 2 | Add Methodology/Data Sources page and HLS decision status |

---

# Appendix B — Required per-role hand-off report template

Every member submits this at the end of Phase 1 and again at the end of Phase 2.

```markdown
# FireGuard Role Hand-off Report

- Member name:
- Assigned role:
- Phase completed:
- Date:
- Branch / PR links:

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|

## 2. Acceptance tests
| Test | Command / procedure | Result | Evidence |
|---|---|---|---|

## 3. Dataset and source decisions
- Verified:
- Assumed:
- Unknown:
- Excluded/leaky:

## 4. Interface hand-offs
- Sent to:
- Artifact/schema:
- Confirmed by receiver:

## 5. Open issues and blockers
| Issue | Impact | Owner | Due date |
|---|---|---|---|

## 6. Phase sign-off
- Role-specific acceptance: PASS / FAIL / BLOCKED
- Global gate impact: NONE / BLOCKING / CARRY-OVER
- Recommended next action:
```

---

**End of Phase 1 & Phase 2 Role-wise Execution Guide.**
