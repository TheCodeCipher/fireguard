# Data provenance and licence register (Role 1, P4-R1-03)

| Asset | Version / identifier | Licence | Checksums | Distributable |
|---|---|---|---|---|
| WildfireSpreadTS dataset (`WildfireSpreadTS.zip`, 13,607 GeoTIFFs, 607 events, 2018-2021) | Zenodo record 8006177, v1.0, published 2023-10-25; archive md5 `dc1a04e63ccc70037b277d585b8fe761` (as listed on the record; Role 1 verified the downloaded archive against the published md5) | CC-BY-4.0 (recorded by Role 1 from the Zenodo record) | per-file sha256: `data/manifests/wsts_file_checksums.csv` | Raw data: no (never committed; size). Derived manifests: yes, with attribution |
| WildfireSpreadTS loader code | `SebastianGer/WildfireSpreadTS` @ `f62dc49f2c2a6d249c071533ae5340c0d01bfb95` + `docs/wsts_baseline_local_patch.diff` | MIT (repository LICENSE) | commit id | Not vendored; referenced |
| WildfireSpreadTS creation code | `SebastianGer/WildfireSpreadTSCreateDataset` @ `f6c5167` | MIT (repository LICENSE) | commit id | Not vendored; referenced |
| FireGuard manifests (eval samples, P1 folds, groups, input manifest) | `data/manifests/p1_split_hashes.txt`, `docs/evidence/role1/phase3_data_release.json` | project licence | sha256 in those files | yes |
| Model artifacts (probability GeoTIFFs) | `runs/<run_id>/model_version.json` (checkpoint sha256, config hash, split hash) | project licence; derived from CC-BY-4.0 data | `oof_forecasts.jsonl` | yes, with WSTS attribution |
| `afall` assets | - | **unresolved** | - | **No.** Not included in any FireGuard artifact or manifest |

Attribution to carry with any distributed derived product: Gerard, Zhao and Sullivan,
"WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction",
NeurIPS 2023 Datasets and Benchmarks; data: Zenodo 10.5281/zenodo.8006177.

Data dictionary version: `docs/data_dictionary.md` section 12 (contract `wsts_strict_input_v1`).
Upstream products inside WSTS (VIIRS VNP09GA and active fire, SRTM, GRIDMET, GFS, MODIS) carry
their own terms; only VIIRS and SRTM-derived channels enter strict FireGuard models.
