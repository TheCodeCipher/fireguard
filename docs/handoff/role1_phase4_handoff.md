# FireGuard Role Hand-off Report

- Member name: (fill in)
- Assigned role: Role 1, WSTS / Geospatial Data Engineer
- Phase completed: Phase 4 (tooling DONE; catalogue needs a real OOF run)
- Date: 2026-11-13

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P4-R1-01 | Pre-reveal event/origin catalogue | Builder **done**; real catalogue waits for a real OOF export | `scripts/build_replay_catalogue.py`; `data/geo_artifacts.py::build_catalogue` |
| P4-R1-02 | Geospatial artifact metadata and checks | **Done** | `RASTER_EXPORT_SETTINGS`, `PREVIEW`, `verify_artifact`; `test_geo_artifacts.py` |
| P4-R1-03 | Provenance and licence package | **Done**; per-file checksums pending the release run | `docs/data_provenance.md` |

## 2. Acceptance tests
| Test | Result | Evidence |
|---|---|---|
| Catalogue holds only events/origins with a validated OOF artifact | PASS (non-OOF and misaligned rasters rejected) | `test_catalogue_keeps_only_validated_oof_forecasts_and_no_target_fields` |
| No target raster or value pre-reveal | PASS (field whitelist + forbidden list) | same |
| Exports align with the WSTS frame and CRS | PASS: size, CRS, transform, dtype, range and checksum checked; padded or cropped rasters rejected | `test_misaligned_or_invalid_rasters_are_rejected`; real-data plumbing run: 15/15 forecasts verified |
| Each artifact traceable to manifest, fold, config, checkpoint | PASS via Role 3's `model_version.json` + catalogue `model_version`/`fold`/checksum | export contract |
| Licensing recorded; unresolved assets excluded | PASS (`afall` excluded) | `docs/data_provenance.md` |

## 5. Open issues
| Issue | Owner |
|---|---|
| Real catalogue after Role 3's selected-model export | Role 1 |
| Uncertainty/explanation layers: settings defined, layers NOT IMPLEMENTED | Role 3 / Role 4 |
