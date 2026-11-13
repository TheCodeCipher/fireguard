# Role 1 work package: Phase 3 and Phase 4

Same paths as the `fireguard` repository. Copy the contents (except this README) into the repo
root on a Role 1 branch. **Merge order: Role 4, then Role 3, then Role 1** (this code imports
Role 3's sample contract and Role 4's channel register).

## Do this first (one command, 10-20 minutes)

```powershell
cd <fireguard repo>
python scripts/build_phase3_data_release.py --data-root C:\Users\ad\PycharmProjects\WildFireDetection\WildfireSpreadTS\WildfireSpreadTS
```
It reads every TIFF once and writes `data/manifests/wsts_file_checksums.csv`,
`data/manifests/phase3_input_manifest.csv`, `docs/evidence/role1/terrain_qa.csv` and
`docs/evidence/role1/phase3_data_release.json`. It must print `"status": "COMPLETE"` and
`"n_problems": 0`. Commit those four files.

Optional real-data test: `$env:FIREGUARD_WSTS_ROOT="<same folder>"; cd ml; python -m pytest -q`.

## What is here
| File | What |
|---|---|
| `ml/src/fireguard_ml/data/wsts_dataset.py` | The dataset interface (P3-R1-05) |
| `ml/src/fireguard_ml/data/geo_artifacts.py` | Artifact checks and replay catalogue (P4-R1-01/02) |
| `ml/src/fireguard_ml/tests/test_wsts_dataset.py`, `test_geo_artifacts.py` | Integrity tests |
| `configs/data/strict_input_v1.yaml` | Strict input contract |
| `scripts/verify_against_pinned_loader.py` | Reproduction against the pinned loader (already run: MATCH) |
| `scripts/build_phase3_data_release.py`, `scripts/build_replay_catalogue.py` | Release and catalogue |
| `docs/data_dictionary.md` | Updated: new section 12 |
| `docs/hls_historical_coverage_audit.md`, `docs/data_provenance.md` | HLS status (NOT RUN), provenance |
| `docs/handoff/role1_phase3_handoff.md`, `role1_phase4_handoff.md` | Hand-offs |
| `docs/evidence/role1/*.json` | Reproduction and real-data plumbing evidence |
