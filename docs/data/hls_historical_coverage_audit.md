# HLS S30 historical coverage audit (Role 1, P3-R1-04)

**Status: NOT RUN. HLS is not a model input.** It stays out of every strict configuration until
this audit is run and every v4 go/no-go threshold is evidenced.

Why not run: the audit needs NASA HLS S30 scene metadata and QA layers for the 607 event AOIs and
2018-2021 forecast origins (NASA CMR/LP DAAC or Earth Engine `NASA/HLS/HLSS30/v002`). That access
was not available in the environment where the Phase 3 data work was done, and no coverage number
has been estimated or assumed. The 2026 sample scene is not a substitute (guide P3-R1-04).

When run, it must produce one row per (event, origin): event ID, origin date, nearest prior clear
scene date, gap in days, clear-land fraction, cloud/aerosol QA summary, projection/grid alignment
status, usable yes/no; and apply the v4 thresholds exactly. Any failed threshold sets HLS to
**FEASIBILITY ONLY** and stops model-input work. Inputs ready for it: event AOIs and origins in
`data/manifests/phase3_input_manifest.csv` (frame bounds from `transform`, `height`, `width`).
