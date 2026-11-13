# Related-fire ("fire complex") grouping: method

- Owner: Role 4 | Updated: 2026-11-13
- Implements Plan v4 section 6.3 and execution guide Role 4 Phase 1 task 5.
- Code: `ml/src/fireguard_ml/data/fire_complex_groups.py` (pure, unit-tested);
  `scripts/derive_event_footprints.py`; `scripts/build_groups_and_p1_splits.py`.
- Results: `docs/fire_complex_groups_report.md` (generated; re-run, do not hand-edit).

## 1. Why group at all

The 607 WSTS event records are not 607 independent fires. GlobFire can split one fire into
several records, and neighbouring fires in the same week share weather and fuel. If one record of
a fire is in training and another in testing, the test score is inflated. So events are grouped
first, and splits move **whole groups** only.

## 2. The rule (Plan v4, pre-registered)

Two events are related if **either**

1. their active-fire footprints overlap (share at least one 375 m cell), **or**
2. their footprint centroids are within **30 km** and the gap between their date intervals is at
   most **14 days** (both thresholds inclusive).

Groups are the connected components of the "related" graph (single linkage). Sensitivity rule:
**50 km / 30 days**, reported side by side and never used for splits.

## 3. Definitions chosen to make the rule computable

| Item | Definition | Reason |
|---|---|---|
| Footprint | Every 375 m cell whose active-fire band is > 0 on at least one day of the event, then cleaned by rule v2 (section 3a) | Role 1 reports `actual_fire_perimeter_status = UNKNOWN` and `fire_footprint_proxy_status = NOT_DERIVED` for all 607 events; frame bounding boxes are ~90 x 115 km image extents (4,113 bbox pairs overlap), so they cannot stand in for footprints. "> 0" matches the loader's target binarisation |
| Cell id | `row * 100000 + col` with `row = floor(y / 375)`, `col = floor(x / 375)` of the cell centre in EPSG:32610 | Every WSTS grid edge is a multiple of 375 m in EPSG:32610, so ids agree across events and overlap is an exact set intersection |
| Centroid | Mean of the footprint cell centres, converted to lon/lat with pyproj | Distances use the great-circle formula: many events lie far east of UTM zone 10, where projected metres are distorted |
| Event date interval | First to last date with any detected fire; falls back to the image-sequence dates only if an event has no detection | The image sequence includes 4 buffer days either side (WSTS paper), which would make every event look 8 days longer |
| Anchor year (P1) | Manifest year of the group's earliest-starting event | Plan v4 says "year of its earliest event start date". One 2018 event starts on 2017-12-29; using its calendar year would create a 2017 fold that does not exist, so the manifest year is used. Stated as a deviation |
| Group id | `G<anchor_year>_<nnnn>`, numbered by earliest start date, then event id | Deterministic and readable |

## 3a. Footprint cleaning (rule `v2_own_fire_no_persistent`, adopted 2026-11-13)

The first frozen run used every detected cell in the frame as the footprint (rule v1). It
over-grouped: one group held 64 events and the 2018 fold's test split held 399 events. Two
causes, both measured on the full dataset:

1. **Persistent thermal anomalies.** Some cells are "on fire" in events from many different years
   (industrial heat sources, gas flares, volcanic vents). Two unrelated fires whose frames both
   contain such a cell "overlap". 134 cells were seen in events from 3 or more distinct years.
2. **Other fires in the frame.** A WSTS frame is about 90 x 115 km and often contains detections of
   neighbouring, unrelated fires; v1 counted them as part of the event's footprint.

Rule v2, applied before grouping (`fire_complex_groups.persistent_cells`, `own_fire_cells`):

- remove cells detected in events from **3 or more distinct years**;
- split the remaining cells into connected clusters after a 2-cell (750 m) dilation, and keep only
  the clusters within **5 km** of the frame centre (WSTS centres each frame on its fire); if none
  is that close, keep the nearest cluster. Dilation only joins clusters; it never adds cells.

Checks: for 95.9 % of events the frame centre lies within 5 km of a kept cluster; the median event
keeps 84.0 % of its raw cells; 54 cross-year pairs still overlap after cleaning, i.e. real re-burns
of the same ground, which correctly stay grouped. Event date intervals still come from the raw
detections (conservative: intervals can only be longer, which can only add links). The rule
version is written into `p1_split_hashes.txt` (`footprint_rule=`), so the hash identifies it.

Effect on the frozen split: 449 groups (364 singletons, largest 11 events); per-fold test events
2018: 220, 2019: 88, 2020: 177, 2021: 122. **The P1 manifest hash changed** to `34e29090...`; any
artifact tied to an earlier hash is invalid.

## 4. Oversized-group safeguard

For every year: the largest group's events in that year divided by that year's events. If any
share exceeds **20 %** (pre-registered warning threshold), the script switches to **complete
linkage** at the same thresholds: a greedy, deterministic partition in which every pair inside a
group is related (events taken in start-date order; each joins the first group it is related to in
full). The decision is printed and stored in the report. The threshold is a warning, not proof of
error; the report also lists the largest group's members so a person can inspect it.

## 5. Results (FROZEN, rule v2)

From `docs/fire_complex_groups_report.md`: 449 groups (364 singletons), largest 11 events; largest
yearly share 1.7-4.1 %, far below 20 %, so single linkage is kept. Related pairs: 129 by
footprint overlap, 66 by distance/date. Sensitivity rule (50 km / 30 days, reported only): 359
groups, largest 21, largest share 8.1 %.

The provisional frame-centre run (506 groups) and the v1 footprint run are superseded.

## 6. How to freeze (about 15 minutes on a machine with the dataset)

```powershell
python scripts/derive_event_footprints.py --data-root <folder that holds 2018..2021>
python scripts/build_groups_and_p1_splits.py
```

The first command reads only the last band of all 13,607 TIFFs and writes
`data/manifests/wsts_event_footprints.csv` and `wsts_event_fire_cells.csv.gz`. The second writes
groups, P1 manifests, the evaluation-sample list, hashes, and this report, stamped `FROZEN` only
if real footprints were used and no TIFF was unreadable. Role 1's grid audit found 20 unreadable
files, all on a `G:\My Drive` (Google Drive streaming) path, so they are most likely incomplete
cloud copies; the footprint script lists any such file per event.

## 7. Answers to Role 1's hand-off questions (`Role_4_Geospatial_Handoff_WSTS.pdf`, section 6)

| Role 1 question | Role 4 answer |
|---|---|
| Rule for related events | Section 2: footprint overlap, or centroid <= 30 km and date gap <= 14 days; single linkage with the 20 % complete-linkage fallback |
| Spatial split constraints | Whole groups only; no group crosses train/validation/test in any fold (enforced by `validate_manifest`). P2 geographic holdout is deferred (`docs/validation_protocols.md` section 16) |
| Geometry required | Active-fire cell footprints, which Role 4's script derives from the TIFFs; no polygons or bounding boxes needed from Role 1 |
| Coordinate convention | Keep EPSG:32610 for cell ids; convert centroids to lon/lat (EPSG:4326) for distances |
| Temporal constraints | Date gap <= 14 days for the distance arm; none for footprint overlap (a re-burn of the same cells in a later year joins the group and is anchored to the earlier year) |
| Checks for Role 1 | (a) re-copy the 20 TIFFs that failed to open in the grid audit and confirm they read; (b) confirm the per-event grid consistency once they read (the 12 inconsistent events are inconsistent only because of those unreadable files) |
| Hand-off schema | `data/manifests/fire_complex_groups.csv`: `event_id, event_year, group_id, anchor_year, group_size`; `data/manifests/p1_loyo_fold_manifest.csv`: `fold_test_year, split, group_id, event_id, event_year, anchor_year` |
