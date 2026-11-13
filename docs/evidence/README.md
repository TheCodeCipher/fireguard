# Audit evidence (Role 3)

`outputs/` is git-ignored (`.gitignore` line 66), so the files the Phase 1 acceptance criteria call for are
committed here instead. They are copies; the scripts still write to `outputs/`.

| File | Produced by | Covers |
|---|---|---|
| `gfs_channel_range_summary.csv` | `scripts/audit_gfs_channels.py` | Per-channel and per-year min/max/mean/percentiles/NaN counts over 1,363 sampled GeoTIFFs |
| `gfs_wind_direction_histogram.png` | `scripts/audit_gfs_channels.py` | GFS and GRIDMET wind-direction distributions |
| `gfs_precip_break_check.csv` | `scripts/check_gfs_precip_break.py` | GFS precipitation against observed GRIDMET precipitation, split by year and by the 2019-11-07 rule change |

Read them through `docs/gfs_channel_audit.md`, which states what each one shows.
