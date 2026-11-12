"""Downwind-weighted distance decay: DISABLED.

Plan v4 6.1 allows this baseline only if a wind direction passes both the encoding audit and the
time audit. Current status (docs/leakage_audit.csv, 2026-11-13):

* GFS wind direction: QUADRANT_LIMITED ([-90, 90], u/v not exported) -> unusable, final.
* GRIDMET wind direction: full-circle, but Level A UNKNOWN (daily window vs target start not yet
  verified; Role 3 Earth Engine check 5) and the FROM convention is assumed, not verified.

Re-enable only after the register marks a direction channel Level-A PASS with a verified
convention, and record the change in docs/baseline_results.md.
"""

from __future__ import annotations

DISABLED_REASON = (
    "downwind baseline disabled: no wind-direction channel has passed both the encoding audit "
    "and Level-A timing (see docs/leakage_audit.csv and docs/gfs_branch_gating.md)"
)


class BaselineDisabled(RuntimeError):
    pass


def downwind_decay(*_args, **_kwargs):
    raise BaselineDisabled(DISABLED_REASON)
