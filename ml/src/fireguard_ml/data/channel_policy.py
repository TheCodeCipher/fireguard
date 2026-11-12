"""Read ``docs/leakage_audit.csv`` and enforce it on model channel lists.

The register is the single source of truth for which channels may enter the strict models
(Plan v4 4.2). Every training or inference entry point should call ``assert_strict_channels``
with the exact channel names it is about to use.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REGISTER_COLUMNS = (
    "channel_name",
    "source_product",
    "product_version",
    "raw_channel_index",
    "processed_channel_index",
    "channel_role",
    "sensing_or_composite_window",
    "forecast_creation_time",
    "forecast_valid_time",
    "available_at",
    "no_lookahead",
    "operationally_available",
    "evidence",
    "input_decision",
    "native_resolution",
    "crs",
    "resampling",
    "nodata",
    "quality_mask",
    "audit_status",
)
TIME_STATUSES = frozenset({"PASS", "FAIL", "UNKNOWN"})
INPUT_DECISIONS = frozenset({"STRICT", "BENCHMARK_ASSUMPTION", "EXCLUDE"})
FORECAST_MARKERS = ("gfs", "forecast")


@dataclass(frozen=True)
class ChannelRecord:
    name: str
    no_lookahead: str
    operationally_available: str
    input_decision: str
    processed_index: str
    evidence: str


def load_register(path: Path) -> dict[str, ChannelRecord]:
    """Parse the register (BOM-tolerant) and check its header matches the Plan v4 contract."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != REGISTER_COLUMNS:
            raise ValueError(f"register header mismatch: {reader.fieldnames}")
        out: dict[str, ChannelRecord] = {}
        for r in reader:
            if r["channel_name"] in out:
                raise ValueError(f"duplicate channel {r['channel_name']}")
            out[r["channel_name"]] = ChannelRecord(
                r["channel_name"],
                r["no_lookahead"],
                r["operationally_available"],
                r["input_decision"],
                r["processed_channel_index"],
                r["evidence"],
            )
    return out


def strict_channels(register: dict[str, ChannelRecord]) -> list[str]:
    """Channels the register allows into strict M1-M4, in register order."""
    return [n for n, r in register.items() if r.input_decision == "STRICT"]


def policy_violations(register: dict[str, ChannelRecord]) -> list[str]:
    """Rule breaches inside the register itself (empty list = consistent)."""
    problems = []
    for n, r in register.items():
        if r.no_lookahead not in TIME_STATUSES:
            problems.append(f"{n}: no_lookahead={r.no_lookahead!r}")
        if r.operationally_available not in TIME_STATUSES:
            problems.append(f"{n}: operationally_available={r.operationally_available!r}")
        if r.input_decision not in INPUT_DECISIONS:
            problems.append(f"{n}: input_decision={r.input_decision!r}")
        if not r.evidence.strip():
            problems.append(f"{n}: empty evidence")
        if r.input_decision == "STRICT" and r.no_lookahead != "PASS":
            problems.append(f"{n}: STRICT but Level A is {r.no_lookahead}")
        if r.input_decision == "STRICT" and any(m in n.lower() for m in FORECAST_MARKERS):
            problems.append(f"{n}: forecast channel marked STRICT")
    return problems


def assert_strict_channels(channel_names: list[str], register: dict[str, ChannelRecord]) -> None:
    """Raise if any requested channel is not STRICT in the register (unknown names included)."""
    allowed = set(strict_channels(register))
    bad = [c for c in channel_names if c not in allowed]
    if bad:
        raise ValueError(
            f"channels not allowed in a strict model: {bad}. Allowed: {sorted(allowed)}. "
            "Change docs/leakage_audit.csv (with evidence) first, never this call."
        )
