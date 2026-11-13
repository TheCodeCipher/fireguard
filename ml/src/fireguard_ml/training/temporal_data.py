"""Temporal sample contract for M3/M4 (Role 3, Phase 3).

Role 3 does not load WSTS rasters itself: building deterministic full-frame batches is Role 1's
task (P3-R1-05). This module fixes the *interface* those batches must satisfy, checks every sample
against it, and pads samples for the U-Nets. Any loader that implements ``TemporalSampleSource``
plugs into ``temporal_trainer`` and ``temporal_oof`` unchanged.

Contract (docs/temporal_input_contract.md):

* ``x``: ``[T, C, H, W]`` float32, frames ordered oldest -> newest, channels in exactly the strict
  order of the run config, finite everywhere (the loader decides fill values; invalid cells are
  flagged in ``valid``). Not standardised: normalisation is fold-specific (``normalization.py``).
* ``input_dates``: T consecutive calendar days; ``target_date`` = last input day + 1.
* ``y``: ``[H, W]`` binary next-day active-fire target (loader: NaN -> 0, then ``> 0``).
* ``valid``: ``[H, W]`` bool, True where the cell may enter loss and metrics.
* ``transform`` / ``crs``: the original frame's affine (GDAL order a, b, c, d, e, f) and CRS, for
  georeferenced export. Optional for synthetic data.

``SyntheticTemporalSource`` exists only for tests and dry runs. Everything it produces is labelled
``MOCK`` and must never be reported as a WSTS result.
"""

from __future__ import annotations

import csv
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F

SampleKey = tuple[str, date]  # (event_id, target_date)


class TemporalContractError(ValueError):
    """A sample violates the temporal input contract."""


@dataclass(frozen=True)
class TemporalSample:
    event_id: str
    input_dates: tuple[date, ...]
    target_date: date
    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray
    channels: tuple[str, ...]
    transform: tuple[float, ...] | None = None
    crs: str | None = None
    is_mock: bool = False

    @property
    def key(self) -> SampleKey:
        return (self.event_id, self.target_date)

    @property
    def origin_date(self) -> date:
        """Forecast origin = the last input day D (the target is D+1)."""
        return self.input_dates[-1]

    @property
    def hw(self) -> tuple[int, int]:
        return int(self.x.shape[-2]), int(self.x.shape[-1])


MASK_CHANNELS = frozenset({"binary_active_fire_mask"})


def validate_sample(s: TemporalSample, T: int, channels: Sequence[str]) -> None:
    """Raise ``TemporalContractError`` on any violation of the contract above."""
    if s.x.ndim != 4:
        raise TemporalContractError(f"{s.event_id}: x must be [T,C,H,W], got {s.x.shape}")
    t, c, h, w = s.x.shape
    if t != T:
        raise TemporalContractError(f"{s.event_id}: expected T={T} frames, got {t}")
    if tuple(s.channels) != tuple(channels) or c != len(channels):
        raise TemporalContractError(
            f"{s.event_id}: channel order {s.channels} differs from the config {tuple(channels)}"
        )
    if len(s.input_dates) != T:
        raise TemporalContractError(f"{s.event_id}: {len(s.input_dates)} dates for {T} frames")
    for a, b in zip(s.input_dates, s.input_dates[1:], strict=False):
        if (b - a).days != 1:
            raise TemporalContractError(
                f"{s.event_id}: input days must be consecutive and oldest first; got {a} -> {b}"
            )
    if s.target_date != s.input_dates[-1] + timedelta(days=1):
        raise TemporalContractError(
            f"{s.event_id}: target {s.target_date} is not the day after the last input "
            f"{s.input_dates[-1]}"
        )
    if s.target_date in s.input_dates:
        raise TemporalContractError(f"{s.event_id}: the target day is among the inputs")
    if s.y.shape != (h, w) or s.valid.shape != (h, w):
        raise TemporalContractError(f"{s.event_id}: y/valid must be [{h},{w}]")
    if s.x.dtype != np.float32:
        raise TemporalContractError(f"{s.event_id}: x must be float32, got {s.x.dtype}")
    if np.isinf(s.x).any():
        raise TemporalContractError(f"{s.event_id}: x contains inf")
    # NaN = a missing measurement (as in the WSTS TIFFs). It is allowed in measured channels and
    # imputed by the fold normaliser (standardised mean, as the pinned loader does). Mask channels
    # must be complete.
    for ci, name in enumerate(s.channels):
        if name in MASK_CHANNELS and np.isnan(s.x[:, ci]).any():
            raise TemporalContractError(f"{s.event_id}: mask channel {name} contains NaN")
    if not np.isin(s.y, (0, 1)).all():
        raise TemporalContractError(f"{s.event_id}: y must be binary")
    if s.valid.dtype != np.bool_:
        raise TemporalContractError(f"{s.event_id}: valid must be bool")


@runtime_checkable
class TemporalSampleSource(Protocol):
    """What Role 1's loader must provide for Role 3 (and, with T=1, for Role 2)."""

    channels: tuple[str, ...]
    sequence_length: int
    is_mock: bool

    def keys(self) -> list[SampleKey]:
        """Every (event_id, target_date) the source can produce."""

    def load(self, key: SampleKey) -> TemporalSample:
        """One sample, validated against the contract."""


def scored_keys(samples_csv: Path, events: Iterable[str]) -> list[SampleKey]:
    """Keys from ``data/manifests/wsts_eval_samples.csv`` for the given events, sorted.

    Every model is trained and scored on these rows (first target index 5), so M1, M3, M4 and
    the shorter-context ablation all see identical (event, day) pairs.
    """
    wanted = set(events)
    out = []
    with Path(samples_csv).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["event_id"] in wanted:
                out.append((r["event_id"], date.fromisoformat(r["target_date"])))
    return sorted(out)


# ------------------------------------------------------------------------------- padding / batch


@dataclass
class PaddedBatch:
    x: torch.Tensor  # [B, T, C, Hp, Wp]
    y: torch.Tensor  # [B, 1, Hp, Wp] float 0/1
    valid: torch.Tensor  # [B, 1, Hp, Wp] bool; padding is False
    orig_hw: list[tuple[int, int]]
    keys: list[SampleKey] = field(default_factory=list)


def pad_samples(samples: Sequence[TemporalSample], multiple: int = 32) -> PaddedBatch:
    """Pad bottom/right to a common size that is a multiple of ``multiple``.

    Same convention as Role 2's ``pad_to_multiple``: zeros in the padding, ``valid`` False there,
    original size kept for exact unpadding. Frames are never cropped.
    """
    if not samples:
        raise ValueError("empty batch")
    hmax = max(s.hw[0] for s in samples)
    wmax = max(s.hw[1] for s in samples)
    hp, wp = hmax + (-hmax) % multiple, wmax + (-wmax) % multiple
    xs, ys, ms = [], [], []
    for s in samples:
        h, w = s.hw
        pad = (0, wp - w, 0, hp - h)
        xs.append(F.pad(torch.from_numpy(s.x), pad))
        ys.append(F.pad(torch.from_numpy(s.y.astype(np.float32))[None], pad))
        ms.append(F.pad(torch.from_numpy(s.valid)[None], pad, value=False))
    return PaddedBatch(
        torch.stack(xs),
        torch.stack(ys),
        torch.stack(ms),
        [s.hw for s in samples],
        [s.key for s in samples],
    )


def unpad(t: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    return t[..., : hw[0], : hw[1]]


# ------------------------------------------------------------------------------- synthetic (MOCK)


def _stable_seed(*parts: object) -> int:
    """Process-independent seed (Python's hash() of str is randomised per process)."""
    return zlib.crc32("|".join(map(str, parts)).encode())


class SyntheticTemporalSource:
    """Deterministic synthetic fires for tests and dry runs. ``is_mock`` is always True.

    Each event is a fire front advancing east one cell per day; the last channel is the binary
    fire state and the others are random static fields. Frame sizes vary per event so
    padding is exercised. Nothing here resembles WSTS performance.
    """

    is_mock = True

    def __init__(
        self,
        events: dict[str, tuple[date, int]],
        channels: Sequence[str],
        T: int,
        hw: tuple[int, int] = (40, 52),
        seed: int = 0,
    ) -> None:
        self.events = dict(events)  # event_id -> (first image date, number of images)
        self.channels = tuple(channels)
        self.sequence_length = T
        self.hw = hw
        self.seed = seed

    def keys(self) -> list[SampleKey]:
        out = []
        for ev, (first, n) in sorted(self.events.items()):
            for k in range(self.sequence_length, n):
                out.append((ev, first + timedelta(days=k)))
        return out

    def _frame(self, ev: str, day: int, h: int, w: int) -> np.ndarray:
        rng = np.random.default_rng(_stable_seed(ev, self.seed))
        base_col = int(rng.integers(2, max(3, w // 3)))
        row0 = int(rng.integers(2, max(3, h - 8)))
        fire = np.zeros((h, w), np.float32)
        col = min(w - 1, base_col + day)
        fire[row0 : row0 + 6, max(0, col - 2) : col + 1] = 1.0
        return fire

    def load(self, key: SampleKey) -> TemporalSample:
        ev, target = key
        first, n = self.events[ev]
        k_target = (target - first).days
        if not self.sequence_length <= k_target < n:
            raise KeyError(key)
        extra = sum(map(ord, ev)) % 5
        h, w = self.hw[0] + extra, self.hw[1] + 2 * extra
        rng = np.random.default_rng(_stable_seed(ev, self.seed, "fields"))
        statics = rng.normal(size=(len(self.channels) - 1, h, w)).astype(np.float32)
        days = range(k_target - self.sequence_length, k_target)
        # the fire state goes in the last channel (binary_active_fire_mask in the strict order)
        x = np.stack([np.concatenate([statics, self._frame(ev, d, h, w)[None]]) for d in days])
        y = self._frame(ev, k_target, h, w).astype(np.uint8)
        dates = tuple(first + timedelta(days=d) for d in days)
        s = TemporalSample(
            ev,
            dates,
            target,
            x.astype(np.float32),
            y,
            np.ones((h, w), bool),
            self.channels,
            (375.0, 0.0, 600000.0, 0.0, -375.0, 4200000.0),
            "EPSG:32610",
            is_mock=True,
        )
        validate_sample(s, self.sequence_length, self.channels)
        return s


# ------------------------------------------------------------------------------- source factory

WSTS_SOURCE_BLOCKED = (
    "BLOCKED: Role 1's WSTS dataset interface (fireguard_ml.data.wsts_dataset, P3-R1-05) is not "
    "installed on this branch. Merge Role 1's branch, then pass --data-root <folder holding "
    "2018..2021>."
)


def synthetic_source_for_protocol(
    samples_csv: Path, channels: Sequence[str], T: int, seed: int = 0
) -> SyntheticTemporalSource:
    """MOCK source keyed to the real event ids and target dates in ``wsts_eval_samples.csv``.

    Lets the whole fold loop run against the frozen manifest without WSTS data. Every sample and
    every artifact it produces carries ``is_mock=True``.
    """
    first: dict[str, date] = {}
    last: dict[str, int] = {}
    with Path(samples_csv).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = int(r["target_index"])
            d0 = date.fromisoformat(r["target_date"]) - timedelta(days=k)
            first.setdefault(r["event_id"], d0)
            last[r["event_id"]] = max(last.get(r["event_id"], 0), k)
    events = {ev: (d0, last[ev] + 1) for ev, d0 in first.items()}
    return SyntheticTemporalSource(events, channels, T, seed=seed)


def make_source(
    kind: str,
    *,
    samples_csv: Path,
    channels: Sequence[str],
    T: int,
    data_root: Path | None = None,
):
    if kind == "synthetic":
        return synthetic_source_for_protocol(samples_csv, channels, T)
    if kind == "wsts":
        try:
            from fireguard_ml.data.wsts_dataset import WSTSTemporalSource  # Role 1
        except ImportError:
            raise SystemExit(WSTS_SOURCE_BLOCKED) from None
        if data_root is None:
            raise SystemExit("--data-root is required with --source wsts")
        return WSTSTemporalSource(data_root, channels, T, samples_csv)
    raise ValueError(f"unknown source {kind!r}")
