"""Fold-specific input standardisation (Role 3, P3-R3-01).

Statistics come from the fold's ``train`` groups only, over valid cells of every input frame. The
fit refuses any sample whose event is not in that fold's train split, so validation and test
events can never contribute, even by accident.

Channels listed in ``no_standardize`` pass through unchanged. The default follows the pinned
baseline loader (data dictionary 8): binary masks and angular features are not standardised.
NaN marks a missing measurement: it is ignored by the fit and becomes 0 after standardisation,
i.e. the fold's training mean (the pinned loader's ``nan_to_num`` after standardising). Invalid
(padding) cells are also set to 0.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fireguard_ml.training.temporal_data import SampleKey, TemporalSampleSource

DEFAULT_NO_STANDARDIZE = ("binary_active_fire_mask", "aspect", "aspect_sin", "aspect_cos")
STD_FLOOR = 1e-6


class NormalizationLeakError(RuntimeError):
    """A non-training sample was offered to the normaliser fit."""


@dataclass(frozen=True)
class FoldNormalizer:
    fold: int
    channels: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    standardized: tuple[bool, ...]
    n_valid_cells: int
    split_manifest_hash: str

    def apply(self, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Standardise ``x`` [T, C, H, W]; NaN and invalid cells become 0."""
        if x.shape[1] != len(self.channels):
            raise ValueError(f"x has {x.shape[1]} channels, normaliser {len(self.channels)}")
        mean = np.array(self.mean, np.float32)[None, :, None, None]
        std = np.array(self.std, np.float32)[None, :, None, None]
        on = np.array(self.standardized)[None, :, None, None]
        out = np.where(on, (x - mean) / std, x).astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out[:, :, ~valid] = 0.0
        return out

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> FoldNormalizer:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def fit_normalizer(
    source: TemporalSampleSource,
    keys: Iterable[SampleKey],
    train_events: set[str],
    fold: int,
    split_manifest_hash: str,
    no_standardize: Sequence[str] = DEFAULT_NO_STANDARDIZE,
) -> FoldNormalizer:
    """Streaming mean/std per channel over valid cells of the fold's training samples."""
    keys = list(keys)
    leaked = sorted({ev for ev, _ in keys if ev not in train_events})
    if leaked:
        raise NormalizationLeakError(
            f"fold {fold}: normalisation offered non-training events {leaked[:5]}"
        )
    if not keys:
        raise ValueError("no training samples to fit the normaliser")
    c = len(source.channels)
    s1 = np.zeros(c)
    s2 = np.zeros(c)
    cnt = np.zeros(c)
    for k in keys:
        smp = source.load(k)
        cells = smp.x[:, :, smp.valid].astype(np.float64)  # [T, C, n_valid]
        ok = ~np.isnan(cells)
        z = np.where(ok, cells, 0.0)
        s1 += z.sum(axis=(0, 2))
        s2 += (z**2).sum(axis=(0, 2))
        cnt += ok.sum(axis=(0, 2))
    if not cnt.any():
        raise ValueError("training samples contain no valid cells")
    n = int(cnt.max())
    safe = np.maximum(cnt, 1)
    mean = s1 / safe
    std = np.sqrt(np.maximum(s2 / safe - mean**2, 0.0))
    standardized = tuple(ch not in set(no_standardize) for ch in source.channels)
    return FoldNormalizer(
        fold=fold,
        channels=tuple(source.channels),
        mean=tuple(float(m) if on else 0.0 for m, on in zip(mean, standardized, strict=True)),
        std=tuple(
            max(float(s), STD_FLOOR) if on else 1.0 for s, on in zip(std, standardized, strict=True)
        ),
        standardized=standardized,
        n_valid_cells=int(n),
        split_manifest_hash=split_manifest_hash,
    )
