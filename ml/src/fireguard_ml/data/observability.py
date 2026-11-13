"""Evaluation-only observability (Plan v4 4.4) and the guard that keeps it out of model inputs.

Next-day (D+1) VNP09GA reflectance validity tells us whether a "no fire" target cell was actually
seen by the satellite. It is information from the target day, so it may *stratify evaluation after
prediction* but must never reach a model input tensor, feature selection, or an input-readiness
signal. Two independent safeguards:

1. Observability values are wrapped in ``EvaluationOnly``; the input assembler rejects that type.
2. The input assembler also rejects any key whose name marks observability or next-day data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

FORBIDDEN_INPUT_MARKERS = (
    "observability",
    "vnp09ga_next",
    "next_day",
    "d_plus_1",
    "dplus1",
    "target",
)


@dataclass(frozen=True)
class EvaluationOnly:
    """A value that may be used to stratify metrics, never to build model inputs."""

    name: str
    value: Any
    evaluation_only: bool = True


def next_day_observability(
    valid_fraction_by_event: Mapping[str, float], threshold: float = 0.5
) -> dict[str, EvaluationOnly]:
    """Tag each event as better- or poorly-observed on D+1 from its valid-reflectance fraction.

    ``valid_fraction_by_event`` is the share of an event's next-day frame with valid VNP09GA
    reflectance/QA, computed offline by Role 1 or Role 3. The output is evaluation-only.
    """
    out = {}
    for ev, frac in valid_fraction_by_event.items():
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"{ev}: valid fraction {frac} outside [0, 1]")
        stratum = "better_observed" if frac >= threshold else "poorly_observed"
        out[ev] = EvaluationOnly(
            "next_day_observability", {"valid_fraction": frac, "stratum": stratum}
        )
    return out


class ObservabilityLeakError(ValueError):
    """Raised when evaluation-only or next-day data is offered as a model input."""


def assemble_model_inputs(channels: Mapping[str, Any]) -> np.ndarray:
    """Stack named input channels into ``[C, H, W]``, refusing anything evaluation-only.

    Every model input builder should pass through this function (or call ``check_input_keys``).
    """
    check_input_keys(channels)
    arrays = [np.asarray(v, dtype=np.float32) for v in channels.values()]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"channels have different shapes: {shapes}")
    return np.stack(arrays, axis=0)


def check_input_keys(channels: Mapping[str, Any]) -> None:
    for name, value in channels.items():
        if isinstance(value, EvaluationOnly) or getattr(value, "evaluation_only", False):
            raise ObservabilityLeakError(f"{name!r} is evaluation-only and cannot be a model input")
        lowered = name.lower()
        hit = [m for m in FORBIDDEN_INPUT_MARKERS if m in lowered]
        if hit:
            raise ObservabilityLeakError(
                f"{name!r} looks like next-day / evaluation-only data (matched {hit})"
            )
