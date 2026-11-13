"""Next-day VNP09GA observability is evaluation-only: the input assembler must refuse it."""

from __future__ import annotations

import numpy as np
import pytest

from fireguard_ml.data.observability import (
    EvaluationOnly,
    ObservabilityLeakError,
    assemble_model_inputs,
    next_day_observability,
)

H, W = 4, 5


def strict_inputs():
    return {"binary_active_fire_mask": np.zeros((H, W)), "elevation": np.ones((H, W))}


def test_strict_inputs_assemble():
    x = assemble_model_inputs(strict_inputs())
    assert x.shape == (2, H, W)
    assert x.dtype == np.float32


def test_observability_object_is_rejected_even_under_an_innocent_name():
    obs = next_day_observability({"fire_1": 0.8})["fire_1"]
    with pytest.raises(ObservabilityLeakError, match="evaluation-only"):
        assemble_model_inputs({**strict_inputs(), "elevation_2": obs})


@pytest.mark.parametrize(
    "name",
    [
        "observability",
        "vnp09ga_next_day_valid",
        "next_day_reflectance",
        "qa_d_plus_1",
        "target_mask",
    ],
)
def test_next_day_or_target_named_channels_are_rejected(name):
    with pytest.raises(ObservabilityLeakError):
        assemble_model_inputs({**strict_inputs(), name: np.zeros((H, W))})


def test_observability_strata_are_tagged_evaluation_only():
    out = next_day_observability({"a": 0.9, "b": 0.1})
    assert all(isinstance(v, EvaluationOnly) and v.evaluation_only for v in out.values())
    assert out["a"].value["stratum"] == "better_observed"
    assert out["b"].value["stratum"] == "poorly_observed"


def test_invalid_fraction_is_rejected():
    with pytest.raises(ValueError):
        next_day_observability({"a": 1.5})
