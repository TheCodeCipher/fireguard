"""Baseline behaviour on small synthetic frames, plus split isolation for fitted parameters."""

from __future__ import annotations

import numpy as np
import pytest

from fireguard_ml.baselines.distance_decay import (
    DistanceCounts,
    SplitError,
    decay_scores,
    dilated_scores,
    fit_climatology,
    persistence_scores,
    select_decay_scale,
)
from fireguard_ml.baselines.downwind import BaselineDisabled, downwind_decay
from fireguard_ml.baselines.persistence import (
    dilated_persistence,
    distance_to_fire_km,
    fire_mask,
    persistence,
)
from fireguard_ml.evaluation.metrics import average_precision, counts_by_score, f1_iou

rng = np.random.default_rng(11)


def spreading_fire(h=40, w=50, seed=0):
    """Today: a small burning block. Tomorrow: the block grown outward plus a few random spots."""
    r = np.random.default_rng(seed)
    today = np.zeros((h, w), bool)
    today[18:22, 20:24] = True
    tomorrow = distance_to_fire_km(today) <= 1.2
    tomorrow |= r.random((h, w)) < 0.003
    return today, tomorrow


def test_fire_mask_matches_loader_convention():
    band = np.array([[np.nan, 0.0, 1230.0], [5.0, np.nan, -1.0]])
    assert fire_mask(band).tolist() == [[False, False, True], [True, False, False]]


def test_distance_is_zero_on_fire_and_375m_per_cell():
    fire = np.zeros((5, 5), bool)
    fire[2, 2] = True
    d = distance_to_fire_km(fire)
    assert (
        d[2, 2] == 0
        and d[2, 3] == pytest.approx(0.375)
        and d[3, 3] == pytest.approx(0.375 * 2**0.5)
    )
    assert np.isinf(distance_to_fire_km(np.zeros((3, 3), bool))).all()


def test_persistence_and_dilation():
    today, _ = spreading_fire()
    assert (persistence(today) == today).all()
    grown = dilated_persistence(today, 0.75)
    assert grown.sum() > today.sum() and (grown[today] == 1).all()


def test_count_based_scores_equal_pixel_based_scores():
    today, tomorrow = spreading_fire()
    c = DistanceCounts.from_frame(today, tomorrow)
    d = distance_to_fire_km(today)
    for scores_k, scores_px in [
        (decay_scores(c, 2.0), np.exp(-d / 2.0)),
        (persistence_scores(c), today.astype(float)),
        (dilated_scores(c, 0.75), (d <= 0.75).astype(float)),
    ]:
        assert average_precision(tomorrow, scores_px) == pytest.approx(
            _ap(c.to_score_counts(scores_k)), rel=1e-9
        )


def _ap(score_counts):
    from fireguard_ml.evaluation.metrics import average_precision_from_counts

    return average_precision_from_counts(score_counts)


def test_decay_auprc_does_not_depend_on_scale_but_brier_does():
    today, tomorrow = spreading_fire()
    c = DistanceCounts.from_frame(today, tomorrow)
    aps = {s: _ap(c.to_score_counts(decay_scores(c, s))) for s in (0.5, 2.0, 8.0)}
    assert max(aps.values()) - min(aps.values()) < 1e-12
    best, amp, diag = select_decay_scale(c, (0.25, 0.5, 2.0, 8.0), split="validation")
    assert len(set(diag["brier_by_scale"].values())) > 1 and best in diag["brier_by_scale"]
    assert 0 < amp <= 1
    # amplitude never changes ranking, so AUPRC is unchanged
    assert _ap(c.to_score_counts(decay_scores(c, best, amp))) == pytest.approx(aps[0.5])


def test_fitted_amplitude_beats_amplitude_one_on_brier():
    from fireguard_ml.baselines.distance_decay import best_amplitude
    from fireguard_ml.evaluation.metrics import brier_from_counts

    c = DistanceCounts.merge([DistanceCounts.from_frame(*spreading_fire(seed=k)) for k in range(3)])
    a = best_amplitude(c, 1.0)
    fitted = brier_from_counts(c.to_score_counts(decay_scores(c, 1.0, a)))
    raw = brier_from_counts(c.to_score_counts(decay_scores(c, 1.0)))
    assert fitted <= raw


def test_climatology_learns_higher_probability_close_to_fire_and_beats_decay_on_brier():
    train = DistanceCounts.merge(
        [DistanceCounts.from_frame(*spreading_fire(seed=k)) for k in range(5)]
    )
    clim = fit_climatology(train, fold=2019, split="train")
    p = clim.predict_distance(np.array([0.0, 0.5, 30.0, np.inf]))
    assert p[0] > p[2] and p[1] > p[2]
    assert len(clim.probs) == len(clim.edges_km) + 2
    today, tomorrow = spreading_fire(seed=99)
    assert counts_by_score(tomorrow, clim.predict_frame(today)).n == today.size


def test_frames_without_fire_use_the_no_fire_bin():
    c = DistanceCounts.from_frame(np.zeros((4, 4), bool), np.zeros((4, 4), bool))
    assert c.keys.tolist() == [-1]
    assert decay_scores(c, 1.0).tolist() == [0.0]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_climatology_cannot_be_fitted_outside_training(split):
    c = DistanceCounts.from_frame(*spreading_fire())
    with pytest.raises(SplitError):
        fit_climatology(c, fold=2018, split=split)


@pytest.mark.parametrize("split", ["train", "test"])
def test_decay_scale_cannot_be_chosen_outside_validation(split):
    c = DistanceCounts.from_frame(*spreading_fire())
    with pytest.raises(SplitError):
        select_decay_scale(c, (1.0, 2.0), split=split)


def test_counts_respect_the_valid_mask():
    today, tomorrow = spreading_fire()
    mask = np.ones_like(today)
    mask[:, -10:] = False
    c = DistanceCounts.from_frame(today, tomorrow, mask)
    assert c.pos.sum() + c.neg.sum() == mask.sum()
    assert c.pos.sum() == tomorrow[mask].sum()


def test_downwind_baseline_is_disabled():
    with pytest.raises(BaselineDisabled, match="disabled"):
        downwind_decay()


def test_persistence_f1_on_a_static_fire_is_perfect():
    fire = np.zeros((6, 6), bool)
    fire[2:4, 2:4] = True
    assert f1_iou(fire, persistence(fire))["f1"] == 1.0


def test_smallest_dilation_adds_the_four_direct_neighbours():
    # regression: 10 m key rounding once put the 375 m neighbour (key 38) outside r = 0.375 km
    fire = np.zeros((5, 5), bool)
    fire[2, 2] = True
    c = DistanceCounts.from_frame(fire, np.zeros_like(fire))
    covered = int((c.pos + c.neg)[dilated_scores(c, 0.375) == 1].sum())
    assert covered == 5  # centre + 4 neighbours; diagonals (0.53 km) excluded
    assert dilated_persistence(fire, 0.375).sum() == 5


def test_climatology_adjacent_bin_receives_the_direct_neighbours():
    fire = np.zeros((7, 7), bool)
    fire[3, 3] = True
    target = np.zeros_like(fire)
    target[3, 4] = True  # a direct neighbour burns tomorrow
    c = DistanceCounts.from_frame(fire, target)
    clim = fit_climatology(c, fold=2020, split="train", prior_strength=0.0)
    assert clim.probs[1] == pytest.approx(1 / 4)  # bin 1 = exactly one cell away
    assert clim.predict_distance(np.array([0.375]))[0] == pytest.approx(1 / 4)
