"""The paired CI decides whether a result is claimed, so its two failure modes are asserted:
an interval that straddles zero when the systems are identical, and one that does not when the
difference is real."""
from __future__ import annotations

import numpy as np
import pytest

from caesd.metrics import compute_eer, paired_ci

EER = lambda s, y: compute_eer(s, y)[0]  # noqa: E731


def data(n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], n // 2)
    scores = rng.normal(labels * 2.0, 1.0)
    return scores, labels


def test_identical_systems_give_an_interval_containing_zero() -> None:
    scores, labels = data()
    lo, hi = paired_ci(EER, scores, scores, labels, n_resamples=300, seed=1)
    assert lo == 0.0 and hi == 0.0  # the same scores on the same resample differ by exactly nothing


def test_a_real_difference_excludes_zero_and_points_the_right_way() -> None:
    scores, labels = data()
    better = scores + labels * 4.0  # pushes bona fide up: strictly more separable
    lo, hi = paired_ci(EER, better, scores, labels, n_resamples=300, seed=1)
    assert hi < 0.0, f"a clearly better system gave CI [{lo}, {hi}]"


def test_a_tiny_difference_is_not_claimed() -> None:
    scores, labels = data()
    rng = np.random.default_rng(7)
    lo, hi = paired_ci(EER, scores + rng.normal(0, 1e-3, scores.size), scores, labels,
                       n_resamples=300, seed=1)
    assert lo <= 0.0 <= hi, "noise was reported as a difference"


def test_single_class_labels_raise_instead_of_fabricating() -> None:
    scores = np.linspace(0, 1, 50)
    with pytest.raises(ValueError, match="fabricated"):
        paired_ci(EER, scores, scores, np.ones(50, dtype=int), n_resamples=50, seed=1)
