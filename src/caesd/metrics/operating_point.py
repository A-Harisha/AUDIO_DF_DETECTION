"""Fixed-operating-point metrics. These carry the project's accuracy claim.

Deployment fixes a threshold before seeing the data. Threshold transfer measures what that
costs under shift: set the threshold on clean dev, apply it unchanged to degraded test.
"""
from __future__ import annotations

import numpy as np

from .eer import _split


def far_frr_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """Decide target if score > threshold. Return (far, frr)."""
    target, nontarget = _split(scores, labels)
    far = float(np.mean(nontarget > threshold))
    frr = float(np.mean(target <= threshold))
    return far, frr


def threshold_at_far(scores: np.ndarray, labels: np.ndarray, far_target: float) -> float:
    """Threshold on this set that yields the requested false-alarm rate."""
    _, nontarget = _split(scores, labels)
    return float(np.quantile(nontarget, 1.0 - far_target))


def miss_at_far(scores: np.ndarray, labels: np.ndarray, far_target: float) -> float:
    """Miss rate (FRR) at the threshold that gives ``far_target`` on the same set."""
    threshold = threshold_at_far(scores, labels, far_target)
    _, frr = far_frr_at_threshold(scores, labels, threshold)
    return frr


def threshold_transfer(
    dev_scores: np.ndarray,
    dev_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    far_target: float = 0.01,
) -> dict[str, float]:
    """Fix the threshold on dev at ``far_target``; report what it becomes on test."""
    threshold = threshold_at_far(dev_scores, dev_labels, far_target)
    test_far, test_frr = far_frr_at_threshold(test_scores, test_labels, threshold)
    return {
        "far_target": float(far_target),
        "threshold": threshold,
        "test_far": test_far,
        "test_frr": test_frr,
    }
