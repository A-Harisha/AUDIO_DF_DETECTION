"""Equal error rate via the canonical ASVspoof detection-curve construction."""
from __future__ import annotations

import numpy as np


def _split(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    target = scores[labels == 1]
    nontarget = scores[labels == 0]
    if target.size == 0 or nontarget.size == 0:
        raise ValueError("need at least one target and one non-target score")
    return target, nontarget


def compute_det_curve(scores: np.ndarray, labels: np.ndarray):
    """Return (frr, far, thresholds). Higher score => more target (bona fide)."""
    target, nontarget = _split(scores, labels)
    n_scores = target.size + nontarget.size
    all_scores = np.concatenate((target, nontarget))
    lab = np.concatenate((np.ones(target.size), np.zeros(nontarget.size)))
    order = np.argsort(all_scores, kind="mergesort")
    lab = lab[order]

    tar_sums = np.cumsum(lab)
    non_sums = nontarget.size - (np.arange(1, n_scores + 1) - tar_sums)
    frr = np.concatenate((np.atleast_1d(0.0), tar_sums / target.size))
    far = np.concatenate((np.atleast_1d(1.0), non_sums / nontarget.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[order[0]] - 1e-3), all_scores[order]))
    return frr, far, thresholds


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold at which it occurs."""
    frr, far, thresholds = compute_det_curve(scores, labels)
    idx = np.nanargmin(np.abs(frr - far))
    eer = float((frr[idx] + far[idx]) / 2.0)
    return eer, float(thresholds[idx])
