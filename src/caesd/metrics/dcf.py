"""Detection cost function: minDCF (best threshold) and actDCF (Bayes threshold on LLRs).

The gap actDCF - minDCF is the calibration loss. It is the central quantity of this project.
"""
from __future__ import annotations

import numpy as np

from .eer import _split, compute_det_curve

# ASVspoof 5 primary-metric defaults.
P_TARGET = 0.05
C_MISS = 1.0
C_FA = 1.0


def _norm(p_target: float, c_miss: float, c_fa: float) -> float:
    return min(c_miss * p_target, c_fa * (1.0 - p_target))


def min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = P_TARGET,
    c_miss: float = C_MISS,
    c_fa: float = C_FA,
) -> float:
    """Minimum normalized DCF over all thresholds. 0 for perfectly separable scores."""
    frr, far, _ = compute_det_curve(scores, labels)
    dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    return float(np.min(dcf) / _norm(p_target, c_miss, c_fa))


def act_dcf(
    llr: np.ndarray,
    labels: np.ndarray,
    p_target: float = P_TARGET,
    c_miss: float = C_MISS,
    c_fa: float = C_FA,
) -> float:
    """Normalized DCF at the theoretical Bayes threshold on calibrated LLRs."""
    target, nontarget = _split(llr, labels)
    theta = np.log((1.0 - p_target) * c_fa / (p_target * c_miss))
    frr = float(np.mean(target <= theta))
    far = float(np.mean(nontarget > theta))
    dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    return dcf / _norm(p_target, c_miss, c_fa)
