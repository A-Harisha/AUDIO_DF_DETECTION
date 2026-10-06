"""Cost of the log-likelihood-ratio (Cllr) and its PAV-optimal floor (minCllr).

Identities used as known-answer tests:
    - llr == 0 everywhere  ->  Cllr == 1      (no information)
    - perfectly separable  ->  Cllr ~ 0
    - minCllr <= Cllr always
"""
from __future__ import annotations

import numpy as np

_LN2 = np.log(2.0)


def _softplus2(x: np.ndarray) -> np.ndarray:
    """log2(1 + exp(x)) computed stably (handles +/- inf)."""
    return np.logaddexp(0.0, x) / _LN2


def cllr(llr: np.ndarray, labels: np.ndarray) -> float:
    llr = np.asarray(llr, dtype=np.float64)
    labels = np.asarray(labels)
    tar = llr[labels == 1]
    non = llr[labels == 0]
    if tar.size == 0 or non.size == 0:
        raise ValueError("need both classes present")
    return float(0.5 * (np.mean(_softplus2(-tar)) + np.mean(_softplus2(non))))


def min_cllr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Cllr after optimal (PAV / isotonic) calibration of the raw scores to LLRs."""
    from sklearn.isotonic import IsotonicRegression

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(np.float64)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    posterior = iso.fit_transform(scores, labels)
    # effective prior 0.5 (Cllr weights the two error terms equally) -> optimal llr is the logit.
    with np.errstate(divide="ignore"):
        opt_llr = np.log(posterior) - np.log1p(-posterior)
    return cllr(opt_llr, labels)
