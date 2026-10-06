"""Bootstrap confidence intervals. A 0.2% EER difference often means nothing without one."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    scores: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for a (scores, labels) -> float metric."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = scores.size
    point = metric_fn(scores, labels)
    draws = np.empty(n_resamples, dtype=np.float64)
    count = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        sub_labels = labels[idx]
        if sub_labels.min() == sub_labels.max():
            continue  # a degenerate resample has only one class; drop it, do not fake a value
        draws[count] = metric_fn(scores[idx], sub_labels)
        count += 1
    draws = draws[:count]
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return {"point": float(point), "lo": lo, "hi": hi, "n_effective": count}
