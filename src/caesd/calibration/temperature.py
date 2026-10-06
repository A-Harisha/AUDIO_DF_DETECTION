"""Temperature scaling baseline: a single scalar, llr = score / T.

A reviewer will ask whether this simple fix is enough. T is chosen to minimise Cllr on the
development set.
"""
from __future__ import annotations

import numpy as np

from ..metrics.cllr import cllr


class TemperatureCalibrator:
    def __init__(self) -> None:
        self.temperature: float | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> TemperatureCalibrator:
        from scipy.optimize import minimize_scalar

        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels)

        def objective(log_t: float) -> float:
            return cllr(scores / np.exp(log_t), labels)

        res = minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded")
        self.temperature = float(np.exp(res.x))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError("TemperatureCalibrator.transform called before fit")
        return np.asarray(scores, dtype=np.float64) / self.temperature
