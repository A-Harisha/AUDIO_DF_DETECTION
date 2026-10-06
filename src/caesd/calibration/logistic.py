"""Linear logistic calibration: the forensic standard. llr = a * score + b.

Parameters are fit with balanced class weighting, i.e. an effective prior of 0.5, so the
output is a log-likelihood-ratio rather than a prior-dependent log-posterior-odds.
"""
from __future__ import annotations

import numpy as np


class LogisticCalibrator:
    def __init__(self) -> None:
        self.a: float | None = None
        self.b: float | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> LogisticCalibrator:
        from sklearn.linear_model import LogisticRegression

        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels).astype(int)
        clf = LogisticRegression(class_weight="balanced", C=1e6, solver="lbfgs")
        clf.fit(scores, labels)
        self.a = float(clf.coef_[0, 0])
        self.b = float(clf.intercept_[0])
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.a is None:
            raise RuntimeError("LogisticCalibrator.transform called before fit")
        return self.a * np.asarray(scores, dtype=np.float64) + self.b
