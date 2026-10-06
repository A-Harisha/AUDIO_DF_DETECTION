"""Per-condition lookup-table calibration: the Phase-0 baseline that justifies the hypernetwork.

Fit one logistic calibrator per condition cell and store it. This is the obvious objection a
reviewer raises: detect the condition, look up the calibration, apply it -- fifty lines, no
hypernetwork. The experiment holds out whole codecs and whole languages so the table has no
cell to use. There is NO fallback: an uncovered condition is reported as uncovered, never
quietly calibrated with the wrong cell.
"""
from __future__ import annotations

import numpy as np

from .logistic import LogisticCalibrator


class LookupTableCalibrator:
    def __init__(self, min_per_cell: int = 50) -> None:
        self.min_per_cell = min_per_cell
        self.cells: dict[str, LogisticCalibrator] = {}

    def fit(self, scores: np.ndarray, labels: np.ndarray, conditions: np.ndarray) -> LookupTableCalibrator:
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels)
        conditions = np.asarray(conditions)
        for cond in np.unique(conditions):
            mask = conditions == cond
            sub_labels = labels[mask]
            # A cell needs both classes and enough samples to fit a stable calibrator.
            if mask.sum() >= self.min_per_cell and sub_labels.min() != sub_labels.max():
                self.cells[str(cond)] = LogisticCalibrator().fit(scores[mask], sub_labels)
        return self

    def transform(self, scores: np.ndarray, conditions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (llr, covered). ``llr`` is NaN where no cell exists; ``covered`` is the mask."""
        scores = np.asarray(scores, dtype=np.float64)
        conditions = np.asarray(conditions)
        llr = np.full(scores.shape, np.nan, dtype=np.float64)
        covered = np.zeros(scores.shape, dtype=bool)
        for cond, calibrator in self.cells.items():
            mask = conditions.astype(str) == cond
            if mask.any():
                llr[mask] = calibrator.transform(scores[mask])
                covered[mask] = True
        return llr, covered
