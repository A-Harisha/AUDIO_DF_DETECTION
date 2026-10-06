"""Block D -- competence gate via Mondrian (group-conditional) conformal prediction.

Split conformal over the validated condition region gives a guaranteed error rate per group
(groups are language and codec). Outside that region -- an unseen group, or a prediction set
that is not a confident singleton -- the system outputs *insufficient evidence* instead of a
number. This matches the forensic framework: the expert reports evidential strength; the
court decides.

Operates on class probabilities, so it is a post-hoc wrapper, not a differentiable module.
"""
from __future__ import annotations

import numpy as np


class MondrianConformalGate:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.quantiles: dict[str, float] = {}  # per-group nonconformity threshold

    def fit(self, probs: np.ndarray, true_labels: np.ndarray, groups: np.ndarray) -> "MondrianConformalGate":
        """Calibrate per-group thresholds. Nonconformity = 1 - p(true class)."""
        probs = np.asarray(probs, dtype=np.float64)
        true_labels = np.asarray(true_labels).astype(int)
        groups = np.asarray(groups).astype(str)
        nonconf = 1.0 - probs[np.arange(len(true_labels)), true_labels]
        for g in np.unique(groups):
            mask = groups == g
            n = int(mask.sum())
            level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
            self.quantiles[g] = float(np.quantile(nonconf[mask], level, method="higher"))
        return self

    def predict(self, probs: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
        """Return prediction sets plus ``abstain`` and ``covered`` masks.

        A sample abstains (insufficient evidence) when its group was never validated, or when
        its conformal set is not a confident singleton.
        """
        probs = np.asarray(probs, dtype=np.float64)
        groups = np.asarray(groups).astype(str)
        n, k = probs.shape
        nonconf = 1.0 - probs
        sets: list[np.ndarray] = []
        abstain = np.zeros(n, dtype=bool)
        covered = np.zeros(n, dtype=bool)
        for i in range(n):
            g = groups[i]
            if g not in self.quantiles:
                sets.append(np.array([], dtype=int))
                abstain[i] = True
                continue
            covered[i] = True
            members = np.where(nonconf[i] <= self.quantiles[g])[0]
            sets.append(members)
            if members.size != 1:
                abstain[i] = True
        return {"sets": sets, "abstain": abstain, "covered": covered}
