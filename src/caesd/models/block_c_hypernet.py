"""Block C -- calibration hypernetwork, as a POST-HOC stage.

A small network reads a condition vector and *generates* the logistic-calibration parameters
(a, b):  llr = a * score + b.  There is no single fixed calibration map.

Three rules, each fixing a failure recorded in docs/experiment_log.md (2026-10-07):
  - It is fit on held-out scores (the dev ``calib`` split), never on detector training data.
  - Scores are inputs, not parameters: no gradient reaches the detector.
  - The condition vector is the codec posterior of a separate condition classifier, not the
    detector features, so (a, b) cannot carry class evidence beyond what the codec carries.
The loss is Cllr averaged over conditions, so no single condition dominates.

condition [N, C], score [N]  ->  llr [N]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..train.losses import cllr_loss


class CalibrationHypernet(nn.Module):
    def __init__(self, condition_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(condition_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.a0 = nn.Parameter(torch.zeros(()))  # global calibration; the network adds a residual
        self.b0 = nn.Parameter(torch.zeros(()))

    def forward(self, score: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        da, db = self.net(condition).unbind(dim=-1)
        a = F.softplus(self.a0 + da + 0.5413)  # softplus(0.5413) = 1; scale >= 0, never flips sign
        return a * score + self.b0 + db

    def fit(self, score: np.ndarray, condition: np.ndarray, label: np.ndarray, group: np.ndarray,
            steps: int = 2000, lr: float = 1e-2, weight_decay: float = 1e-3, seed: int = 0) -> list[float]:
        """Full-batch AdamW on the condition-balanced Cllr. Returns the loss curve."""
        torch.manual_seed(seed)
        nn.init.zeros_(self.net[-1].weight)  # start exactly at one global calibration
        nn.init.zeros_(self.net[-1].bias)
        s, c, y = (torch.tensor(np.asarray(v), dtype=torch.float32) for v in (score, condition, label))
        groups = [torch.as_tensor(group == g) for g in np.unique(group)]
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        curve = []
        for _ in range(steps):
            llr = self(s, c)
            loss = torch.stack([cllr_loss(llr[m], y[m]) for m in groups]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            curve.append(float(loss))
        return curve

    @torch.no_grad()
    def transform(self, score: np.ndarray, condition: np.ndarray) -> np.ndarray:
        s, c = (torch.tensor(np.asarray(v), dtype=torch.float32) for v in (score, condition))
        return self(s, c).numpy().astype(np.float64)
