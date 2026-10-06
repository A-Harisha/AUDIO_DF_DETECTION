"""Block C -- calibration hypernetwork. The core novelty.

A small network reads the condition vector and *generates* the logistic-calibration
parameters (a, b). There is no single fixed calibration map. For the same raw score the LLR
shrinks toward zero as the condition leaves the validated region, because the generated scale
``a`` collapses there -- the system learns to distrust its own evidence off-condition.

condition [B, C]  ->  {a [B], b [B]};   llr = a * raw_score + b
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationHypernet(nn.Module):
    def __init__(self, condition_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )

    def forward(self, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        a_raw, b = self.net(condition).unbind(dim=-1)
        a = F.softplus(a_raw)  # scale >= 0: evidence never flips sign, only weakens
        return {"a": a, "b": b}

    def calibrate(self, raw_score: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        params = self.forward(condition)
        return params["a"] * raw_score + params["b"]
