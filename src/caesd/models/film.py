"""FiLM conditioning: the condition vector scales and shifts pooled features.

This is a different site from the calibration hypernetwork (block C). FiLM steers the evidence;
the hypernetwork steers how much that evidence is worth.

features [B, D] + condition [B, C]  ->  modulated features [B, D]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(condition_dim, 2 * feature_dim)
        self.feature_dim = feature_dim

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(condition).chunk(2, dim=-1)
        return (1.0 + gamma) * features + beta
