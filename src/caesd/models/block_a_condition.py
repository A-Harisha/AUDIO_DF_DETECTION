"""Block A -- condition encoder.

A small multi-task head on the backbone predicts the channel, language, bandwidth and SNR.
The augmentation grid supplies these labels for free. The shared output is a continuous
*condition vector*, not a class, so unseen conditions land between known ones.

pooled features [B, D]  ->  {condition [B, C], channel_logits, language_logits, snr, bandwidth}
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        condition_dim: int = 32,
        n_channels: int = 8,
        n_languages: int = 12,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, condition_dim)
        )
        self.channel_head = nn.Linear(condition_dim, n_channels)
        self.language_head = nn.Linear(condition_dim, n_languages)
        self.snr_head = nn.Linear(condition_dim, 1)
        self.bandwidth_head = nn.Linear(condition_dim, 1)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        condition = self.trunk(pooled)
        return {
            "condition": condition,
            "channel_logits": self.channel_head(condition),
            "language_logits": self.language_head(condition),
            "snr": self.snr_head(condition).squeeze(-1),
            "bandwidth": self.bandwidth_head(condition).squeeze(-1),
        }
