"""Full CAESD model. One model, no ensemble. The condition vector is the only shared signal
and it does two jobs: it tells FiLM where the evidence is weak, and it tells the hypernetwork
how much the evidence is worth.

Three output classes -- bona fide, processed bona fide, synthetic -- stop the system flagging
denoised or restored real speech as fake.

waveform [B, T] -> DetectorOutput
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .backbone import Backbone
from .block_a_condition import ConditionEncoder
from .block_c_hypernet import CalibrationHypernet
from .film import FiLM

CLASS_NAMES = ("bona_fide", "processed_bona_fide", "synthetic")
GENUINE_IDX = (0, 1)
SYNTHETIC_IDX = 2


@dataclass
class DetectorOutput:
    features: torch.Tensor          # [B, L, D]
    pooled: torch.Tensor            # [B, D]
    condition: torch.Tensor         # [B, C]
    condition_aux: dict             # channel_logits, language_logits, snr, bandwidth
    class_logits: torch.Tensor      # [B, 3]
    raw_score: torch.Tensor         # [B]  higher => bona fide


class CaesdDetector(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        condition_dim: int = 32,
        n_channels: int = 8,
        n_languages: int = 12,
    ) -> None:
        super().__init__()
        d = backbone.feature_dim
        self.backbone = backbone
        self.condition_encoder = ConditionEncoder(d, condition_dim, n_channels, n_languages)
        self.film = FiLM(d, condition_dim)
        self.classifier = nn.Linear(d, len(CLASS_NAMES))
        self.hypernet = CalibrationHypernet(condition_dim)

    def forward(self, wave: torch.Tensor) -> DetectorOutput:
        features = self.backbone(wave)
        pooled = features.mean(dim=1)
        cond = self.condition_encoder(pooled)
        condition = cond.pop("condition")
        modulated = self.film(pooled, condition)
        logits = self.classifier(modulated)
        genuine = torch.logsumexp(logits[:, GENUINE_IDX], dim=-1)
        raw_score = genuine - logits[:, SYNTHETIC_IDX]
        return DetectorOutput(features, pooled, condition, cond, logits, raw_score)

    def calibrated_llr(self, out: DetectorOutput) -> torch.Tensor:
        """Condition-dependent LLR from block C."""
        return self.hypernet.calibrate(out.raw_score, out.condition)
