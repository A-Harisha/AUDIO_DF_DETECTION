"""Losses. The detector learns to separate; calibration is a separate, post-hoc stage."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_LN2 = 0.6931471805599453


def cllr_loss(llr: torch.Tensor, binary_label: torch.Tensor) -> torch.Tensor:
    """Differentiable Cllr. ``binary_label`` 1 = bona fide, 0 = spoof. Both classes must be present."""
    genuine = binary_label == 1
    return 0.5 * (F.softplus(-llr[genuine]).mean() + F.softplus(llr[~genuine]).mean()) / _LN2


def detection_loss(out: dict, label: torch.Tensor, channel: torch.Tensor,
                   cond_weight: float) -> tuple[torch.Tensor, dict]:
    """Class-balanced BCE on the score, plus codec supervision of block A when it is enabled."""
    y = label.float()
    pos = y.mean().clamp(1e-3, 1 - 1e-3)
    weight = torch.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
    bce = F.binary_cross_entropy_with_logits(out["score"], y, weight=weight)
    parts = {"bce": float(bce)}
    loss = bce
    if out["channel_logits"] is not None:
        cond = F.cross_entropy(out["channel_logits"], channel)
        loss = loss + cond_weight * cond
        parts["cond"] = float(cond)
    return loss, parts
