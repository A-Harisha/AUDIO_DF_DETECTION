"""Losses. The classification head learns to separate; the hypernetwork learns to calibrate.

Training the calibrated LLR with a soft-Cllr objective is what makes calibration an accuracy
result, not an afterthought.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.detector import CLASS_NAMES

CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
_LN2 = 0.6931471805599453


def cllr_loss(llr: torch.Tensor, binary_label: torch.Tensor) -> torch.Tensor:
    """Differentiable Cllr. ``binary_label`` 1 = genuine (bona/processed), 0 = synthetic."""
    genuine = binary_label == 1
    synthetic = ~genuine
    terms = []
    if genuine.any():
        terms.append(F.softplus(-llr[genuine]).mean() / _LN2)
    if synthetic.any():
        terms.append(F.softplus(llr[synthetic]).mean() / _LN2)
    return 0.5 * torch.stack(terms).sum()


def condition_losses(aux: dict, targets: dict) -> torch.Tensor:
    """Multi-task condition supervision: free labels from the augmentation grid."""
    loss = aux["channel_logits"].new_zeros(())
    if "channel" in targets:
        loss = loss + F.cross_entropy(aux["channel_logits"], targets["channel"])
    if "language" in targets:
        loss = loss + F.cross_entropy(aux["language_logits"], targets["language"])
    if "snr" in targets:
        loss = loss + F.mse_loss(aux["snr"], targets["snr"])
    if "bandwidth" in targets:
        loss = loss + F.mse_loss(aux["bandwidth"], targets["bandwidth"])
    return loss


def total_loss(
    out,
    llr: torch.Tensor,
    class_target: torch.Tensor,
    binary_label: torch.Tensor,
    condition_targets: dict | None = None,
    weights: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    w = {"cls": 1.0, "cllr": 1.0, "cond": 0.5, **(weights or {})}
    cls = F.cross_entropy(out.class_logits, class_target)
    cal = cllr_loss(llr, binary_label)
    cond = condition_losses(out.condition_aux, condition_targets) if condition_targets else llr.new_zeros(())
    loss = w["cls"] * cls + w["cllr"] * cal + w["cond"] * cond
    return loss, {"loss": float(loss), "cls": float(cls), "cllr": float(cal), "cond": float(cond)}
