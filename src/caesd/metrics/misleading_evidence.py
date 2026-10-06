"""Rates of misleading evidence: how often a calibrated LLR points the wrong way.

Required for forensic validation. For a bona-fide trial, llr < 0 is misleading; for a
spoof trial, llr > 0 is misleading.
"""
from __future__ import annotations

import numpy as np

from .eer import _split


def misleading_evidence_rates(llr: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    target, nontarget = _split(llr, labels)
    return {
        "misleading_bonafide": float(np.mean(target < 0.0)),
        "misleading_spoof": float(np.mean(nontarget > 0.0)),
    }
