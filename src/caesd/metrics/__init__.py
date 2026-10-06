"""Evaluation metrics. Built once and unit-tested against known-answer identities.

Score convention used everywhere in this package:
    higher score  -> more likely BONA FIDE (the positive / target class)
    label == 1    -> bona fide (target)
    label == 0    -> spoof (non-target)
LLR convention: llr = log[ p(x | bona fide) / p(x | spoof) ];  llr > 0 supports bona fide.
"""
from .bootstrap import bootstrap_ci
from .cllr import cllr, min_cllr
from .dcf import act_dcf, min_dcf
from .eer import compute_det_curve, compute_eer
from .misleading_evidence import misleading_evidence_rates
from .operating_point import far_frr_at_threshold, miss_at_far, threshold_at_far, threshold_transfer

__all__ = [
    "bootstrap_ci",
    "cllr",
    "min_cllr",
    "act_dcf",
    "min_dcf",
    "compute_det_curve",
    "compute_eer",
    "misleading_evidence_rates",
    "far_frr_at_threshold",
    "miss_at_far",
    "threshold_at_far",
    "threshold_transfer",
]
