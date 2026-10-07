"""The eight evaluation protocols (proposal section 10).

Five standard protocols keep us honest on generalization; three non-standard ones carry the
calibration claim. Each descriptor says how to pick train/eval data and how to group results.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    name: str
    hold_out: str            # the axis held out between train and test
    group_by: str            # how per-group metrics are aggregated
    note: str


PROTOCOLS: dict[str, Protocol] = {
    # --- five standard ---
    "cross_generator": Protocol(
        "cross_generator", "generator", "generator", "train on some generators, test on unseen ones"
    ),
    "cross_corpus": Protocol(
        "cross_corpus", "dataset", "dataset", "train on ASVspoof5 + MLAAD, test on ITW/ML-ITW/DF-Eval, no retraining"
    ),
    "cross_lingual": Protocol(
        "cross_lingual", "language", "language", "hold out whole languages; report calibration loss per language"
    ),
    "codec_stress": Protocol(
        "codec_stress", "codec", "condition", "apply the condition grid; report every metric per condition, never pooled"
    ),
    "benign_edit": Protocol(
        "benign_edit", "benign_edit", "benign_edit", "feed denoised/restored real speech; count false alarms"
    ),
    # --- three non-standard, and decisive ---
    "threshold_transfer": Protocol(
        "threshold_transfer", "condition", "condition", "set threshold on clean dev, apply unchanged; report actDCF drift"
    ),
    "shortcut_control": Protocol(
        "shortcut_control", "silence", "condition", "trim silence + add a silence-only baseline; gains surviving trimming are real"
    ),
    "evidence_faithfulness": Protocol(
        "evidence_faithfulness", "region", "generator", "IoU of evidence directions vs partial-spoof ground-truth regions"
    ),
}
