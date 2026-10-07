"""Phase 1 measurement: a globally-calibrated detector reports confident, wrong LLRs under
shift. Fit one global calibration on clean dev, then report the metric suite per condition and
per language -- the baselines nobody else publishes under cross-dataset shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..calibration import LogisticCalibrator
from .evaluate import evaluate_per_group, evaluate_scores


def run_measurement(df: pd.DataFrame, far_target: float = 0.01) -> dict:
    """df columns: score, label, condition, codec, language, split."""
    dev_clean = df[(df["split"] == "dev") & (df["codec"] == "clean")]
    if np.unique(dev_clean["label"]).size < 2:
        raise ValueError("need clean dev data with both classes to fit the global calibration")

    cal = LogisticCalibrator().fit(dev_clean["score"].to_numpy(), dev_clean["label"].to_numpy())
    test = df[df["split"] == "test"].copy()
    test["llr"] = cal.transform(test["score"].to_numpy())

    return {
        "calibration": {"a": cal.a, "b": cal.b, "far_target": far_target},
        "pooled": evaluate_scores(test),
        "per_condition": evaluate_per_group(test, "condition"),
        "per_language": evaluate_per_group(test, "language"),
    }
