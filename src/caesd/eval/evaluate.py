"""Run the metric suite. Discrimination from raw scores; calibration from LLRs.

Always reported per group (language, codec), never only pooled -- inside one published grid,
losses ran from -0.4 to +53.5 points, which a single number hides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import (
    act_dcf,
    cllr,
    compute_eer,
    min_cllr,
    min_dcf,
    misleading_evidence_rates,
    miss_at_far,
)


def evaluate_scores(
    df: pd.DataFrame, raw_col: str = "score", llr_col: str = "llr", label_col: str = "label"
) -> dict[str, float]:
    raw = df[raw_col].to_numpy(dtype=float)
    labels = df[label_col].to_numpy()
    llr = df[llr_col].to_numpy(dtype=float) if llr_col in df else raw
    eer, _ = compute_eer(raw, labels)
    result = {
        "n": len(df),
        "eer": eer,
        "min_dcf": min_dcf(raw, labels),
        "act_dcf": act_dcf(llr, labels),
        "cllr": cllr(llr, labels),
        "min_cllr": min_cllr(raw, labels),
        "miss_at_1pct_fa": miss_at_far(raw, labels, 0.01),
        "miss_at_0p1pct_fa": miss_at_far(raw, labels, 0.001),
    }
    result["calibration_loss"] = result["act_dcf"] - result["min_dcf"]
    result.update(misleading_evidence_rates(llr, labels))
    return result


def evaluate_per_group(
    df: pd.DataFrame, group_col: str, min_count: int = 30, **kwargs
) -> dict[str, dict]:
    """Metric suite for each group that has both classes present."""
    out: dict[str, dict] = {}
    for group, sub in df.groupby(group_col):
        labels = sub[kwargs.get("label_col", "label")].to_numpy()
        if len(sub) < min_count or np.unique(labels).size < 2:
            continue
        out[str(group)] = evaluate_scores(sub, **kwargs)
    return out
