"""Segment-level IoU between evidence regions and partial-spoof ground truth.

High evidence with no overlap means the explanation is false. Uses partial-spoof data with
ground-truth manipulated regions (seconds).
"""
from __future__ import annotations

import numpy as np


def _total_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return float(sum(e - s for s, e in merged))


def segment_iou(
    predicted: list[tuple[float, float]], ground_truth: list[tuple[float, float]]
) -> float:
    """Intersection-over-union of two sets of time intervals (seconds)."""
    if not predicted and not ground_truth:
        return 1.0
    grid = np.unique(np.concatenate([np.array(predicted + ground_truth).ravel(), [0.0]]))
    inter = 0.0
    for a, b in zip(grid[:-1], grid[1:]):
        mid = 0.5 * (a + b)
        in_pred = any(s <= mid <= e for s, e in predicted)
        in_gt = any(s <= mid <= e for s, e in ground_truth)
        if in_pred and in_gt:
            inter += b - a
    union = _total_length(predicted) + _total_length(ground_truth) - inter
    return float(inter / union) if union > 0 else 0.0
