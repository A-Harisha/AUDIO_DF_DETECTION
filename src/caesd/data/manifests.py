"""Manifests and leakage control.

Train / dev / test must be disjoint by speaker, sentence and generator. Leakage is the most
common quiet failure in this area, so the split is enforced, not assumed.
"""
from __future__ import annotations

import hashlib

import pandas as pd

COLUMNS = [
    "path", "label", "class_name", "speaker", "sentence",
    "generator", "language", "dataset", "condition", "split",
]


def build_manifest(records: list[dict]) -> pd.DataFrame:
    """Build a manifest DataFrame, filling missing columns with None."""
    df = pd.DataFrame(records)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[COLUMNS]


def assign_split_by(df: pd.DataFrame, key: str, fractions=(0.8, 0.1, 0.1), seed: int = 0) -> pd.DataFrame:
    """Assign train/dev/test deterministically so every value of ``key`` lands in one split."""
    names = ["train", "dev", "test"]
    bounds = [fractions[0], fractions[0] + fractions[1]]
    df = df.copy()

    def bucket(value: object) -> str:
        h = hashlib.md5(f"{seed}:{value}".encode()).hexdigest()
        frac = int(h[:8], 16) / 0xFFFFFFFF
        return names[0] if frac < bounds[0] else names[1] if frac < bounds[1] else names[2]

    df["split"] = df[key].map(bucket)
    return df


def assert_no_leakage(df: pd.DataFrame, keys=("speaker", "sentence", "generator")) -> None:
    """Raise if any key value appears in more than one split."""
    for key in keys:
        present = df[df[key].notna()]
        counts = present.groupby(key)["split"].nunique()
        leaked = counts[counts > 1]
        if len(leaked):
            raise AssertionError(f"leakage on '{key}': {list(leaked.index)[:10]} span multiple splits")
