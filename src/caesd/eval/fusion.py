"""Reading the members of an ensemble: aligned score matrices and the scale they are combined on.

Shared by the score-level study (``scripts/ensemble.py``) and the attack harness
(``scripts/attack.py``) so both combine members on exactly the same statistics. If they did not,
the robustness number would be measured on a different system from the accuracy number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def member_scores(runs_dir: Path, runs: list[str], name: str, meta_columns: list[str]
                  ) -> tuple[pd.DataFrame, np.ndarray]:
    """Metadata and the [N, M] score matrix of score set ``name``, aligned on ``utt``.

    Every member must have scored exactly the same clips, in the same order. Alignment is checked
    rather than assumed: a silent misalignment here would look like a weak member."""
    frames = [pd.read_parquet(runs_dir / run / f"scores_{name}.parquet") for run in runs]
    meta = frames[0][meta_columns].reset_index(drop=True)
    utt = frames[0]["utt"].to_numpy()
    scores = []
    for run, frame in zip(runs, frames, strict=True):
        if len(frame) != len(meta) or not (frame["utt"].to_numpy() == utt).all():
            raise RuntimeError(f"{run} scored {name} on different clips than {runs[0]}")
        scores.append(frame["score"].to_numpy())
    return meta, np.stack(scores, axis=1)


def calib_scale(runs_dir: Path, runs: list[str], views: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-member mean and standard deviation over the dev ``calib`` rows of ``views``.

    Members differ in score scale, so every combiner reads z-scores. The statistics come from the
    calibration split only, which model selection never used."""
    raw, keep = [], []
    for view in views:
        meta, s = member_scores(runs_dir, runs, f"dev_{view}", ["utt", "split"])
        raw.append(s)
        keep.append((meta["split"] == "calib").to_numpy())
    raw, keep = np.concatenate(raw), np.concatenate(keep)
    if keep.sum() < 100 * len(runs):
        raise RuntimeError(f"only {keep.sum()} calib rows for {len(runs)} members: scale would be noise")
    return raw[keep].mean(0), raw[keep].std(0).clip(1e-6)


def descriptor(features_dir: Path, name: str) -> pd.DataFrame:
    """``utt``, ``sim_codec`` and the cached channel descriptor ``z`` of score set ``name``.

    The gate and the lookup baselines read only this. It is computed from the waveform and never
    from the SSL features the members use, which is what keeps the gate class-blind; joining on
    ``utt`` means nothing assumes an order shared with the score files."""
    path = features_dir / "descriptor" / name
    if not path.with_suffix(".parquet").exists():
        partition, view = name.split("_", 1)
        raise FileNotFoundError(
            f"{path}.parquet missing: run scripts/extract_features.py "
            f"+partition={partition} +view={view} +descriptor_only=true")
    meta = pd.read_parquet(path.with_suffix(".parquet"), columns=["utt", "sim_codec"])
    meta["z"] = list(np.load(path.with_suffix(".npy")))
    return meta
