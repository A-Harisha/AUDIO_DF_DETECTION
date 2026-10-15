"""Reading the members of an ensemble: aligned score matrices and the scale they are combined on.

Shared by the score-level study (``scripts/ensemble.py``) and the attack harness
(``scripts/attack.py``) so both combine members on exactly the same statistics. If they did not,
the robustness number would be measured on a different system from the accuracy number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger("caesd.eval.fusion")


def member_scores(runs_dir: Path, runs: list[str], name: str, meta_columns: list[str]
                  ) -> tuple[pd.DataFrame, np.ndarray]:
    """Metadata and the [N, M] score matrix of score set ``name``, on the clips EVERY member scored.

    Members need not have scored the same clips. A head trained on cached features scores the dev
    sets that were extracted (47,400 clips), while a member trained on audio scores the whole dev
    manifest (140,950), so requiring identical clip sets made a heterogeneous ensemble impossible to
    fuse at all. The rows are therefore the intersection of the members' ``utt`` values, ordered by
    ``utt`` so the result does not depend on which run came first.

    Alignment is still explicit rather than assumed -- every member is indexed BY ``utt``, never by
    position, because a silent misalignment here would look exactly like a weak member. What is
    checked instead is that the intersection is not a small corner of the smallest member, which
    would quietly change what the reported metrics are computed over."""
    frames = {run: pd.read_parquet(runs_dir / run / f"scores_{name}.parquet") for run in runs}
    common = set.intersection(*(set(f["utt"]) for f in frames.values()))
    if not common:
        raise RuntimeError(f"{name}: the members share no clip; nothing can be fused")
    smallest = min(len(f) for f in frames.values())
    if len(common) < 0.9 * smallest:
        sizes = {run: len(f) for run, f in frames.items()}
        raise RuntimeError(
            f"{name}: only {len(common)} clips are common to every member, against {smallest} in the "
            f"smallest ({sizes}); the metrics would be over an unintended subset")
    order = np.sort(np.fromiter(common, dtype=frames[runs[0]]["utt"].to_numpy().dtype, count=len(common)))
    aligned = {run: f.set_index("utt").loc[order] for run, f in frames.items()}
    meta = aligned[runs[0]].reset_index()[meta_columns].reset_index(drop=True)
    scores = [aligned[run]["score"].to_numpy() for run in runs]
    if len(common) != smallest:
        log.info("%s: fusing on %d clips common to every member (smallest member has %d)",
                 name, len(common), smallest)
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
