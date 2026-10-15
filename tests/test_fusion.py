"""Member scores must be aligned by ``utt``, never by position.

A head trained on cached features scores the dev sets that were extracted; a member trained on audio
scores the whole dev manifest. Fusing those by row position would silently pair one clip's score
with another clip's label, which reads as a weak member rather than as an error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caesd.eval import member_scores


def write(run_dir, utts: list[str], scores: list[float]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"utt": utts, "score": scores, "label": [i % 2 for i in range(len(utts))],
                  "split": ["calib"] * len(utts)}).to_parquet(run_dir / "scores_dev_clean.parquet")


def test_members_are_paired_by_utt_not_by_row_order(tmp_path) -> None:
    write(tmp_path / "a", ["u1", "u2", "u3"], [1.0, 2.0, 3.0])
    write(tmp_path / "b", ["u3", "u1", "u2"], [30.0, 10.0, 20.0])  # same clips, shuffled
    meta, raw = member_scores(tmp_path, ["a", "b"], "dev_clean", ["utt", "label"])
    assert list(meta["utt"]) == ["u1", "u2", "u3"]
    assert raw[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert raw[:, 1].tolist() == [10.0, 20.0, 30.0]  # paired by utt, so b's order does not matter


def test_a_member_that_scored_more_clips_is_intersected(tmp_path) -> None:
    """The audio-trained member scores a superset; fusion happens on the common clips."""
    write(tmp_path / "features", ["u1", "u2", "u3"], [1.0, 2.0, 3.0])
    write(tmp_path / "audio", ["u0", "u1", "u2", "u3"], [0.0, 10.0, 20.0, 30.0])
    meta, raw = member_scores(tmp_path, ["features", "audio"], "dev_clean", ["utt", "label"])
    assert list(meta["utt"]) == ["u1", "u2", "u3"]
    assert raw.shape == (3, 2)
    assert raw[:, 1].tolist() == [10.0, 20.0, 30.0]


def test_a_tiny_overlap_is_refused(tmp_path) -> None:
    """A near-disjoint pair would silently report metrics over an unintended corner of the data."""
    write(tmp_path / "a", [f"u{i}" for i in range(100)], list(np.arange(100.0)))
    write(tmp_path / "b", [f"u{i}" for i in range(90, 190)], list(np.arange(100.0)))
    with pytest.raises(RuntimeError, match="common to every member"):
        member_scores(tmp_path, ["a", "b"], "dev_clean", ["utt", "label"])


def test_disjoint_members_are_refused(tmp_path) -> None:
    write(tmp_path / "a", ["u1", "u2"], [1.0, 2.0])
    write(tmp_path / "b", ["v1", "v2"], [1.0, 2.0])
    with pytest.raises(RuntimeError, match="share no clip"):
        member_scores(tmp_path, ["a", "b"], "dev_clean", ["utt", "label"])
