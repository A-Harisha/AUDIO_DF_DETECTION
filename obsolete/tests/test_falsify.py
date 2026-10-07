"""Phase-0 falsification on synthetic scores engineered so the premise holds:
within every condition the scores separate well (small minDCF), but a held-out codec shifts
the score distribution, so a globally-fit calibration is badly wrong there (large actDCF gap),
and the lookup table has no cell for it.
"""
import numpy as np
import pandas as pd

from caesd.eval import run_falsification

rng = np.random.default_rng(0)


def _rows(codec, shift, n=400):
    recs = []
    for split in ("dev", "test"):
        tar = rng.normal(2 + shift, 0.5, n)
        non = rng.normal(-2 + shift, 0.5, n)
        for s in tar:
            recs.append({"score": s, "label": 1, "condition": f"{codec}_en", "codec": codec,
                         "language": "en", "split": split})
        for s in non:
            recs.append({"score": s, "label": 0, "condition": f"{codec}_en", "codec": codec,
                         "language": "en", "split": split})
    return recs


def _frame():
    recs = _rows("clean", 0.0) + _rows("mp3", 0.0) + _rows("speex8k", 5.0)
    return pd.DataFrame(recs)


def test_premise_holds_gives_go():
    result = run_falsification(_frame(), holdout_codecs=["speex8k"], holdout_languages=[],
                               gap_is_large=0.10, lookup_closes_gap=0.5, min_per_cell=20)
    assert result["decision"] == "GO"
    assert result["gap_global_heldout"] >= 0.10
    assert result["heldout_covered_fraction"] == 0.0  # the table has no cell for speex8k
