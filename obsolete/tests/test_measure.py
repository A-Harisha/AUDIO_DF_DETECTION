"""Phase 1 measurement runs and reports per-condition / per-language metrics."""
import numpy as np
import pandas as pd

from caesd.eval import run_measurement

rng = np.random.default_rng(1)


def _rows(codec, shift, n=400):
    recs = []
    for split in ("dev", "test"):
        for lab, mu in ((1, 2 + shift), (0, -2 + shift)):
            for s in rng.normal(mu, 0.5, n):
                recs.append({"score": float(s), "label": lab, "condition": f"{codec}_en",
                             "codec": codec, "language": "en", "split": split})
    return recs


def test_measurement_reports_groups_and_calibration_loss():
    df = pd.DataFrame(_rows("clean", 0.0) + _rows("mp3", 0.0) + _rows("speex8k", 5.0))
    result = run_measurement(df)
    assert "eer" in result["pooled"] and "calibration_loss" in result["pooled"]
    assert set(result["per_condition"]) == {"clean_en", "mp3_en", "speex8k_en"}
    # the shifted codec must show a larger fixed-threshold calibration loss than clean
    assert result["per_condition"]["speex8k_en"]["calibration_loss"] >= \
        result["per_condition"]["clean_en"]["calibration_loss"]
