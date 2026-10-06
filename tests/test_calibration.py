"""Calibration behaviour: logistic produces sensible LLRs, the lookup table never silently
calibrates an uncovered condition."""
import numpy as np

from caesd.calibration import LogisticCalibrator, LookupTableCalibrator, TemperatureCalibrator
from caesd.metrics import cllr

rng = np.random.default_rng(0)


def _data(n=400):
    scores = np.concatenate([rng.normal(2, 1, n), rng.normal(-2, 1, n)])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return scores, labels


def test_logistic_improves_cllr():
    scores, labels = _data()
    raw = 100.0 * scores  # badly scaled -> terrible calibration
    cal = LogisticCalibrator().fit(scores, labels)
    assert cllr(cal.transform(scores), labels) < cllr(raw, labels)


def test_temperature_runs_and_calibrates():
    scores, labels = _data()
    cal = TemperatureCalibrator().fit(10.0 * scores, labels)
    assert cal.temperature > 0


def test_lookup_table_flags_uncovered_condition():
    scores, labels = _data(200)
    seen = np.array(["mp3"] * 400)
    table = LookupTableCalibrator(min_per_cell=10).fit(scores, labels, seen)
    test_scores = np.r_[scores[:5], scores[:5]]
    test_cond = np.array(["mp3"] * 5 + ["speex8k"] * 5)  # speex8k never seen
    llr, covered = table.transform(test_scores, test_cond)
    assert covered[:5].all() and not covered[5:].any()
    assert np.isnan(llr[5:]).all()  # uncovered -> NaN, never a wrong-cell value
