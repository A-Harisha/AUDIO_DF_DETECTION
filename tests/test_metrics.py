"""Known-answer tests for the metric suite. These identities are the reference the whole
project trusts: Cllr==1 for zero-information LLRs, EER==0 for separable scores, minCllr<=Cllr.
"""
import numpy as np
import pytest

from caesd.metrics import (
    act_dcf,
    bootstrap_ci,
    cllr,
    compute_eer,
    min_cllr,
    min_dcf,
    misleading_evidence_rates,
    threshold_transfer,
)

rng = np.random.default_rng(0)


def _separable(n=300):
    scores = np.concatenate([rng.normal(5, 0.1, n), rng.normal(-5, 0.1, n)])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return scores, labels


def test_eer_separable_is_zero():
    scores, labels = _separable()
    eer, _ = compute_eer(scores, labels)
    assert eer < 0.01


def test_eer_identical_is_half():
    scores = rng.normal(0, 1, 600)
    labels = np.r_[np.ones(300), np.zeros(300)]
    eer, _ = compute_eer(scores, labels)
    assert abs(eer - 0.5) < 0.1


def test_cllr_zero_llr_is_one():
    labels = np.r_[np.ones(100), np.zeros(100)]
    assert cllr(np.zeros(200), labels) == pytest.approx(1.0, abs=1e-9)


def test_cllr_separable_small_and_min_le_cllr():
    scores, labels = _separable()
    llr = 4.0 * scores  # roughly calibrated, strongly separating
    assert cllr(llr, labels) < 0.1
    assert min_cllr(scores, labels) <= cllr(llr, labels) + 1e-9


def test_min_dcf_separable_zero_and_actdcf_noinfo_one():
    scores, labels = _separable()
    assert min_dcf(scores, labels) == pytest.approx(0.0, abs=1e-9)
    labels2 = np.r_[np.ones(100), np.zeros(100)]
    assert act_dcf(np.zeros(200), labels2, p_target=0.5) == pytest.approx(1.0, abs=1e-9)


def test_threshold_transfer_matched_hits_target():
    scores, labels = _separable()
    res = threshold_transfer(scores, labels, scores, labels, far_target=0.05)
    assert res["test_far"] == pytest.approx(0.05, abs=0.02)


def test_misleading_rates_bounds():
    scores, labels = _separable()
    rates = misleading_evidence_rates(scores, labels)
    assert rates["misleading_bonafide"] < 0.01 and rates["misleading_spoof"] < 0.01


def test_bootstrap_ci_brackets_point():
    scores, labels = _separable()
    ci = bootstrap_ci(lambda s, y: compute_eer(s, y)[0], scores, labels, n_resamples=200, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
