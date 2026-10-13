"""Known-answer tests for the condition gate.

The gate exists to route between members whose strengths depend on the condition, and to collapse to
one global mixture where the condition is unlike anything it fit on. Both are asserted on a problem
whose right answer is known by construction, because on real data neither is visible directly.
"""
from __future__ import annotations

import numpy as np

from caesd.metrics import compute_eer
from caesd.models.gate import ConditionGate

N, M, Z = 4000, 3, 25


def routing_problem(seed: int = 0) -> tuple[np.ndarray, ...]:
    """Member 0 separates the classes only in group a, member 1 only in group b, member 2 never.
    A global mixture must therefore be beaten by any gate that reads the group from ``z``."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(N, Z)).astype("float32")
    group = np.where(z[:, 0] > 0, "a", "b")
    label = rng.integers(0, 2, N)
    scores = np.stack([
        np.where(group == "a", label * 2.0, 0.0) + rng.normal(0, 0.5, N),
        np.where(group == "b", label * 2.0, 0.0) + rng.normal(0, 0.5, N),
        rng.normal(0, 1, N),
    ], axis=1).astype("float32")
    holdout = np.zeros(N, dtype=bool)
    holdout[: N // 5] = True
    return scores, z, label, group, holdout


def test_gate_routes_to_the_expert_of_each_condition() -> None:
    scores, z, label, group, holdout = routing_problem()
    gate = ConditionGate(Z, M, n_anchors=8)
    info = gate.fit(scores, z, label, group, holdout, steps=300)
    out = gate.transform(scores, z)

    assert compute_eer(out["gated"], label)[0] < 0.6 * compute_eer(out["global"], label)[0]
    weights = out["weights"]
    assert weights[group == "a", 0].mean() > weights[group == "a", 1].mean()
    assert weights[group == "b", 1].mean() > weights[group == "b", 0].mean()
    # The useless member must be down-weighted by the global mixture, condition or not.
    assert info["global_weights"][2] < 0.1
    assert weights[:, 2].mean() < 0.1


def test_gate_collapses_to_the_global_mixture_far_from_every_anchor() -> None:
    """A condition nothing was fit on must get the global mixture exactly, not an extrapolation."""
    scores, z, label, group, holdout = routing_problem()
    gate = ConditionGate(Z, M, n_anchors=8)
    gate.fit(scores, z, label, group, holdout, steps=200)

    far = np.full((16, Z), 50.0, dtype="float32")  # 50 standard deviations from the data
    out = gate.transform(scores[:16], far)
    assert np.all(out["null_weight"] > 0.999)
    assert np.allclose(out["gated"], out["global"], atol=1e-4)


def test_gate_rejects_a_member_count_mismatch() -> None:
    scores, z, label, group, holdout = routing_problem()
    gate = ConditionGate(Z, M + 1, n_anchors=4)
    try:
        gate.fit(scores, z, label, group, holdout, steps=1)
    except ValueError as exc:
        assert "member" in str(exc)
    else:
        raise AssertionError("a gate over 4 members accepted 3 member scores")
