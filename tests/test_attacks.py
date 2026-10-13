"""Known-answer tests for the attack budgets and for the attack directions.

An attack that silently misses its budget would overstate robustness, and one that perturbs the
zero padding would understate it, so both are asserted rather than assumed.
"""
from __future__ import annotations

import torch

from caesd.attacks import l2_budget, pgd, project, snr_db, spsa

LENGTHS = torch.tensor([400, 256])


def waves() -> torch.Tensor:
    torch.manual_seed(0)
    wave = torch.randn(2, 400)
    wave[1, 256:] = 0.0  # the second clip is padded
    return wave


def linear_score(weight: torch.Tensor):
    """A differentiable detector whose gradient is known exactly: score = <w, x>."""
    return lambda wave, lengths: wave @ weight


def test_l2_budget_matches_requested_snr() -> None:
    wave = waves()
    budget = l2_budget(wave, LENGTHS, snr_db=30.0)
    signal = ((wave * (torch.arange(400)[None] < LENGTHS[:, None])) ** 2).sum(1, keepdim=True).sqrt()
    assert torch.allclose(budget / signal, torch.full_like(budget, 10 ** -1.5))


def test_project_respects_both_balls_and_the_padding() -> None:
    wave = waves()
    delta = torch.randn_like(wave)
    mask = (torch.arange(400)[None] < LENGTHS[:, None]).to(wave.dtype)
    budget = torch.full((2, 1), 0.01)
    linf = project(delta, mask, "linf", budget)
    assert linf.abs().max() <= 0.01 + 1e-9
    l2 = project(delta, mask, "l2", budget)
    assert torch.all(l2.norm(dim=1) <= 0.01 + 1e-6)
    for d in (linf, l2):
        assert torch.all(d[1, 256:] == 0.0)


def test_pgd_moves_the_score_in_the_goal_direction_within_budget() -> None:
    wave = waves()
    weight = torch.randn(400)
    score_fn = linear_score(weight)
    budget = l2_budget(wave, LENGTHS, snr_db=20.0)
    before = score_fn(wave, LENGTHS)
    for goal in (1, -1):
        attacked = pgd(score_fn, wave, LENGTHS, goal, budget, norm="l2", steps=20)
        after = score_fn(attacked, LENGTHS)
        assert torch.all(goal * (after - before) > 0)
        delta = attacked - wave
        assert torch.all(delta.norm(dim=1) <= budget.squeeze(1) + 1e-5)
        assert torch.all(delta[1, 256:] == 0.0)
        # A linear score is maximised on the sphere, so PGD must reach the budget, not stop short.
        assert torch.all(delta.norm(dim=1) > 0.99 * budget.squeeze(1))
        assert torch.all(snr_db(wave, attacked, LENGTHS) < 20.5)


def test_spsa_moves_the_score_without_a_gradient() -> None:
    wave = waves()
    score_fn = linear_score(torch.randn(400))
    budget = l2_budget(wave, LENGTHS, snr_db=20.0)
    before = score_fn(wave, LENGTHS)
    attacked, queries = spsa(score_fn, wave, LENGTHS, 1, budget, steps=40, n_samples=8, sigma=1e-2)
    assert queries == 40 * 2 * 8
    assert torch.all(score_fn(attacked, LENGTHS) - before > 0)
    assert torch.all((attacked - wave).norm(dim=1) <= budget.squeeze(1) + 1e-5)
    assert torch.all((attacked - wave)[1, 256:] == 0.0)
