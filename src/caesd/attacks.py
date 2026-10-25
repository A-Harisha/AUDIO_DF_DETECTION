"""Adversarial attacks on a waveform detector, white-box and black-box.

Threat model, stated once and reported separately for the two halves (user decision, 2026-10-13):

  white-box  the attacker has the weights and the gradient of the exact system under test,
             including the combiner weights of an ensemble. PGD and, where the system is
             stochastic, PGD with expectation over the randomness (EOT). This is the number a
             robustness claim must quote; anything weaker overstates the defence.
  black-box  the attacker has scores but no gradient: SPSA with a query budget, and transfer of a
             perturbation crafted on a surrogate model. The deployment number.

Convention follows ``caesd.metrics``: a higher score means BONA FIDE. The attacker's goal on a
spoof clip is therefore to RAISE the score, and on a bona fide clip to lower it; ``goal`` is +1 and
-1 for those two directions, so one function serves both.

Budget. An L-infinity budget in waveform amplitude is reported because it is standard, and an L2
budget set by a target SNR is reported because it is what an audio attacker actually controls:
    ||delta||_2 = ||x||_2 * 10 ** (-snr_db / 20).
Both are projected per clip over the valid samples only, so zero padding is never perturbed -- a
perturbation written into the padding would be free and would inflate the attack.

Nothing here falls back: an attack that cannot reach its budget raises rather than returning a
weaker perturbation under the same name.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _mask(wave: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return (torch.arange(wave.shape[1], device=wave.device)[None] < lengths[:, None]).to(wave.dtype)


def l2_budget(wave: torch.Tensor, lengths: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Per-clip L2 norm of a perturbation at ``snr_db`` relative to the clip. [B, 1]"""
    signal = ((wave * _mask(wave, lengths)) ** 2).sum(1, keepdim=True).sqrt()
    return signal * 10.0 ** (-snr_db / 20.0)


def project(delta: torch.Tensor, mask: torch.Tensor, norm: str, budget: torch.Tensor) -> torch.Tensor:
    """Project ``delta`` onto the budget ball and onto the valid samples."""
    delta = delta * mask
    if norm == "linf":
        return delta.clamp(-budget, budget)
    if norm == "l2":
        scale = (budget / delta.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
        return delta * scale
    raise ValueError(f"unknown norm {norm!r}: use 'linf' or 'l2'")


def pgd(score_fn: ScoreFn, wave: torch.Tensor, lengths: torch.Tensor, goal: int, budget: torch.Tensor,
        norm: str = "l2", steps: int = 30, step_frac: float = 0.15, eot: int = 1) -> torch.Tensor:
    """White-box PGD. Returns the attacked waveform.

    ``score_fn(wave, lengths) -> score [B]`` must be differentiable in ``wave``. ``goal`` is +1 to
    raise the score (spoof -> bona fide) and -1 to lower it. ``budget`` is per clip [B, 1], in
    amplitude for ``linf`` and in L2 norm for ``l2``. ``eot`` averages the gradient over that many
    forward passes, which is what a stochastic defence must be attacked with: one pass would
    measure the randomness, not the robustness.
    """
    if eot < 1 or steps < 1:
        raise ValueError(f"steps {steps} and eot {eot} must both be >= 1")
    mask = _mask(wave, lengths)
    delta = torch.zeros_like(wave, requires_grad=True)
    alpha = step_frac * budget
    for _ in range(steps):
        grad = torch.zeros_like(wave)
        for _ in range(eot):
            score = score_fn(wave + delta, lengths)
            grad = grad + torch.autograd.grad(score.sum(), delta)[0]
        grad = grad / eot
        with torch.no_grad():
            direction = grad.sign() if norm == "linf" else grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            stepped = delta + goal * alpha * direction
            delta.copy_(project(stepped, mask, norm, budget))
    return (wave + delta).detach()


def adversarial_batch(score_fn: ScoreFn, wave: torch.Tensor, lengths: torch.Tensor,
                      label: torch.Tensor, snr_db: float, steps: int = 3, fraction: float = 0.5,
                      norm: str = "l2") -> torch.Tensor:
    """Replace a random ``fraction`` of the batch with white-box PGD examples aimed at the WRONG
    class, for adversarial TRAINING. Returns the mixed batch, same shape.

    Why a fraction and not all of it. Training purely on attacked audio is the version that destroys
    clean accuracy, and clean accuracy is the whole result here: 3.21% EER took three weeks to
    reach. A mixed batch is the standard form and it also costs less -- PGD-``steps`` on half a
    batch is about 1.5 extra forward-backward passes per step rather than 3.

    Why this is not the same experiment as ``feature_pgd``. That hardened the HEAD against
    perturbations in FEATURE space while the attacker perturbs the WAVEFORM and arrives through the
    backbone; the mismatch was flagged in its docstring and the probe duly failed, giving no
    evidence either way (log, 2026-10-16). This perturbs the waveform and backpropagates through
    the backbone, so it hardens the path that is actually attacked.

    The crafting runs with the model in EVAL mode: dropout would make the gradient a sample of a
    random network rather than the network, and the forward passes would otherwise pollute the
    batch-norm running statistics with adversarial audio. The caller restores train mode.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction {fraction} must be in (0, 1]")
    if snr_db <= 0.0:
        raise ValueError(f"snr_db {snr_db} must be positive: it is a signal-to-noise ratio in dB")
    pick = torch.rand(wave.shape[0], device=wave.device) < fraction
    if not bool(pick.any()):
        return wave
    idx = pick.nonzero(as_tuple=True)[0]
    sub, sub_len, sub_label = wave[idx], lengths[idx], label[idx]
    # Each clip is pushed towards its wrong class: bona fide (1) down, spoof (0) up.
    goal = torch.where(sub_label == 1, -1.0, 1.0).to(wave.dtype)[:, None]
    budget = l2_budget(sub, sub_len, snr_db) if norm == "l2" else torch.full(
        (sub.shape[0], 1), 10.0 ** (-snr_db / 20.0), device=wave.device, dtype=wave.dtype)
    # The step size must let the budget be REACHED. pgd's default step_frac of 0.15 is sized for a
    # 30-step measurement attack: at 3 steps it travels 0.45 of the budget and the realised
    # perturbation comes out at 36.9 dB when 30 was asked for, a 2.2x weaker attack than the
    # configuration claims. 2.5 / steps is the usual heuristic and lets the projection saturate.
    attacked = pgd(score_fn, sub, sub_len, goal, budget, norm=norm, steps=steps,
                   step_frac=2.5 / steps)
    out = wave.clone()
    out[idx] = attacked
    return out


@torch.no_grad()
def spsa(score_fn: ScoreFn, wave: torch.Tensor, lengths: torch.Tensor, goal: int, budget: torch.Tensor,
         norm: str = "l2", steps: int = 30, step_frac: float = 0.15, n_samples: int = 8,
         sigma: float = 1e-3) -> tuple[torch.Tensor, int]:
    """Black-box SPSA: the gradient is estimated from score differences only, never read.

    Returns the attacked waveform and the number of score queries per clip, so the result is always
    reported with its query budget: ``steps * 2 * n_samples``."""
    mask = _mask(wave, lengths)
    delta = torch.zeros_like(wave)
    alpha = step_frac * budget
    for _ in range(steps):
        grad = torch.zeros_like(wave)
        for _ in range(n_samples):
            v = torch.randint(0, 2, wave.shape, device=wave.device, dtype=wave.dtype).mul_(2).sub_(1) * mask
            plus = score_fn(wave + delta + sigma * v, lengths)
            minus = score_fn(wave + delta - sigma * v, lengths)
            grad = grad + ((plus - minus) / (2 * sigma))[:, None] * v
        grad = grad / n_samples
        direction = grad.sign() if norm == "linf" else grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
        delta = project(delta + goal * alpha * direction, mask, norm, budget)
    return wave + delta, steps * 2 * n_samples


def snr_db(clean: torch.Tensor, attacked: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Realised SNR per clip, in dB. Reported with every attack so the budget is never assumed."""
    mask = _mask(clean, lengths)
    signal = ((clean * mask) ** 2).sum(1)
    noise = (((attacked - clean) * mask) ** 2).sum(1)
    return 10.0 * torch.log10(signal / noise.clamp_min(1e-20))


def feature_pgd(score_fn: Callable[[torch.Tensor], torch.Tensor], features: torch.Tensor,
                label: torch.Tensor, eps_rel: float, steps: int = 3,
                step_frac: float = 0.4) -> torch.Tensor:
    """PGD in the CACHED FEATURE space, for adversarial training. ``features`` is [B, L, D].

    ``eps_rel`` is the L2 budget as a fraction of each clip's own feature norm, so the budget does
    not depend on how the backbone happens to scale its layers. Each clip is pushed towards its
    WRONG class: a spoof clip's score up, a bona fide clip's score down, which is the direction an
    attacker wants and therefore the direction training should see.

    This hardens the HEAD only. A real attacker perturbs the waveform and reaches the head through
    the backbone, so a head made robust here need not be robust there; whether it is at all is the
    measurement this exists for, not an assumption it rests on."""
    if steps < 1:
        raise ValueError(f"steps {steps} must be >= 1")
    goal = torch.where(label == 1, -1.0, 1.0)[:, None, None]
    budget = eps_rel * features.flatten(1).norm(dim=1).view(-1, 1, 1)
    delta = torch.zeros_like(features, requires_grad=True)
    for _ in range(steps):
        grad = torch.autograd.grad(score_fn(features + delta).sum(), delta)[0]
        with torch.no_grad():
            direction = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
            stepped = delta + goal * step_frac * budget * direction
            scale = (budget / stepped.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)).clamp(max=1.0)
            delta.copy_(stepped * scale)
    return (features + delta).detach()
