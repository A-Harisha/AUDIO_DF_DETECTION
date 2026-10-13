"""The condition gate: mixture weights over ensemble members from the channel descriptor alone.

    scores [N, M], z [N, Z]  ->  one score [N]

Why the descriptor and not the detector's own features. The gate must route, never classify. ``z``
is a 25-dimensional spectral descriptor of the waveform, and ``label_probe_eer`` on it must sit near
0.5: at that point the gate provably carries no bona fide / spoof evidence of its own, so any gain
it produces is routing between members and not a second detector smuggled into the combiner. The
first version of block C read a classifier trained on the DETECTOR's features and that leak
invalidated its whole result (docs/experiment_log.md, 2026-10-07). The same precondition is the
reason this gate is reportable.

Why the kernel-and-null-anchor shape. It is block C's one good property, kept: the condition-dependent
part is a kernel average over K anchors against a null anchor that carries no shift, so as ``z``
leaves the anchors' span every kernel weight vanishes and the gate becomes EXACTLY one global
mixture. An unseen codec therefore gets the best condition-independent fusion rather than an
extrapolated weighting, which is what made v1's unconstrained MLP swing between runs. That global
mixture is trained by its own term in the loss and detached from the residual path, so the fallback
is the best global fusion under this objective rather than merely some fusion -- and it doubles as
the control the gate must beat.

Scores are inputs, never parameters: no gradient reaches any member.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..train.losses import cllr_loss
from .block_c_hypernet import label_probe_eer

__all__ = ["ConditionGate", "label_probe_eer"]


class ConditionGate(nn.Module):
    def __init__(self, descriptor_dim: int, n_members: int, n_anchors: int = 8) -> None:
        super().__init__()
        self.n_members = n_members
        self.anchors = nn.Parameter(torch.zeros(n_anchors, descriptor_dim))  # set from data in fit
        self.residual = nn.Parameter(torch.zeros(n_anchors, n_members))  # per-anchor logit shift
        self.log_tau = nn.Parameter(torch.zeros(()))     # kernel width; softplus(0.5413) = 1
        self.null_logit = nn.Parameter(torch.zeros(()))  # how far z must be before the gate stops acting
        self.base = nn.Parameter(torch.zeros(n_members))   # the global mixture it starts from
        self.scale = nn.Parameter(torch.zeros(n_members))  # per-member affine: members differ in scale
        self.bias = nn.Parameter(torch.zeros(n_members))
        self.register_buffer("z_mean", torch.zeros(descriptor_dim))
        self.register_buffer("z_std", torch.ones(descriptor_dim))
        self._fitted = False

    def kernel_weights(self, z: torch.Tensor) -> torch.Tensor:
        """Kernel weight of every anchor, with the null anchor last. [N, K+1]"""
        d2 = torch.cdist((z - self.z_mean) / self.z_std, self.anchors) ** 2
        logits = torch.cat([-d2 / F.softplus(self.log_tau + 0.5413),
                            self.null_logit.expand(z.shape[0], 1)], dim=1)
        return torch.softmax(logits, dim=1)

    def members(self, scores: torch.Tensor, detach: bool = False) -> torch.Tensor:
        """Members on one comparable scale. [N, M]"""
        scale, bias = (self.scale.detach(), self.bias.detach()) if detach else (self.scale, self.bias)
        return F.softplus(scale + 0.5413) * scores + bias  # scale >= 0: never flips a member's sign

    def global_mixture(self, scores: torch.Tensor) -> torch.Tensor:
        """The condition-independent weighted fusion: what a clip gets once every kernel weight has
        vanished, and the control the gate must beat."""
        return (torch.softmax(self.base, dim=0) * self.members(scores)).sum(-1)

    def mixture_weights(self, z: torch.Tensor) -> torch.Tensor:
        """Member weights per clip. [N, M]"""
        shift = self.kernel_weights(z)[:, :-1] @ self.residual
        return torch.softmax(self.base.detach() + shift, dim=-1)

    def forward(self, scores: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # base, scale and bias are detached here and trained by the global term instead, so the
        # fallback is the best global fusion with no tug-of-war against the residual path.
        return (self.mixture_weights(z) * self.members(scores, detach=True)).sum(-1)

    def null_weight(self, z: torch.Tensor) -> torch.Tensor:
        """Weight on the null anchor per clip. 1 means the clip got the plain global mixture.
        Report its mean per test condition: it says exactly when the gate chose to act."""
        return self.kernel_weights(z)[:, -1]

    def fit(self, scores: np.ndarray, z: np.ndarray, label: np.ndarray, group: np.ndarray,
            holdout: np.ndarray, steps: int = 2000, lr: float = 1e-2, weight_decay: float = 1e-3,
            global_weight: float = 1.0, seed: int = 0) -> dict:
        """AdamW on the condition-balanced Cllr of the gated mixture, plus ``global_weight`` times
        the same loss on the global mixture alone. ``holdout`` marks the speaker-disjoint rows used
        for step selection only; the best held-out state is restored."""
        torch.manual_seed(seed)
        s, zz, y = (torch.tensor(np.asarray(v), dtype=torch.float32) for v in (scores, z, label))
        if s.shape[1] != self.n_members:
            raise ValueError(f"{s.shape[1]} member scores for a gate over {self.n_members} members")
        hold = torch.as_tensor(np.asarray(holdout))

        with torch.no_grad():
            self.z_mean.copy_(zz[~hold].mean(0))
            self.z_std.copy_(zz[~hold].std(0).clamp_min(1e-6))
            normed = (zz[~hold] - self.z_mean) / self.z_std
            pick = torch.randperm(len(normed))[: self.anchors.shape[0]]
            self.anchors.copy_(normed[pick])  # anchors start ON the data manifold
            self.residual.zero_()             # and the gate starts at exactly the global mixture

        def group_masks(rows: torch.Tensor) -> list[torch.Tensor]:
            masks = [rows & torch.as_tensor(group == g) for g in np.unique(group)]
            return [m for m in masks if m.any() and y[m].min() != y[m].max()]

        def balanced_cllr(masks: list[torch.Tensor], out: torch.Tensor) -> torch.Tensor:
            """Cllr averaged over conditions, so no single condition dominates the gate."""
            return torch.stack([cllr_loss(out[m], y[m]) for m in masks]).mean()

        fit_masks, stop_masks = group_masks(~hold), group_masks(hold)
        if not fit_masks or not stop_masks:
            raise ValueError("the gate needs at least one condition with both classes in each half")
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        best, best_step, curve = float("inf"), 0, []
        state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        for step in range(steps):
            loss = (balanced_cllr(fit_masks, self(s, zz))
                    + global_weight * balanced_cllr(fit_masks, self.global_mixture(s)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                held = float(balanced_cllr(stop_masks, self(s, zz)))
            curve.append((float(loss), held))
            if held < best:
                best, best_step = held, step
                state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        self.load_state_dict(state)
        self._fitted = True
        return {"best_step": best_step, "best_holdout_cllr": best, "steps": steps,
                "global_weights": torch.softmax(self.base, 0).tolist()}

    @torch.no_grad()
    def transform(self, scores: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
        """The gated score, the global-mixture control and the diagnostics, for a test set."""
        if not self._fitted:
            raise RuntimeError("ConditionGate.transform called before fit")
        s, zz = (torch.tensor(np.asarray(v), dtype=torch.float32) for v in (scores, z))
        return {"gated": self(s, zz).numpy().astype(np.float64),
                "global": self.global_mixture(s).numpy().astype(np.float64),
                "weights": self.mixture_weights(zz).numpy(),
                "null_weight": self.null_weight(zz).numpy()}
