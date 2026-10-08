"""Block C -- condition-dependent calibration, as a POST-HOC stage.

    llr = a(z) * score + b(z),   z = a signal-level channel descriptor.

Three defects of the first version are fixed by construction (docs/experiment_log.md, 2026-10-07):

  leakage        v1 read the codec log-posterior of a classifier trained on the DETECTOR's own
                 frozen features, so (a, b) carried class evidence and the LLR moved the EER.
                 Here z comes from the waveform only. ``label_probe_eer`` is the precondition:
                 it must sit near 0.5, or no claim about block C is valid.
  extrapolation  v1's unconstrained MLP mapped an unseen condition anywhere, so leave-one-codec-out
                 Cllr ranged 0.209-0.517. Here the residual is a kernel average over K anchors
                 against a null anchor carrying no residual: as z leaves the anchors' span every
                 kernel weight vanishes and the map becomes EXACTLY one global calibration. That
                 global map (a0, b0) is trained by its OWN term in the loss and is detached from
                 the residual path, so the fallback is the best global calibration under this
                 objective rather than merely some global calibration -- without that, the fallback
                 is reached but is poor (an unseen codec scored Cllr 1.39 against 0.56 for pooled
                 at null weight 1.00). It minimises Cllr averaged over conditions, which is the
                 right uniform prior for a condition that has not been seen; a pooled logistic fit
                 optimises something else, so the two need not agree on any single condition.
  overfitting    v1 ran 2000 fixed steps on all of the calibration data. Here the step is chosen
                 on a speaker-disjoint held-out part of it.

Scores are inputs, never parameters: no gradient reaches the detector.
The loss is Cllr averaged over conditions, so no single condition dominates.

score [N], z [N, Z]  ->  llr [N]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..train.losses import cllr_loss


def label_probe_eer(z_fit: np.ndarray, label_fit: np.ndarray,
                    z_test: np.ndarray, label_test: np.ndarray) -> float:
    """The precondition. A logistic probe on the descriptor ALONE must give ~0.5 EER: the channel
    descriptor must not predict bona fide / spoof. Anything lower means z leaks class evidence,
    which is exactly what invalidated v1's seen-codec win."""
    from sklearn.linear_model import LogisticRegression

    from ..metrics import compute_eer

    probe = LogisticRegression(class_weight="balanced", max_iter=2000).fit(z_fit, label_fit)
    return float(compute_eer(probe.decision_function(z_test), label_test)[0])


class CalibrationHypernet(nn.Module):
    def __init__(self, descriptor_dim: int, n_anchors: int = 8) -> None:
        super().__init__()
        self.anchors = nn.Parameter(torch.zeros(n_anchors, descriptor_dim))  # set from data in fit
        self.residual = nn.Parameter(torch.zeros(n_anchors, 2))  # (da, db) per anchor
        self.log_tau = nn.Parameter(torch.zeros(()))     # kernel width; softplus(0.5413) = 1
        self.null_logit = nn.Parameter(torch.zeros(()))  # how far z must be before C stops acting
        self.a0 = nn.Parameter(torch.zeros(()))          # the global calibration C starts from
        self.b0 = nn.Parameter(torch.zeros(()))
        self.register_buffer("z_mean", torch.zeros(descriptor_dim))
        self.register_buffer("z_std", torch.ones(descriptor_dim))
        self._fitted = False

    def weights(self, z: torch.Tensor) -> torch.Tensor:
        """Kernel weight of every anchor, with the null anchor last. [N, K+1]"""
        d2 = torch.cdist((z - self.z_mean) / self.z_std, self.anchors) ** 2
        logits = torch.cat([-d2 / F.softplus(self.log_tau + 0.5413),
                            self.null_logit.expand(z.shape[0], 1)], dim=1)
        return torch.softmax(logits, dim=1)

    def global_calibration(self, score: torch.Tensor) -> torch.Tensor:
        """The map a clip gets once every kernel weight has vanished: the fallback for a condition
        far from every anchor. ``fit`` optimises it on its own as well as inside the full map."""
        return F.softplus(self.a0 + 0.5413) * score + self.b0

    def forward(self, score: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # The base is detached, so ONLY the anchor term of the loss moves (a0, b0): the fallback is
        # then exactly the best global calibration under the objective, with no tug-of-war against
        # the residual path. The residuals are unbounded, so nothing is lost in expressiveness.
        da, db = (self.weights(z)[:, :-1] @ self.residual).unbind(dim=-1)
        a = F.softplus(self.a0.detach() + da + 0.5413)  # scale >= 0: never flips the score sign
        return a * score + self.b0.detach() + db

    def fit(self, score: np.ndarray, z: np.ndarray, label: np.ndarray, group: np.ndarray,
            holdout: np.ndarray, steps: int = 2000, lr: float = 1e-2, weight_decay: float = 1e-3,
            anchor_weight: float = 1.0, seed: int = 0) -> dict:
        """AdamW on the condition-balanced Cllr of the full map, plus ``anchor_weight`` times the
        same loss on the global map alone, so the fallback is itself a good global calibration.
        ``holdout`` marks the speaker-disjoint rows used for step selection only; the best
        held-out state is restored. Returns the curves."""
        torch.manual_seed(seed)
        s, zz, y = (torch.tensor(np.asarray(v), dtype=torch.float32) for v in (score, z, label))
        hold = torch.as_tensor(np.asarray(holdout))

        with torch.no_grad():
            self.z_mean.copy_(zz[~hold].mean(0))
            self.z_std.copy_(zz[~hold].std(0).clamp_min(1e-6))
            normed = (zz[~hold] - self.z_mean) / self.z_std
            pick = torch.randperm(len(normed))[: self.anchors.shape[0]]
            self.anchors.copy_(normed[pick])  # anchors start ON the data manifold
            self.residual.zero_()             # and the map starts at exactly one global calibration

        def group_masks(rows: torch.Tensor) -> list[torch.Tensor]:
            """One mask per condition that has both classes in ``rows``."""
            masks = [rows & torch.as_tensor(group == g) for g in np.unique(group)]
            return [m for m in masks if m.any() and y[m].min() != y[m].max()]

        def balanced_cllr(masks: list[torch.Tensor], llr: torch.Tensor) -> torch.Tensor:
            """Cllr averaged over conditions, so no single condition dominates."""
            return torch.stack([cllr_loss(llr[m], y[m]) for m in masks]).mean()

        fit_masks, stop_masks = group_masks(~hold), group_masks(hold)
        if not fit_masks or not stop_masks:
            raise ValueError("block C needs at least one condition with both classes in each half")
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        best, best_step, curve = float("inf"), 0, []
        state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        for step in range(steps):
            loss = (balanced_cllr(fit_masks, self(s, zz))
                    + anchor_weight * balanced_cllr(fit_masks, self.global_calibration(s)))
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
        return {"best_step": best_step, "best_holdout_cllr": best, "steps": steps, "curve": curve}

    @torch.no_grad()
    def transform(self, score: np.ndarray, z: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("CalibrationHypernet.transform called before fit")
        s, zz = (torch.tensor(np.asarray(v), dtype=torch.float32) for v in (score, z))
        return self(s, zz).numpy().astype(np.float64)

    @torch.no_grad()
    def null_weight(self, z: np.ndarray) -> np.ndarray:
        """Weight on the null anchor per clip. 1 means the clip got the plain global calibration.
        Report its mean per test condition: it says exactly when block C chose to act."""
        return self.weights(torch.tensor(np.asarray(z), dtype=torch.float32))[:, -1].numpy()
