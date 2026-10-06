"""Minimal, observable training loop. Scaled up in Phase 2; correct and logged from day one."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..models.detector import CaesdDetector
from ..utils.logging import get_logger, log_shapes
from .losses import CLASS_INDEX, total_loss

log = get_logger("caesd.train")


def fit(
    model: CaesdDetector,
    loader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
    loss_weights: dict | None = None,
) -> list[dict]:
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    log.info("trainable params: %.2fM", sum(p.numel() for p in params) / 1e6)

    history: list[dict] = []
    logged_shapes = False
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(loader):
            wave = batch["wave"].to(device)
            binary_label = batch["label"].to(device)
            class_target = torch.tensor(
                [CLASS_INDEX[c] for c in batch["class_name"]], device=device
            )
            out = model(wave)
            llr = model.calibrated_llr(out)
            if not logged_shapes:
                log_shapes(log, wave=wave, features=out.features, condition=out.condition, llr=llr)
                logged_shapes = True
            loss, parts = total_loss(out, llr, class_target, binary_label, weights=loss_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0:
                log.info("epoch %d step %d | %s", epoch, step, parts)
            parts["epoch"] = epoch
            history.append(parts)
    return history
