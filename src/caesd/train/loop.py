"""Minimal, observable training loop. Scaled up in Phase 2; correct and logged from day one."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models.detector import CaesdDetector
from ..utils.logging import get_logger, log_shapes, write_json
from .losses import CLASS_INDEX, total_loss

log = get_logger("caesd.train")


def _codec_of(condition: str) -> str:
    """'mp3_16k' -> 'mp3', 'clean' -> 'clean'."""
    return condition.split("_")[0]


def fit(
    model: CaesdDetector,
    loader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
    loss_weights: dict | None = None,
    channel_map: dict[str, int] | None = None,
    progress_path: str | Path | None = None,
) -> list[dict]:
    """Train the detector. When ``channel_map`` (codec -> class index) is given, block A's
    channel head is supervised with the codec label from each clip's condition -- this is what
    gives the condition vector its meaning and lets block C calibrate per condition.

    ``progress_path`` gets a flushed JSON heartbeat (epoch/step/loss) each log step, so the run is
    monitorable regardless of stdout buffering."""
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    log.info("trainable params: %.2fM | channel supervision: %s", sum(p.numel() for p in params) / 1e6,
             bool(channel_map))

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
            condition_targets = None
            if channel_map is not None:
                condition_targets = {"channel": torch.tensor(
                    [channel_map[_codec_of(c)] for c in batch["condition"]], device=device)}
            out = model(wave)
            llr = model.calibrated_llr(out)
            if not logged_shapes:
                log_shapes(log, wave=wave, features=out.features, condition=out.condition, llr=llr)
                logged_shapes = True
            loss, parts = total_loss(out, llr, class_target, binary_label,
                                     condition_targets=condition_targets, weights=loss_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0:
                log.info("epoch %d step %d | %s", epoch, step, parts)
                if progress_path is not None:
                    write_json(progress_path, {"epoch": epoch, "epochs": epochs, "step": step,
                                               "steps_per_epoch": len(loader), "loss": parts["loss"],
                                               "cls": parts["cls"], "cllr": parts["cllr"],
                                               "cond": parts["cond"], "ts": time.time()})
            parts["epoch"] = epoch
            history.append(parts)
    return history
