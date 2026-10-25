"""Observable training loop with validation on unseen attacks, early stopping and one best
checkpoint. The previous loop had no validation, and the model memorised its training attacks
(train EER 0.000, unseen-attack EER 0.217; docs/experiment_log.md, 2026-10-07)."""
from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..metrics import compute_eer, min_dcf
from ..models.detector import CaesdDetector
from ..utils.logging import get_logger
from ..attacks import adversarial_batch, feature_pgd
from ..data.condition_grid import encodec_batch
from .losses import detection_loss

log = get_logger("caesd.train")


@torch.no_grad()
def predict(model: CaesdDetector, loader: Iterable[dict], device: str, transform=None) -> pd.DataFrame:
    """Score every item of ``loader``. Returns idx, score, the true channel index, and the
    predicted channel when block A is on.

    ``transform(x, lengths, channel, idx) -> (x, channel)`` runs on the device before the model. It
    exists for the ``encodec`` view, whose neural codec cannot run in the DataLoader workers: on CPU
    it took minutes per batch and died (2026-10-09). It may relabel the channel, because a clip that
    went through MP3 first is eval C07 and the rest are C04.
    """
    model.eval()
    idx, score, channel, channel_pred = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        lengths = None if batch["lengths"] is None else batch["lengths"].to(device)
        chan = batch["channel"].to(device)
        if transform is not None:
            x, chan = transform(x, lengths, chan, batch["idx"].numpy())
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, lengths)
        idx.append(batch["idx"].cpu().numpy())
        score.append(out["score"].float().cpu().numpy())
        channel.append(chan.cpu().numpy())
        if out["channel_logits"] is not None:
            channel_pred.append(out["channel_logits"].argmax(-1).cpu().numpy())
    frame = pd.DataFrame({"idx": np.concatenate(idx), "score": np.concatenate(score),
                          "channel": np.concatenate(channel)})
    if channel_pred:
        frame["channel_pred"] = np.concatenate(channel_pred)
    return frame.sort_values("idx").reset_index(drop=True)


def validate(model: CaesdDetector, val: dict[str, tuple[Iterable[dict], pd.DataFrame]], device: str) -> dict:
    """EER and minDCF per validation set, plus EER per unseen attack. ``select`` is the mean EER."""
    metrics: dict = {}
    for name, (loader, meta) in val.items():
        scores = predict(model, loader, device)["score"].to_numpy()
        labels = meta["label"].to_numpy()
        metrics[f"{name}/eer"] = compute_eer(scores, labels)[0]
        metrics[f"{name}/min_dcf"] = min_dcf(scores, labels)
        bona = labels == 1
        for attack in sorted(meta.loc[~bona, "generator"].unique()):
            keep = bona | (meta["generator"] == attack).to_numpy()
            metrics[f"{name}/eer_{attack}"] = compute_eer(scores[keep], labels[keep])[0]
    metrics["select"] = float(np.mean([v for k, v in metrics.items() if k.endswith("/eer")]))
    return metrics


def trainable_state(model: torch.nn.Module) -> dict:
    """Every trainable parameter, plus every buffer.

    Buffers are included because a trained-from-scratch front end keeps its batch-norm running
    statistics there, not in its parameters. Without them a fresh load -- a white-box attack
    building the model from the run directory, say -- would score with untrained normalisation
    statistics and silently measure a different model. A frozen backbone has almost no buffers, so
    this costs nothing in the common case."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    names |= {n for n, _ in model.named_buffers()}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def load_trainable_state(model: torch.nn.Module, state: dict) -> None:
    """Load a ``trainable_state``. Every trainable parameter must be present; nothing may be unknown."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    lost = {n for n, p in model.named_parameters() if p.requires_grad} & set(missing)
    if unexpected or lost:
        raise RuntimeError(f"checkpoint mismatch: unexpected={unexpected} missing trainable={sorted(lost)}")


def fit(
    model: CaesdDetector,
    train_loader: Iterable[dict],
    val: dict[str, tuple[Iterable[dict], pd.DataFrame]],
    run_dir: Path,
    epochs: int,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    patience: int,
    cond_weight: float,
    device: str = "cuda",
    select_window: int = 3,
    adv_eps: float = 0.0,
    adv_steps: int = 3,
    encodec_p: float = 0.0,
    wav_adv_snr: float = 0.0,
    wav_adv_steps: int = 3,
    wav_adv_fraction: float = 0.5,
) -> dict:
    """Train; after every epoch validate, append to ``metrics.jsonl`` and keep the best checkpoint.

    The epoch is chosen by the lowest ``select_window``-epoch mean of the validation EER, and the
    weights kept are the middle epoch's own, not an average of them. The raw per-epoch curve moves
    by +-0.5 points, so its argmin is the minimum of the noise: seed 2 of frozen_wavlm_enc selected
    epoch 1, an undertrained head (docs/experiment_log.md, 2026-10-09). Averaging the weights
    instead was tested as EMA and is worse on held-out dev, because late weights overfit the train
    attacks. ``select_window=1`` restores the old rule. Stops after ``patience`` epochs with no
    improvement.

    ``encodec_p`` > 0 puts that fraction of every training batch through EnCodec on the GPU. It is
    the one augmentation a features=false run cannot get from its DataLoader workers, and eval's two
    hardest conditions for a fine-tuned model are exactly the neural codecs (2026-10-17). Off by
    default, so no existing run changes.

    ``wav_adv_snr`` > 0 turns on WAVEFORM adversarial training: that fraction of each batch is
    first replaced by white-box PGD examples at that SNR, crafted through the backbone. This is the
    last untried robustness route -- a single model, four layer bands, two SSL backbones and
    feature-space hardening all left any gradient-bearing attacker at 98-100% success. It costs
    about 2.5x a normal epoch at the default 3 steps on half the batch.

    ``adv_eps`` > 0 turns on feature-space adversarial training: every batch is first pushed towards
    its wrong class inside an L2 ball of ``adv_eps`` times that clip's own feature norm. It hardens
    the HEAD only -- a real attacker perturbs the waveform and arrives through the backbone -- so
    whether it buys any waveform robustness is a measurement, not an assumption. Off by default, so
    no existing run changes."""
    if epochs < select_window:
        raise ValueError(f"epochs {epochs} is below select_window {select_window}: nothing could be selected")
    model.to(device)
    head = list(model.head.parameters())
    body = [p for p in model.backbone.parameters() if p.requires_grad] if model.backbone is not None else []
    groups = [{"params": head, "lr": lr}] + ([{"params": body, "lr": backbone_lr}] if body else [])
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], pct_start=0.1,
                                                total_steps=epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda")
    log.info("trainable params: head %.2fM, backbone %.2fM", sum(p.numel() for p in head) / 1e6,
             sum(p.numel() for p in body) / 1e6)

    best, best_mean = {"select": float("inf"), "epoch": 0}, float("inf")
    recent: deque[tuple[dict, dict]] = deque(maxlen=select_window)
    for epoch in range(epochs):
        model.train()
        start, sums, steps = time.time(), {}, 0
        for step, batch in enumerate(train_loader):
            lengths = None if batch["lengths"] is None else batch["lengths"].to(device)
            x, y = batch["x"].to(device), batch["label"].to(device)
            channel = batch["channel"].to(device)
            if encodec_p > 0.0:
                x, channel = encodec_batch(x, channel, encodec_p, lengths)
            if wav_adv_snr > 0.0:
                # Crafted in eval mode and in float32: dropout would make the gradient a sample of
                # a random network, and a normalised gradient direction underflows in float16.
                model.eval()
                x = adversarial_batch(lambda w, n: model(w, n)["score"].float(), x, lengths, y,
                                      wav_adv_snr, wav_adv_steps, wav_adv_fraction)
                model.train()
            if adv_eps > 0.0:
                # Feature-space adversarial training. Crafted in float32 outside autocast: the
                # perturbation direction is a normalised gradient and float16 loses it to underflow.
                x = feature_pgd(lambda f: model(f, lengths)["score"].float(), x, y, adv_eps, adv_steps)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x, lengths)
            out = {k: None if v is None else v.float() for k, v in out.items()}
            loss, parts = detection_loss(out, y, channel, cond_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            steps += 1
            if epoch == 0 and step == 0:
                log.info("shape | x = %s score = %s", tuple(batch["x"].shape), tuple(out["score"].shape))
            if step % 200 == 0:
                log.info("epoch %d step %d/%d | %s", epoch, step, len(train_loader), parts)
        metrics = {"epoch": epoch, "seconds": time.time() - start,
                   **{f"train/{k}": v / steps for k, v in sums.items()}, **validate(model, val, device)}
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics) + "\n")
        log.info("epoch %d | %s", epoch, {k: round(v, 4) for k, v in metrics.items()})
        recent.append((metrics, trainable_state(model)))
        window_mean = sum(m["select"] for m, _ in recent) / len(recent)
        if len(recent) == select_window and window_mean < best_mean:
            best_mean, (best, state) = window_mean, recent[select_window // 2]
            best["select_window_mean"] = window_mean
            torch.save(state, run_dir / "model.pt")
            log.info("kept epoch %d | window mean %.4f", best["epoch"], window_mean)
        elif epoch - best["epoch"] >= patience:
            log.info("early stop: no improvement for %d epochs", patience)
            break
    load_trainable_state(model, torch.load(run_dir / "model.pt"))
    return best
