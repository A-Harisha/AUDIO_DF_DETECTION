"""The epoch-selection rule of ``fit``, on scripted validation curves.

Claim served: the kept epoch must not be the argmin of the per-epoch noise. Seed 2 of
frozen_wavlm_enc selected epoch 1, an undertrained head, because its noisy select EER happened to
be the global minimum (docs/experiment_log.md, 2026-10-09).
"""
import json

import pytest
import torch

from caesd.models import CaesdDetector, Head
from caesd.train import loop


@pytest.fixture
def run(monkeypatch):
    """One ``fit`` whose validation returns a scripted curve epoch by epoch."""
    def go(curve: list[float], tmp_path, select_window: int) -> dict:
        model = CaesdDetector(None, Head(2, 4, dropout=0.0))
        batch = {"x": torch.randn(4, 2, 4), "label": torch.tensor([0, 1, 0, 1]),
                 "channel": torch.zeros(4, dtype=torch.long), "lengths": None}
        seen = iter(curve)
        monkeypatch.setattr(loop, "validate", lambda *a, **k: {"select": next(seen)})
        return loop.fit(model, [batch], {}, tmp_path, epochs=len(curve), lr=1e-3, backbone_lr=1e-5,
                        weight_decay=0.0, patience=len(curve), cond_weight=0.0, device="cpu",
                        select_window=select_window)
    return go


def test_window_one_reproduces_the_old_argmin(run, tmp_path):
    """select_window=1 must keep the old rule, so earlier runs stay explainable."""
    assert run([0.5, 0.02, 0.4, 0.3], tmp_path, select_window=1)["epoch"] == 1


def test_an_isolated_dip_is_not_selected(run, tmp_path):
    """The seed 2 failure: one lucky epoch inside an undertrained region."""
    best = run([0.5, 0.02, 0.45, 0.2, 0.19, 0.21, 0.2], tmp_path, select_window=3)
    assert best["epoch"] != 1
    assert best["epoch"] in (3, 4, 5)


def test_the_kept_weights_are_one_epoch_not_an_average(run, tmp_path):
    """EMA was rejected on held-out dev, so the checkpoint must be a single epoch's weights."""
    best = run([0.4, 0.3, 0.2, 0.1, 0.1, 0.1], tmp_path, select_window=3)
    state = torch.load(tmp_path / "model.pt")
    # Exactly the trainable parameters plus the buffers: no optimiser state, no averaged copy, and
    # nothing missing. Buffers are in there because a from-scratch front end keeps its batch-norm
    # running statistics in them, and a reload without those scores a different model.
    reference = CaesdDetector(None, Head(2, 4))
    assert set(state) == ({n for n, p in reference.named_parameters() if p.requires_grad}
                          | {n for n, _ in reference.named_buffers()})
    assert best["select"] in (0.1, 0.2)


def test_metrics_are_still_logged_every_epoch(run, tmp_path):
    run([0.4, 0.3, 0.2, 0.1], tmp_path, select_window=3)
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["epoch"] for ln in lines] == [0, 1, 2, 3]


def test_a_window_wider_than_training_is_refused(run, tmp_path):
    with pytest.raises(ValueError, match="select_window"):
        run([0.4, 0.3], tmp_path, select_window=3)
