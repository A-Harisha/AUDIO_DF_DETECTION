"""A run directory must reload into exactly the model that produced it.

The white-box attack builds its system from run directories, so if a reload differs from the
trained model the robustness numbers describe something else. The case that nearly broke this is a
trained-from-scratch front end: its batch-norm running statistics live in buffers, not parameters,
and the checkpoint used to hold parameters only.
"""
from __future__ import annotations

import json

import torch

from caesd.models import CaesdDetector, Head, RawFrontEnd
from caesd.models.ensemble import ScoreEnsemble
from caesd.train.loop import trainable_state

LENGTHS = torch.tensor([6400, 4800])
BACKBONE = {"name": "raw", "tag": "raw_sinc", "channels": 16, "n_stages": 3, "n_filters": 12,
            "kernel_size": 65}


def write_run(run_dir, detector: CaesdDetector) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "backbone": BACKBONE,
        "experiment": {"dropout": 0.2, "band": None, "film": False, "lora_rank": 0, "unfreeze": False},
    }), encoding="utf-8")
    torch.save(trainable_state(detector), run_dir / "model.pt")


def trained_detector() -> CaesdDetector:
    """A detector whose batch-norm running statistics are NOT at their defaults."""
    torch.manual_seed(0)
    front = RawFrontEnd(channels=BACKBONE["channels"], n_stages=BACKBONE["n_stages"],
                        n_filters=BACKBONE["n_filters"], kernel_size=BACKBONE["kernel_size"])
    detector = CaesdDetector(front, Head(front.n_layers, front.feature_dim, dropout=0.2))
    detector.train()
    for _ in range(3):  # a few forward passes in train mode move the running stats off 0 and 1
        detector(torch.randn(4, 6400) * 3.0, torch.full((4,), 6400))
    return detector.eval()


def test_a_raw_member_reloads_to_the_same_scores(tmp_path) -> None:
    detector = trained_detector()
    stats = detector.backbone.stages[0].norm1.running_mean
    assert stats.abs().max() > 1e-3, "the fixture did not move the batch-norm statistics"

    write_run(tmp_path / "raw_run", detector)
    wave = torch.randn(2, 6400)
    with torch.no_grad():
        before = detector(wave, LENGTHS)["score"]

    ensemble = ScoreEnsemble.from_runs(tmp_path, ["raw_run"], torch.zeros(1), torch.ones(1),
                                        torch.ones(1), device="cpu")
    with torch.no_grad():
        after = ensemble(wave, LENGTHS)["score"]
    assert torch.allclose(before, after, atol=1e-5), f"reload changed the scores: {before} vs {after}"


def test_the_attack_gets_a_waveform_gradient_through_a_raw_member(tmp_path) -> None:
    write_run(tmp_path / "raw_run", trained_detector())
    ensemble = ScoreEnsemble.from_runs(tmp_path, ["raw_run"], torch.zeros(1), torch.ones(1),
                                        torch.ones(1), device="cpu")
    wave = torch.randn(2, 6400, requires_grad=True)
    grad = torch.autograd.grad(ensemble.score_fn()(wave, LENGTHS).sum(), wave)[0]
    assert grad.norm() > 0
    assert all(not p.requires_grad for p in ensemble.parameters())
