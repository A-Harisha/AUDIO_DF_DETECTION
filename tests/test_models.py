"""Forward-pass smoke tests for every block, offline, on random tensors.

Uses Backbone.tiny() so nothing is downloaded. Checks shape contracts, not learned behaviour.
"""
import numpy as np
import torch

from caesd.metrics import cllr
from caesd.models import (
    Backbone,
    CaesdDetector,
    CalibrationHypernet,
    ConditionEncoder,
    FiLM,
    Head,
    MondrianConformalGate,
    SparseAutoencoder,
)

B, D, C = 2, 32, 16


def test_backbone_returns_every_layer_and_ignores_padding():
    backbone = Backbone.tiny(D).eval()
    wave = torch.randn(B, 16000)
    lengths = torch.tensor([16000, 8000])
    pooled = backbone(wave, lengths)
    assert pooled.shape == (B, backbone.n_layers, D)
    # the second clip alone, with no padding, must give the same features
    alone = backbone(wave[1:, :8000], lengths[1:])
    assert torch.allclose(pooled[1], alone[0], atol=1e-4)


def test_condition_encoder_film_and_sae():
    out = ConditionEncoder(D, condition_dim=C)(torch.randn(B, D))
    assert out["condition"].shape == (B, C)
    assert FiLM(D, C)(torch.randn(B, D), torch.randn(B, C)).shape == (B, D)
    sae = SparseAutoencoder(D, n_latents=8)
    out = sae(torch.randn(B, D))
    assert out["recon"].shape == (B, D) and out["code"].shape == (B, 8)


def test_detector_on_audio_and_on_cached_features():
    backbone = Backbone.tiny(D)
    model = CaesdDetector(backbone, Head(backbone.n_layers, D, n_channels=5))
    out = model(torch.randn(B, 16000), torch.tensor([16000, 12000]))
    assert out["score"].shape == (B,) and out["channel_logits"].shape == (B, 5)
    cached = CaesdDetector(None, Head(backbone.n_layers, D))(torch.randn(B, backbone.n_layers, D))
    assert cached["score"].shape == (B,) and cached["channel_logits"] is None


def test_hypernet_learns_a_condition_dependent_shift():
    """Two conditions with opposite score offsets: one global calibration cannot fit both."""
    rng = np.random.default_rng(0)
    n = 2000
    label = rng.integers(0, 2, n)
    group = rng.integers(0, 2, n)
    score = rng.normal(np.where(label == 1, 2.0, -2.0), 1.0) + np.where(group == 1, 4.0, -4.0)
    condition = np.eye(2)[group]
    hyper = CalibrationHypernet(2)
    hyper.fit(score, condition, label, group, steps=400)
    assert cllr(hyper.transform(score, condition), label) < 0.5 < cllr(score, label)


def test_conformal_gate_abstains_on_unseen_group():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=200)
    labels = probs.argmax(1)
    groups = np.array(["en"] * 200)
    gate = MondrianConformalGate(alpha=0.1).fit(probs, labels, groups)
    pred = gate.predict(probs[:4], np.array(["en", "en", "te", "te"]))
    assert pred["covered"][0] and not pred["covered"][2]
    assert pred["abstain"][2]  # unseen group -> insufficient evidence
