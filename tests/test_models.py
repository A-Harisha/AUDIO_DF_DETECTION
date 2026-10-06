"""Forward-pass smoke tests for every block, offline, on random tensors.

Uses Backbone.tiny() so nothing is downloaded. Checks shape contracts, not learned behaviour.
"""
import numpy as np
import torch

from caesd.models import (
    Backbone,
    CaesdDetector,
    CalibrationHypernet,
    ConditionEncoder,
    FiLM,
    MondrianConformalGate,
    SparseAutoencoder,
)

B, D, C = 2, 32, 16


def test_backbone_tiny_forward():
    feats = Backbone.tiny(D)(torch.randn(B, 16000))
    assert feats.ndim == 3 and feats.shape[0] == B and feats.shape[-1] == D


def test_condition_encoder_shapes():
    out = ConditionEncoder(D, condition_dim=C)(torch.randn(B, D))
    assert out["condition"].shape == (B, C)
    assert out["channel_logits"].shape[0] == B and out["snr"].shape == (B,)


def test_film_and_sae_and_hypernet():
    assert FiLM(D, C)(torch.randn(B, D), torch.randn(B, C)).shape == (B, D)
    sae = SparseAutoencoder(D, n_latents=8)
    out = sae(torch.randn(B, D))
    assert out["recon"].shape == (B, D) and out["code"].shape == (B, 8)
    assert sae.loss(torch.randn(B, D), out).ndim == 0
    llr = CalibrationHypernet(C).calibrate(torch.randn(B), torch.randn(B, C))
    assert llr.shape == (B,)


def test_detector_end_to_end():
    model = CaesdDetector(Backbone.tiny(D), condition_dim=C)
    out = model(torch.randn(B, 16000))
    assert out.class_logits.shape == (B, 3) and out.raw_score.shape == (B,)
    assert model.calibrated_llr(out).shape == (B,)


def test_conformal_gate_abstains_on_unseen_group():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=200)
    labels = probs.argmax(1)
    groups = np.array(["en"] * 200)
    gate = MondrianConformalGate(alpha=0.1).fit(probs, labels, groups)
    pred = gate.predict(probs[:4], np.array(["en", "en", "te", "te"]))
    assert pred["covered"][0] and not pred["covered"][2]
    assert pred["abstain"][2]  # unseen group -> insufficient evidence
