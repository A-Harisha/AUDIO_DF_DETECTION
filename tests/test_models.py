"""Forward-pass smoke tests for every block, offline, on random tensors.

Uses Backbone.tiny() so nothing is downloaded. Checks shape contracts, not learned behaviour.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from caesd.calibration import LogisticCalibrator
from caesd.data.condition_grid import DESCRIPTOR_DIM, channel_descriptor
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
    label_probe_eer,
)

B, D, C = 2, 32, 16


def two_condition_scores(n: int = 2000, seed: int = 0):
    """Two channels whose scores need opposite shifts: one global calibration cannot fit both."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    group = rng.integers(0, 2, n)
    score = rng.normal(np.where(label == 1, 2.0, -2.0), 1.0) + np.where(group == 1, 4.0, -4.0)
    z = np.eye(2)[group] + rng.normal(0, 0.05, (n, 2))  # a descriptor that encodes the channel
    stop = np.arange(n) % 4 == 0                        # the held-out rows for step selection
    return score, z, label, group, stop


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
    score, z, label, group, stop = two_condition_scores()
    hyper = CalibrationHypernet(2, n_anchors=4)
    info = hyper.fit(score, z, label, group, stop, steps=400)
    assert 0 <= info["best_step"] < info["steps"]
    assert cllr(hyper.transform(score, z), label) < 0.5 < cllr(score, label)


def test_hypernet_gives_exactly_one_global_calibration_far_from_every_anchor():
    """The fix for the leave-one-codec-out divergence (Cllr 0.209 / 0.517 in v1): off the anchors'
    manifold every kernel weight vanishes, so an UNSEEN condition gets the plain global
    calibration instead of an arbitrary extrapolation."""
    score, z, label, group, stop = two_condition_scores()
    hyper = CalibrationHypernet(2, n_anchors=4)
    hyper.fit(score, z, label, group, stop, steps=400)

    far = np.full((3, 2), 50.0)  # far outside the fitted descriptors
    assert hyper.null_weight(far).min() > 0.99
    assert hyper.null_weight(z).mean() < 0.99  # it does act on the conditions it was fit on

    far_score = np.array([-1.0, 0.0, 1.0])
    a = float(F.softplus(hyper.a0 + 0.5413))
    assert np.allclose(hyper.transform(far_score, far), a * far_score + float(hyper.b0), atol=1e-4)


def test_hypernet_fallback_is_no_worse_than_a_global_calibration():
    """An unseen condition must COST nothing, not merely reach the fallback. The null anchor alone
    guarantees only that some global map is used; without training (a0, b0) on its own that map
    can be poor -- an unseen codec scored Cllr 1.39 against 0.56 for pooled logistic calibration
    at null weight 1.00. ``anchor_weight`` is what closes that gap.
    """
    score, z, label, group, stop = two_condition_scores()
    hyper = CalibrationHypernet(2, n_anchors=4)
    hyper.fit(score, z, label, group, stop, steps=600)

    unseen = np.full((len(score), 2), 50.0)  # a condition far from every anchor
    assert hyper.null_weight(unseen).min() > 0.99
    fallback = cllr(hyper.transform(score, unseen), label)
    pooled = cllr(LogisticCalibrator().fit(score, label).transform(score), label)
    assert fallback < 1.15 * pooled, f"fallback Cllr {fallback:.3f} vs pooled {pooled:.3f}"


def test_hypernet_refuses_to_transform_before_fit():
    with pytest.raises(RuntimeError):
        CalibrationHypernet(2).transform(np.zeros(4), np.zeros((4, 2)))


def test_channel_descriptor_tracks_bandwidth_and_ignores_loudness():
    """Block C's condition input. It must describe the channel, not the level."""
    rng = np.random.default_rng(0)
    wide = rng.normal(0, 0.1, 16000)
    narrow = np.convolve(wide, np.ones(32) / 32, mode="same")  # low-pass: a narrowband channel
    z_wide, z_loud, z_narrow = (channel_descriptor(w) for w in (wide, 4.0 * wide, narrow))
    assert z_wide.shape == (DESCRIPTOR_DIM,)
    assert np.allclose(z_wide, z_loud, atol=1e-5)  # a gain change must not move the descriptor
    assert z_narrow[-1] < z_wide[-1]               # the rolloff edge moves down


def test_label_probe_eer_catches_a_leaking_descriptor():
    """The precondition for every block C claim. v1's condition vector came from the detector's
    own features and carried class evidence; this is the test that would have caught it."""
    rng = np.random.default_rng(0)
    n, half = 1000, 500
    label = rng.integers(0, 2, n)
    leaky = label[:, None] + rng.normal(0, 0.1, (n, 2))  # encodes the class
    clean = rng.normal(0, 1.0, (n, 2))                   # encodes nothing
    assert label_probe_eer(leaky[:half], label[:half], leaky[half:], label[half:]) < 0.1
    assert label_probe_eer(clean[:half], label[:half], clean[half:], label[half:]) > 0.4


def test_conformal_gate_abstains_on_unseen_group():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=200)
    labels = probs.argmax(1)
    groups = np.array(["en"] * 200)
    gate = MondrianConformalGate(alpha=0.1).fit(probs, labels, groups)
    pred = gate.predict(probs[:4], np.array(["en", "en", "te", "te"]))
    assert pred["covered"][0] and not pred["covered"][2]
    assert pred["abstain"][2]  # unseen group -> insufficient evidence
