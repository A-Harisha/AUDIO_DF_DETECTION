"""Known-answer tests for the raw-waveform front end.

The two that matter are invariance properties. A clip's pooled vector must not depend on how long
the other clips in its batch are, and no gradient may reach the zero padding. Either failure would
make every score and every attack measurement depend on batch composition.
"""
from __future__ import annotations

import torch

from caesd.models import CaesdDetector, Head, RawFrontEnd
from caesd.models.raw_frontend import SincConv, mel_edges

LENGTHS = torch.tensor([64000, 48000, 32000])


def batch() -> torch.Tensor:
    torch.manual_seed(0)
    wave = torch.randn(3, 64000)
    for i, n in enumerate(LENGTHS.tolist()):
        wave[i, n:] = 0.0
    return wave


def small() -> RawFrontEnd:
    return RawFrontEnd(channels=16, n_stages=3, n_filters=12, kernel_size=65).eval()


def test_shape_follows_the_backbone_interface() -> None:
    """``Head`` is built from ``n_layers`` and ``feature_dim``, so they must describe the output."""
    model = small()
    pooled = model(batch(), LENGTHS)
    assert pooled.shape == (3, model.n_layers, model.feature_dim)


def test_pooled_vector_does_not_depend_on_the_batch() -> None:
    model = small()
    wave = batch()
    pooled = model(wave, LENGTHS)
    for extra in (8000, 32000):  # a longer clip joins the batch
        padded = torch.cat([wave, torch.zeros(3, extra)], dim=1)
        assert (pooled - model(padded, LENGTHS)).abs().max() < 1e-4
    for i, n in enumerate(LENGTHS.tolist()):  # the clip scored alone
        alone = model(wave[i : i + 1, :n], LENGTHS[i : i + 1])
        assert (alone - pooled[i : i + 1]).abs().max() < 1e-4


def test_no_gradient_reaches_the_padding() -> None:
    model = small()
    wave = batch().requires_grad_(True)
    detector = CaesdDetector(model, Head(model.n_layers, model.feature_dim, hidden=8)).eval()
    grad = torch.autograd.grad(detector(wave, LENGTHS)["score"].sum(), wave)[0]
    assert grad.norm() > 0  # the attack needs a gradient at all
    for i, n in enumerate(LENGTHS.tolist()):
        padding = grad[i, n:]  # empty for the clip that fills the batch tensor
        assert padding.numel() == 0 or padding.abs().max() == 0.0


def test_sinc_filters_stay_band_passes() -> None:
    """Only the cutoffs are learned, so even absurd parameter values must leave a valid band."""
    sinc = SincConv(out_channels=8, kernel_size=65)
    with torch.no_grad():
        sinc.low_hz.fill_(-50_000.0)   # negative and far past Nyquist
        sinc.band_hz.fill_(1e6)
    filters = sinc.filters()
    assert filters.shape == (8, 1, 65)
    assert torch.isfinite(filters).all()
    assert torch.allclose(filters.abs().amax(dim=-1), torch.ones(8, 1), atol=1e-6)


def test_mel_edges_are_ordered_and_inside_the_band() -> None:
    edges = mel_edges(32, 16000)
    assert edges.numel() == 33
    assert torch.all(edges[1:] > edges[:-1])
    assert edges[0] >= 30.0 and edges[-1] <= 8000.0


def test_odd_kernel_is_required() -> None:
    try:
        SincConv(out_channels=4, kernel_size=64)
    except ValueError as exc:
        assert "odd" in str(exc)
    else:
        raise AssertionError("an even kernel was accepted; the filter would not be symmetric")
