"""The frame back-end exists to give the ensemble a member with a different VIEW, so the two things
that would silently destroy that are asserted: pooling over the padding, and the pooled path
changing underneath the existing runs.
"""
from __future__ import annotations

import pytest
import torch

from caesd.models import MHFA, Backbone, CaesdDetector

N_LAYERS, DIM, T = 5, 32, 11


def frames(batch: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch, N_LAYERS, T, DIM)
    mask = torch.ones(batch, T, dtype=torch.bool)
    mask[1, 7:] = False  # the second clip is padded
    mask[2, 4:] = False
    return x, mask


def test_output_shape_and_keys() -> None:
    x, mask = frames()
    out = MHFA(N_LAYERS, DIM, compression=8, n_heads=4)(x, mask)
    assert out["score"].shape == (3,)
    assert out["channel_logits"] is None


def test_the_padding_cannot_change_a_score() -> None:
    """A clip's score must depend on its own frames only. If the attention leaked onto the zero
    padding, a score would move when the batch's longest clip changed."""
    torch.manual_seed(0)
    model = MHFA(N_LAYERS, DIM, compression=8, n_heads=4).eval()
    x, mask = frames()
    with torch.no_grad():
        before = model(x, mask)["score"]
        noisy = x.clone()
        noisy[1, :, 7:] = 99.0  # garbage strictly inside the padded region
        noisy[2, :, 4:] = -42.0
        after = model(noisy, mask)["score"]
    assert torch.allclose(before, after, atol=1e-5), "the back-end read the padding"


def test_a_padded_clip_scores_the_same_alone_as_in_a_batch() -> None:
    torch.manual_seed(0)
    model = MHFA(N_LAYERS, DIM, compression=8, n_heads=4).eval()
    x, mask = frames()
    with torch.no_grad():
        batched = model(x, mask)["score"]
        alone = torch.cat([model(x[k : k + 1, :, : int(mask[k].sum())],
                                mask[k : k + 1, : int(mask[k].sum())])["score"] for k in range(3)])
    assert torch.allclose(batched, alone, atol=1e-5)


def test_two_layer_weightings_are_independent() -> None:
    """Factorized means the layers that say WHERE to look are not forced to be the layers that
    carry the evidence. One shared weighting would collapse that back to the old head."""
    model = MHFA(N_LAYERS, DIM)
    assert model.key_logits is not model.value_logits
    with torch.no_grad():
        model.key_logits[0] = 5.0
    assert not torch.allclose(model.key_logits, model.value_logits)


def test_the_pooled_backbone_path_is_unchanged() -> None:
    """Existing runs must be byte-identical: pooled=True returns a bare tensor, not a tuple."""
    backbone = Backbone.tiny(feature_dim=DIM)
    wave, lengths = torch.randn(2, 4000), torch.tensor([4000, 2500])
    out = backbone(wave, lengths)
    assert isinstance(out, torch.Tensor) and out.shape == (2, backbone.n_layers, DIM)


def test_the_unpooled_backbone_returns_frames_and_a_mask() -> None:
    backbone = Backbone.tiny(feature_dim=DIM)
    backbone.pooled = False
    wave, lengths = torch.randn(2, 4000), torch.tensor([4000, 2500])
    out = backbone(wave, lengths)
    assert isinstance(out, tuple) and len(out) == 2
    stack, mask = out
    assert stack.shape[:2] == (2, backbone.n_layers) and stack.shape[3] == DIM
    assert mask.shape == (2, stack.shape[2]) and mask.dtype == torch.bool
    assert mask[0].sum() > mask[1].sum(), "the shorter clip kept the same number of valid frames"
    # the detector must unpack the tuple for the frame back-end
    model = CaesdDetector(backbone, MHFA(backbone.n_layers, DIM, compression=8, n_heads=2))
    assert model(wave, lengths)["score"].shape == (2,)
