"""The ``encodec`` scoring view is a SELECTION set, so the two things that would make it useless are
asserted: a bandwidth that is not fixed per clip (then the view is an augmentation and a model
selected on it was selected on noise), and a channel label that does not say which eval condition a
clip stands for.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from caesd.data import condition_grid as cg
from caesd.data.datasets import CODEC_NAMES

MP3 = CODEC_NAMES.index("mp3_wb")
ENC = CODEC_NAMES.index("encodec_wb")
MP3_ENC = CODEC_NAMES.index("mp3_encodec_wb")


@pytest.fixture
def marker(monkeypatch):
    """Stand in for EnCodec: encodes the bandwidth into the output so it can be read back."""
    monkeypatch.setattr(cg, "encodec", lambda wave, kbps, sr=16000: torch.full_like(wave, float(kbps)))


def test_the_bandwidth_is_fixed_per_clip_not_per_call(marker) -> None:
    wave = torch.randn(4, 100)
    channel = torch.zeros(4, dtype=torch.long)
    idx = np.array([7, 1, 9, 3])
    first, _ = cg.encodec_view(wave, None, channel, idx)
    second, _ = cg.encodec_view(wave, None, channel, idx)
    assert torch.equal(first, second), "the view changed between calls; it is an augmentation"
    # and each clip's rate follows its own index, so reordering the batch follows the clips
    order = [2, 0, 3, 1]
    shuffled, _ = cg.encodec_view(wave[order], None, channel[order], idx[order])
    assert torch.allclose(shuffled, first[order])


def test_every_rate_used_is_one_of_the_model_bandwidths(marker) -> None:
    wave = torch.randn(40, 20)
    out, _ = cg.encodec_view(wave, None, torch.zeros(40, dtype=torch.long), np.arange(40))
    used = {float(v) for v in out[:, 0].tolist()}
    assert used <= set(float(k) for k in cg.ENCODEC_KBPS), used
    assert len(used) > 1, "a single rate for 40 clips means the index is being ignored"


def test_the_channel_says_which_eval_condition_a_clip_stands_for(marker) -> None:
    wave = torch.randn(4, 50)
    channel = torch.tensor([0, MP3, 0, MP3])
    _, relabelled = cg.encodec_view(wave, None, channel, np.arange(4))
    # a clip the workers put through MP3 first is eval C07; a clean one is C04
    assert relabelled.tolist() == [ENC, MP3_ENC, ENC, MP3_ENC]


def test_the_padding_stays_zero(marker) -> None:
    wave = torch.randn(2, 100)
    wave[1, 60:] = 0.0
    out, _ = cg.encodec_view(wave, torch.tensor([100, 60]), torch.zeros(2, dtype=torch.long),
                             np.arange(2))
    assert torch.all(out[1, 60:] == 0.0), "EnCodec decoded the padding into the clip"
    assert torch.all(out[0] != 0.0)
