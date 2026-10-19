"""``encodec_batch`` decides what fraction of training sees the neural codec and what channel label
it carries, so both are asserted. The real EnCodec model is not downloaded here: it is replaced by
a marker transform, which is enough to test the selection, the relabelling and the padding."""
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
    """Stand in for EnCodec: returns -1 everywhere, so a processed clip is unmistakable."""
    monkeypatch.setattr(cg, "encodec", lambda wave, kbps, sr=16000: torch.full_like(wave, -1.0))


def test_p_zero_changes_nothing(marker) -> None:
    wave = torch.randn(4, 100)
    channel = torch.zeros(4, dtype=torch.long)
    out, new = cg.encodec_batch(wave, channel, 0.0)
    assert torch.equal(out, wave) and torch.equal(new, channel)


def test_every_clip_is_processed_at_p_one_and_relabelled(marker) -> None:
    wave = torch.randn(4, 100)
    channel = torch.tensor([0, MP3, 0, MP3])
    out, new = cg.encodec_batch(wave, channel, 1.0)
    assert torch.all(out == -1.0)
    # a clip the workers already put through MP3 is eval C07, a clean one is C04
    assert new.tolist() == [ENC, MP3_ENC, ENC, MP3_ENC]


def test_untouched_clips_keep_their_wave_and_label(marker) -> None:
    torch.manual_seed(0)
    wave = torch.randn(200, 50)
    channel = torch.full((200,), MP3)
    out, new = cg.encodec_batch(wave, channel, 0.5)
    done = (out == -1.0).all(dim=1)
    assert 0.3 < done.float().mean() < 0.7, "the selected fraction is nowhere near p"
    assert torch.equal(out[~done], wave[~done])
    assert torch.all(new[done] == MP3_ENC) and torch.all(new[~done] == MP3)


def test_padding_stays_zero(marker) -> None:
    wave = torch.randn(2, 100)
    wave[1, 60:] = 0.0
    lengths = torch.tensor([100, 60])
    out, _ = cg.encodec_batch(wave, torch.zeros(2, dtype=torch.long), 1.0, lengths)
    assert torch.all(out[1, 60:] == 0.0), "EnCodec decoded the zero padding into the crop"


def test_a_bad_probability_is_rejected(marker) -> None:
    with pytest.raises(ValueError, match="probability"):
        cg.encodec_batch(torch.randn(2, 10), torch.zeros(2, dtype=torch.long), 1.5)


def test_the_selected_clips_cost_one_encodec_forward(monkeypatch) -> None:
    """A per-clip bandwidth would cost up to five forwards and about 30 min of every epoch."""
    calls = []

    def counting(wave, kbps, sr=16000):
        calls.append(kbps)
        return torch.full_like(wave, -1.0)

    monkeypatch.setattr(cg, "encodec", counting)
    torch.manual_seed(0)
    cg.encodec_batch(torch.randn(64, 100), torch.zeros(64, dtype=torch.long), 1.0)
    assert len(calls) == 1, f"{len(calls)} EnCodec forwards for one batch"
    assert calls[0] in cg.ENCODEC_KBPS
