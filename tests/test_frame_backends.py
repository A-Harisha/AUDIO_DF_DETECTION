"""Every frame back-end exists to give the ensemble a member with a different VIEW. Two faults would
silently destroy that, so both are asserted for each of them: reading the zero padding, and a score
that moves when the batch's longest clip changes.

The invariance tested is the one the pipeline can actually violate: the SAME clip, in batches padded
to DIFFERENT total lengths. Comparing against a waveform truncated to the clip's own length is not
that test -- a window-based feature cannot compute its final frame from a truncated signal, and the
pipeline never asks it to, because every batch is zero-padded to its longest member.

The LFCC hybrid carries a third fault of its own: its cepstra must land frame-for-frame on the
backbone's time grid, since a stretch would shift its second representation against the first with
no error raised.
"""
from __future__ import annotations

import pytest
import torch

from caesd.models import MHFA, LayerTimeGraph, LFCCHybrid

N_LAYERS, DIM = 5, 64
HOP = 320  # the LFCC hop, which is also the backbone's frame stride


def make(total: int, valid: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch padded to ``total`` frames, with each clip valid for its own number of frames."""
    torch.manual_seed(0)
    frames = torch.randn(len(valid), N_LAYERS, 64, DIM)[:, :, :total].contiguous()
    mask = torch.zeros(len(valid), total, dtype=torch.bool)
    for k, n in enumerate(valid):
        mask[k, :n] = True
    torch.manual_seed(1)
    wave = torch.randn(len(valid), 64 * HOP)[:, : total * HOP].contiguous()
    for k, n in enumerate(valid):
        wave[k, n * HOP :] = 0.0  # a real batch is zero-padded past each clip
    return frames, mask, wave


VALID = (11, 7, 4)


def heads():
    torch.manual_seed(0)
    return [
        ("mhfa", MHFA(N_LAYERS, DIM, compression=16, n_heads=2)),
        ("lfcc", LFCCHybrid(N_LAYERS, DIM, compression=16, n_heads=2)),
        ("graph", LayerTimeGraph(N_LAYERS, DIM, compression=16, n_time=4)),
    ]


@pytest.mark.parametrize("name,head", heads())
def test_output_shape_and_keys(name, head) -> None:
    frames, mask, wave = make(11, VALID)
    out = head(frames, mask, wave=wave, lengths=None)
    assert out["score"].shape == (3,), name
    assert out["channel_logits"] is None


@pytest.mark.parametrize("name,head", heads())
def test_the_frame_padding_cannot_change_a_score(name, head) -> None:
    head.eval()
    frames, mask, wave = make(11, VALID)
    with torch.no_grad():
        before = head(frames, mask, wave=wave, lengths=None)["score"]
        noisy = frames.clone()
        for k, n in enumerate(VALID):
            noisy[k, :, n:] = 99.0 if k % 2 else -42.0
        after = head(noisy, mask, wave=wave, lengths=None)["score"]
    assert torch.allclose(before, after, atol=1e-5), f"{name} read the frame padding"


@pytest.mark.parametrize("name,head", heads())
def test_a_score_does_not_move_when_the_batch_gets_longer(name, head) -> None:
    """The real invariance: pad the same clips to 11 frames and to 30, and the scores must agree.
    BatchNorm in the graph back-end makes eval mode part of what this checks."""
    head.eval()
    with torch.no_grad():
        short = head(*make(11, VALID)[:2], wave=make(11, VALID)[2], lengths=None)["score"]
        long_frames, long_mask, long_wave = make(30, VALID)
        longer = head(long_frames, long_mask, wave=long_wave, lengths=None)["score"]
    assert torch.allclose(short, longer, atol=1e-5), (
        f"{name} depends on the batch's longest clip: {short.tolist()} vs {longer.tolist()}")


def test_lfcc_lands_frame_for_frame_on_the_backbone_grid() -> None:
    head = LFCCHybrid(N_LAYERS, DIM, compression=16, n_heads=2).eval()
    _, mask, wave = make(11, VALID)
    with torch.no_grad():
        cep = head.cepstra(wave, mask)
    assert cep.shape == (3, 60, 11), cep.shape
    # zeroed BEFORE the Conv1d, so its kernel cannot drag padding inwards
    for k, n in enumerate(VALID):
        assert torch.all(cep[k, :, n:] == 0.0)


def test_lfcc_refuses_a_grid_it_cannot_align_to() -> None:
    """A hop that no longer matches the backbone stride must raise, not be resampled quietly."""
    head = LFCCHybrid(N_LAYERS, DIM, compression=16, n_heads=2).eval()
    wave = torch.randn(2, 11 * HOP)
    with pytest.raises(ValueError, match="rounding"):
        head.cepstra(wave, torch.ones(2, 40, dtype=torch.bool))


def test_lfcc_refuses_to_run_without_the_waveform() -> None:
    head = LFCCHybrid(N_LAYERS, DIM, compression=16, n_heads=2)
    frames, mask, _ = make(11, VALID)
    with pytest.raises(ValueError, match="waveform"):
        head(frames, mask, wave=None, lengths=None)


def test_the_graph_builds_one_node_per_layer_and_n_time_slots() -> None:
    """The node sets are the whole design: layers are a real second axis of this representation,
    standing in for the frequency axis SSL frames do not have."""
    head = LayerTimeGraph(N_LAYERS, DIM, compression=16, n_time=4).eval()
    frames, mask, _ = make(11, VALID)
    captured: dict[str, tuple] = {}
    for key, graph in (("time", head.time_graph), ("layer", head.layer_graph)):
        graph.register_forward_pre_hook(
            lambda _m, args, key=key: captured.__setitem__(key, tuple(args[0].shape)))
    with torch.no_grad():
        head(frames, mask, wave=None, lengths=None)
    assert captured["time"] == (3, 4, 16), captured["time"]
    assert captured["layer"] == (3, N_LAYERS, 16), captured["layer"]
