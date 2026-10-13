"""A raw-waveform front end, trained from scratch, with the same interface as ``Backbone``.

    waveform [B, T], lengths [B]  ->  pooled [B, L, D]   (L = n_stages, D = channels)

Why it exists. Fusion of frozen-SSL members saturated at 3.60% EER on 2026-10-13: four layer bands,
the full-stack head and a whole second SSL backbone all land within 0.07 points of each other, because
every member reads a time-mean of one pretrained transformer's layers. Cross-backbone correlation
(0.80-0.90) sat INSIDE the band-to-band range (0.59-0.94), so a second SSL model brought no diversity
the bands had not already given. This member shares no weights, no pretraining and no pooling grid
with any of them, so it is the largest decorrelation available in this design.

What it is, named honestly: a learnable band-pass filterbank (the SincNet / RawNet2 front end,
implemented here from the filter definition rather than copied) followed by residual 1-D convolution
stages. It is NOT AASIST: there is no graph-attention stack, and no result from it may be reported
under that name.

Each stage emits the same channel count, so the stack is shaped exactly like an SSL layer stack and
the existing ``Head`` -- learned stage weights, optional band, MLP -- works unchanged. That is also
what lets ``ScoreEnsemble`` put this member and an SSL member in one graph for a white-box attack.

Lengths are carried through every stride analytically and the time-mean is masked, so zero padding
never enters a pooled vector. Padding that leaked into the mean would make a clip's score depend on
its batch.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.io import TARGET_SR


def _conv_len(n: torch.Tensor, kernel: int, stride: int, padding: int) -> torch.Tensor:
    """Output length of a conv or pool with these settings: floor((n + 2p - k) / s) + 1."""
    return (torch.div(n + 2 * padding - kernel, stride, rounding_mode="floor") + 1).clamp(min=1)


def mel_edges(n_filters: int, sample_rate: int, low_hz: float = 30.0) -> torch.Tensor:
    """``n_filters + 1`` band edges, equally spaced on the mel scale: the standard initialisation,
    so the filterbank starts with the resolution of a mel front end and learns away from it."""
    high_hz = sample_rate / 2 - (low_hz + 1)
    to_mel = lambda hz: 2595.0 * math.log10(1.0 + hz / 700.0)  # noqa: E731
    to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)  # noqa: E731
    mel = torch.linspace(to_mel(low_hz), to_mel(high_hz), n_filters + 1)
    return to_hz(mel)


class SincConv(nn.Module):
    """Band-pass filterbank. Only the two cutoffs per filter are learned, so the kernel stays a
    band-pass by construction instead of drifting into an arbitrary filter."""

    def __init__(self, out_channels: int = 70, kernel_size: int = 129, sample_rate: int = TARGET_SR,
                 stride: int = 10) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size {kernel_size} must be odd: the filter is symmetric")
        self.sample_rate, self.kernel_size, self.stride = sample_rate, kernel_size, stride
        edges = mel_edges(out_channels, sample_rate)
        self.low_hz = nn.Parameter(edges[:-1].clone())
        self.band_hz = nn.Parameter((edges[1:] - edges[:-1]).clone())
        half = torch.arange(1, kernel_size // 2 + 1, dtype=torch.float32)
        self.register_buffer("n_half", half / sample_rate)
        # Hamming window on the left half; the filter is mirrored, so only half is needed.
        self.register_buffer("window", 0.54 - 0.46 * torch.cos(2 * math.pi * half / kernel_size))

    def filters(self) -> torch.Tensor:
        """[out_channels, 1, kernel_size]. Cutoffs are clamped into (0, Nyquist] and ordered, so a
        filter can never invert or leave the band."""
        low = self.low_hz.abs().clamp(min=1.0, max=self.sample_rate / 2 - 1.0)
        # low <= Nyquist - 1, so `low + 1` never exceeds Nyquist and the two bounds cannot cross.
        high = torch.maximum((low + self.band_hz.abs()).clamp(max=self.sample_rate / 2), low + 1.0)
        # Half of a band-pass impulse response: 2f2 sinc(2 pi f2 t) - 2f1 sinc(2 pi f1 t).
        t = self.n_half[None, :]
        left = (torch.sin(2 * math.pi * high[:, None] * t) - torch.sin(2 * math.pi * low[:, None] * t)) / t
        left = left * self.window[None, :] / self.sample_rate
        centre = 2 * (high - low)[:, None] / self.sample_rate
        kernel = torch.cat([left.flip(-1), centre, left], dim=-1)
        return (kernel / kernel.abs().amax(dim=-1, keepdim=True)).unsqueeze(1)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # A filterbank is computed at a frame rate, not at the sample rate: the hop cuts this layer
        # and every stage after it by ``stride``. At 16 kHz, stride 10 is a 0.625 ms hop, well above
        # the modulation rates a spoofing artefact lives at.
        return F.conv1d(wave.unsqueeze(1), self.filters(), stride=self.stride,
                        padding=self.kernel_size // 2)


class ResBlock(nn.Module):
    """Two convolutions with a skip, then a stride-``pool`` max-pool. Constant channel width, so
    every stage output is directly comparable and the stack is shaped like an SSL layer stack.

    ``valid`` [B, 1, T] marks the real frames and is re-applied after every normalisation. Without
    it the batch norm maps the zero padding to its own bias, the next convolution reads that bias at
    the boundary, and a clip's pooled vector then depends on whether a longer clip shared its batch
    (measured: 4e-2 on a 4.8 scale, concentrated in the last stage where 29 frames make one frame
    worth 3%). Masking the input alone does not fix it; the bias is reintroduced inside the block."""

    def __init__(self, channels: int, pool: int = 3) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.3)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)) * valid)
        h = self.conv2(self.act(self.norm2(h)) * valid)
        return self.pool((x + h) * valid)


class RawFrontEnd(nn.Module):
    """``Backbone``-compatible: exposes ``n_layers`` and ``feature_dim`` and returns a pooled stack,
    so ``CaesdDetector``, ``Head`` and ``ScoreEnsemble`` take it with no change."""

    def __init__(self, channels: int = 128, n_stages: int = 5, n_filters: int = 70,
                 kernel_size: int = 129, pool: int = 3, sample_rate: int = TARGET_SR,
                 sinc_stride: int = 10, first_pool: int = 2) -> None:
        super().__init__()
        self.sinc = SincConv(n_filters, kernel_size, sample_rate, stride=sinc_stride)
        self.first_pool = nn.MaxPool1d(first_pool)
        self.sinc_stride, self.first_pool_size = sinc_stride, first_pool
        self.project = nn.Conv1d(n_filters, channels, 1)
        self.stages = nn.ModuleList([ResBlock(channels, pool) for _ in range(n_stages)])
        self.feature_dim = channels
        self.n_layers = n_stages  # one pooled vector per stage; the Head learns their weights
        self.pool = pool

    @staticmethod
    def _valid(length: int, frames: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """[B, 1, length] mask of the real frames."""
        return (torch.arange(length, device=like.device)[None] < frames[:, None]).to(like.dtype)[:, None, :]

    def _pool_len(self, n: torch.Tensor) -> torch.Tensor:
        """Valid frames after one ``MaxPool1d(self.pool)``, by the layer's OWN formula. Using
        ``n // pool`` instead is off by one wherever the clip fills the tensor, and in the last
        stage (a few dozen frames) one frame is several percent of the mean -- measured as a 4e-2
        shift before this was fixed."""
        return _conv_len(n, self.pool, self.pool, 0)

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Per-utterance normalisation over the valid samples, as for the SSL front ends.
        mask = (torch.arange(wave.shape[1], device=wave.device)[None] < lengths[:, None]).to(wave.dtype)
        n = lengths[:, None].to(wave.dtype)
        mean = (wave * mask).sum(1, keepdim=True) / n
        var = (((wave - mean) * mask) ** 2).sum(1, keepdim=True) / n
        x = (wave - mean) * mask / torch.sqrt(var + 1e-7)

        x = self.project(self.first_pool(self.sinc(x).abs()))  # |.| : the filterbank's envelope
        # Each layer's own output-length formula; see _pool_len.
        frames = _conv_len(lengths, self.sinc.kernel_size, self.sinc_stride,
                           self.sinc.kernel_size // 2)
        frames = _conv_len(frames, self.first_pool_size, self.first_pool_size, 0)
        # ``project`` has a bias, so it maps the zero padding to that bias rather than to zero. Left
        # unmasked, the first stage's convolution reads it at the boundary and a clip's pooled vector
        # moves by 4e-2 depending on whether the batch had a longer clip in it.
        x = x * self._valid(x.shape[-1], frames, x)
        pooled = []
        for stage in self.stages:
            x = stage(x, self._valid(x.shape[-1], frames, x))
            frames = self._pool_len(frames)
            valid = self._valid(x.shape[-1], frames, x)
            x = x * valid
            pooled.append(x.sum(-1) / frames[:, None].to(x.dtype))
        return torch.stack(pooled, dim=1)
