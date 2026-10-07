"""RawBoost-style augmentation for domain randomization over the condition grid.

Three data-independent operations from the RawBoost paper: a linear+nonlinear convolutive
filter, impulsive signal-dependent noise, and stationary signal-independent coloured noise.
Wide random sampling of these beats matching a fixed test set; RawBoost cut noisy EER from
16.24% to 2.55%.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .condition_grid import CODECS, apply_codec


def _rand(lo: float, hi: float) -> float:
    return lo + (hi - lo) * np.random.rand()


def _norm(wave: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(wave))
    return wave / peak if peak > 0 else wave


def convolutive_noise(wave: np.ndarray, sr: int, n_bands: int = 5, max_db: float = 10.0) -> np.ndarray:
    """Linear + nonlinear convolutive noise via a random cascade of notch filters."""
    out = np.zeros_like(wave)
    n_taps = int(0.01 * sr) | 1  # odd
    for order in range(1, n_bands + 1):
        f0 = _rand(0.05, 0.45)  # normalised frequency
        bw = _rand(0.01, 0.1)
        b = signal.firwin(n_taps, [max(f0 - bw, 1e-3), min(f0 + bw, 0.999)], pass_zero=False)
        gain = 10.0 ** (_rand(-max_db, max_db) / 20.0)
        out += gain * signal.lfilter(b, 1.0, wave) ** order
    return _norm(out + wave)


def impulsive_noise(wave: np.ndarray, prob: float = 0.001, max_db: float = 10.0) -> np.ndarray:
    """Impulsive, signal-dependent additive noise at random samples."""
    mask = np.random.rand(wave.size) < prob
    amp = 10.0 ** (_rand(0.0, max_db) / 20.0)
    noise = mask * amp * (2.0 * np.random.rand(wave.size) - 1.0) * np.abs(wave)
    return wave + noise


def coloured_noise(wave: np.ndarray, snr_db_range=(10.0, 40.0)) -> np.ndarray:
    """Stationary, signal-independent coloured additive noise."""
    white = np.random.randn(wave.size)
    b = signal.firwin(129, _rand(0.05, 0.9))
    colour = signal.lfilter(b, 1.0, white)
    snr = _rand(*snr_db_range)
    scale = np.sqrt(np.mean(wave**2) / (np.mean(colour**2) + 1e-12) / 10.0 ** (snr / 10.0))
    return wave + scale * colour


class RawBoost:
    """Random composition of the three operations. Default is the 'series' combination (algo 4)."""

    def __init__(self, p: float = 0.8) -> None:
        self.p = p

    def __call__(self, wave: np.ndarray, sr: int) -> np.ndarray:
        if np.random.rand() > self.p:
            return wave
        wave = convolutive_noise(wave, sr)
        wave = impulsive_noise(wave)
        wave = coloured_noise(wave)
        return _norm(wave).astype(np.float32)


class CodecAugment:
    """With probability ``p``, one random simulated codec at a random quality (else clean).
    Returns (wave, codec name)."""

    def __init__(self, p: float = 1.0, codecs: tuple[str, ...] = tuple(CODECS)) -> None:
        self.p = p
        self.codecs = codecs

    def __call__(self, wave: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
        if np.random.rand() >= self.p:
            return wave, "clean"
        codec = self.codecs[np.random.randint(len(self.codecs))]
        return apply_codec(wave, codec, int(np.random.randint(1, 6)), sr), codec
