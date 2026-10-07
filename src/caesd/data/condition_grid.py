"""The condition grid: codecs, noise, reverb and benign edits.

Built once with ffmpeg and open noise/IR corpora, then FROZEN after Phase 0. It supplies free
labels for the condition encoder (block A). Benign edits apply to real speech only and become
the third class (processed bona fide).

There are no silent fallbacks: an unsupported codec raises, rather than passing audio through
unchanged and mislabelling it.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils.io import TARGET_SR, read_audio, write_audio

# Simulated codecs, named and configured as in ASVspoof5.codec.config.csv (eval codecs C01-C10).
#   name: (encoder, container, sample rate Hz, bitrate kbps for CODEC_Q 1..5)
# mp3_wb is VBR in the csv (ranges 45-85 ... 220-260 kbps); we use the range midpoints.
# Not simulated, so truly unseen at eval: C04 encodec_wb, C07 mp3_encodec_wb, C11 devices.
CODECS = {
    "opus_wb": ("libopus", "ogg", 16000, (6, 12, 18, 24, 30)),
    "amr_wb": ("libvo_amrwbenc", "amr", 16000, (6.6, 8.85, 14.25, 18.25, 23.05)),
    "speex_wb": ("libspeex", "ogg", 16000, (5.75, 9.8, 16.8, 23.8, 34.2)),
    "mp3_wb": ("libmp3lame", "mp3", 16000, (65, 100, 135, 190, 240)),
    "m4a_wb": ("aac", "m4a", 16000, (16, 32, 64, 96, 128)),
    "opus_nb": ("libopus", "ogg", 8000, (4, 8, 12, 16, 20)),
    "amr_nb": ("libopencore_amrnb", "amr", 8000, (4.75, 6.7, 8.85, 10.2, 12.2)),
    "speex_nb": ("libspeex", "ogg", 8000, (3.95, 5.95, 11, 18.2, 24.6)),
}
# Eval-set codec id -> the simulated codec that matches it (for the lookup-table baseline).
EVAL_CODEC = {"C01": "opus_wb", "C02": "amr_wb", "C03": "speex_wb", "C05": "mp3_wb", "C06": "m4a_wb",
              "C08": "opus_nb", "C09": "amr_nb", "C10": "speex_nb"}


@dataclass(frozen=True)
class Condition:
    """A point in the grid. ``cond_id`` is the lookup-table cell key."""

    codec: str = "clean"
    bitrate_kbps: int | None = None
    snr_db: float | None = None
    reverb: bool = False
    benign_edit: str | None = None

    @property
    def cond_id(self) -> str:
        parts = [self.codec]
        if self.bitrate_kbps is not None:
            parts.append(f"{self.bitrate_kbps}k")
        if self.snr_db is not None:
            parts.append(f"snr{self.snr_db:g}")
        if self.reverb:
            parts.append("rev")
        if self.benign_edit:
            parts.append(self.benign_edit)
        return "_".join(parts)


def apply_codec(wave: np.ndarray, codec: str, quality: int, sr: int = TARGET_SR) -> np.ndarray:
    """Transcode through a lossy codec at CODEC_Q ``quality`` (1..5); decode to 16 kHz mono."""
    if codec == "clean":
        return wave
    encoder, container, codec_sr, bitrates = CODECS[codec]  # unknown codec raises KeyError
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.wav"
        enc = Path(tmp) / f"enc.{container}"
        dec = Path(tmp) / "dec.wav"
        write_audio(src, wave, sr)
        _run(["ffmpeg", "-y", "-i", str(src), "-ar", str(codec_sr), "-ac", "1", "-c:a", encoder,
              "-b:a", f"{bitrates[quality - 1]}k", str(enc)])
        # Decode back with ffmpeg (libsndfile cannot read amr/m4a/speex), to 16 kHz mono wav.
        _run(["ffmpeg", "-y", "-i", str(enc), "-ar", str(sr), "-ac", "1", str(dec)])
        return read_audio(dec, sr)


def add_noise(wave: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix additive noise at a target SNR (dB)."""
    noise = _match_length(noise, wave.size)
    sig_power = np.mean(wave**2) + 1e-12
    noise_power = np.mean(noise**2) + 1e-12
    scale = np.sqrt(sig_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    return wave + scale * noise


def apply_reverb(wave: np.ndarray, impulse_response: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    wet = fftconvolve(wave, impulse_response)[: wave.size]
    peak = np.max(np.abs(wet)) + 1e-12
    return wet / peak * (np.max(np.abs(wave)) + 1e-12)


def _match_length(noise: np.ndarray, length: int) -> np.ndarray:
    if noise.size >= length:
        return noise[:length]
    reps = int(np.ceil(length / noise.size))
    return np.tile(noise, reps)[:length]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


@dataclass
class ConditionGrid:
    """The full, frozen set of conditions. Loaded from configs/condition_grid/grid.yaml."""

    codecs: list[str] = field(default_factory=lambda: ["clean", "mp3", "opus", "amr", "speex8k", "aac"])
    bitrates_kbps: list[int] = field(default_factory=lambda: [16, 32])
    snr_db: list[float] = field(default_factory=lambda: [20.0, 10.0, 0.0])
    reverb: list[bool] = field(default_factory=lambda: [False, True])
    benign_edits: list[str] = field(default_factory=lambda: ["denoise", "restoration"])

    def enumerate_conditions(self) -> list[Condition]:
        """All codec x bitrate x snr x reverb cells (benign edits are applied to real speech only)."""
        conditions: list[Condition] = []
        for codec in self.codecs:
            bitrates = self.bitrates_kbps if codec not in ("clean",) else [None]
            for bitrate in bitrates:
                for snr in [None, *self.snr_db]:
                    for rev in self.reverb:
                        conditions.append(
                            Condition(codec=codec, bitrate_kbps=bitrate, snr_db=snr, reverb=rev)
                        )
        return conditions
