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
# These eight are the ``codec`` view. C11 (devices) is never simulated.
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
# Neural codecs (eval C04 encodec_wb, C07 mp3_encodec_wb): the ``encodec`` view. ``apply_codec`` does
# not know them; the caller runs ``encodec`` on the GPU (after ``mp3_wb`` for C07). On eval they are
# the hardest conditions for a detector that never saw them (EER 8-17% against 1.5-4% on clean;
# docs/experiment_log.md, 2026-10-08). The csv bitrates are exactly the bandwidths of the public
# 24 kHz EnCodec model; that ASVspoof 5 used this model is an ASSUMPTION, not verified.
NEURAL_CODECS = ("encodec_wb", "mp3_encodec_wb")
ENCODEC_KBPS = (1.5, 3.0, 6.0, 12.0, 24.0)
_ENCODEC_SR = 24000
_encodec_model = None  # loaded once, on the device of the first ``encodec`` call
# Length of channel_descriptor's output: n_bands + the rolloff edge.
DESCRIPTOR_DIM = 25

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


def encodec(wave, kbps: float, sr: int = TARGET_SR):
    """Encode and decode a torch waveform with EnCodec (24 kHz model) at ``kbps``; back to ``sr``,
    same length, on the device of ``wave``. Takes [T] or a batch [B, T] of equal-length clips and
    returns the same shape. It runs in the main process on the GPU: on CPU in DataLoader workers it
    took minutes per batch and died with "bad allocation" (2026-10-09)."""
    global _encodec_model
    import torch
    import torchaudio.functional as AF

    if _encodec_model is None:
        from transformers import EncodecModel

        _encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").eval().to(wave.device)
    flat = wave.dim() == 1
    batch = wave[None] if flat else wave
    with torch.no_grad():
        y = _encodec_model(AF.resample(batch[:, None], sr, _ENCODEC_SR), bandwidth=kbps).audio_values
    out = AF.resample(y[:, 0], _ENCODEC_SR, sr)
    if out.shape[1] < batch.shape[1]:
        raise ValueError(f"EnCodec returned {out.shape[1]} samples for {batch.shape[1]}: "
                         "silently shortening a training crop would corrupt the batch")
    out = out[:, : batch.shape[1]]
    return out[0] if flat else out


def encodec_batch(wave, channel, p: float, lengths=None, sr: int = TARGET_SR):
    """Put a random ``p``-fraction of a batch [B, T] through EnCodec and relabel its channel.

    Why it exists: the eval set's two hardest conditions for a fine-tuned model are C04 and C07,
    the NEURAL codecs, and a features=false run never sees one -- ``CodecAugment`` runs in the
    DataLoader workers and EnCodec cannot (see above). A frozen run gets them from the cached
    ``encodec`` view, which is why the only frozen control that beats LoRA on eval is the one
    trained with that view (2026-10-17).

    The relabelling matches ``extract_features``: a clip the workers already put through MP3 becomes
    ``mp3_encodec_wb`` (eval C07), a clip that did not becomes ``encodec_wb`` (C04).

    ONE bandwidth per call, drawn uniformly, so the selected clips cost ONE EnCodec forward. A
    per-clip bandwidth would cost up to five and add about 30 min to every epoch; one per batch
    still gives each of the five rates roughly 1,100 batches per epoch, which is ample coverage.

    EnCodec's output depends on the batch SIZE: a clip encoded alone and the same clip encoded in a
    batch differ by 22-37 dB SNR, because a hair of numerical difference at the encoder flips a
    residual-quantiser code. There is no cross-talk between clips -- identical rows give identical
    output, and a clip is unaffected by loud neighbours -- and the difference sits 24 dB or more
    BELOW the degradation the codec itself applies, so both are valid realisations of the same
    codec. It does mean a batched crop is not bit-identical to the cached ``encodec`` view
    (2026-10-19).
    """
    import torch

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"encodec_p {p} is not a probability")
    names = ("clean", *CODECS, *NEURAL_CODECS)
    pick = torch.rand(wave.shape[0], device=wave.device) < p
    if not bool(pick.any()):
        return wave, channel
    idx = pick.nonzero(as_tuple=True)[0]
    rate = float(ENCODEC_KBPS[int(torch.randint(len(ENCODEC_KBPS), ()))])
    out = wave.clone()
    out[idx] = encodec(wave[idx], rate, sr).to(out.dtype)
    if lengths is not None:  # EnCodec also "decodes" the zero padding; put the zeros back
        keep = torch.arange(wave.shape[1], device=wave.device)[None] < lengths[:, None]
        out = out * keep
    new = channel.clone()
    new[idx] = torch.where(channel[idx] == names.index("mp3_wb"),
                           names.index("mp3_encodec_wb"), names.index("encodec_wb"))
    return out, new


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


def channel_descriptor(wave: np.ndarray, sr: int = TARGET_SR, n_bands: int = 24) -> np.ndarray:
    """Signal-level channel descriptor for block C: the long-term average spectrum in log-spaced
    bands (mean-removed, so loudness drops out) plus the 99% rolloff edge.

    It describes the CHANNEL -- bandwidth, quantisation noise floor -- and it never sees the SSL
    features the detector uses, which is what made v1's condition vector leak class evidence.
    Too coarse to resolve a vocoder's harmonic / phase artefact; ``models.label_probe_eer`` checks
    that claim rather than assuming it.  [n_bands + 1]
    """
    from scipy.signal import welch

    freq, power = welch(wave, fs=sr, nperseg=512)
    if not power.any():
        raise ValueError("silent clip has no channel descriptor")
    # The noise floor is clamped RELATIVE to the peak (100 dB) and the band means are read by
    # interpolation, not by averaging bins: both keep the descriptor exactly loudness-invariant,
    # which an absolute floor and empty low log-spaced bands would not.
    spectrum = np.log10(np.maximum(power, power.max() * 1e-10))
    log_power = np.interp(np.geomspace(50.0, sr / 2, n_bands), freq, spectrum)
    rolloff = freq[np.searchsorted(np.cumsum(power) / power.sum(), 0.99)] / (sr / 2)
    return np.append(log_power - log_power.mean(), rolloff).astype(np.float32)


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
