"""Audio and file I/O. All audio enters the pipeline as mono float32 at TARGET_SR."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000


def read_audio(
    path: str | Path, target_sr: int = TARGET_SR, offset: int | None = None, size: int | None = None
) -> np.ndarray:
    """Read an audio file as mono float32 at ``target_sr``. Returns shape [T].

    With ``offset`` and ``size``, ``path`` is an uncompressed tar and the clip is the byte range
    inside it (see ``index_asvspoof5``). This gives random access with no extraction to disk.
    """
    source: str | io.BytesIO = str(path)
    if offset is not None:
        with open(path, "rb") as fh:
            fh.seek(offset)
            source = io.BytesIO(fh.read(size))
    wave, sr = sf.read(source, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)  # downmix to mono
    if sr != target_sr:
        wave = _resample(wave, sr, target_sr)
    return wave


def _resample(wave: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    import torch
    import torchaudio.functional as AF

    tensor = torch.from_numpy(wave).unsqueeze(0)
    out = AF.resample(tensor, sr, target_sr)
    return out.squeeze(0).numpy()


def write_audio(path: str | Path, wave: np.ndarray, sr: int = TARGET_SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave, sr)
