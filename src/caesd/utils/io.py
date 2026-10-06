"""Audio and file I/O. All audio enters the pipeline as mono float32 at TARGET_SR."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000


def read_audio(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read an audio file as mono float32 at ``target_sr``. Returns shape [T]."""
    wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
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
