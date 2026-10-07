"""Torch dataset over a manifest. Returns mono 16 kHz waveforms plus the index and codec label."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.io import TARGET_SR, read_audio
from .condition_grid import CODECS

CODEC_NAMES = ("clean", *CODECS)  # channel-class index used everywhere


class AudioDataset(Dataset):
    """``rawboost`` and ``codec`` are optional augmentations (RawBoost first, then the codec, as a
    recording is degraded before it is transmitted). ``train`` crops at a random position."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        max_seconds: float,
        rawboost=None,
        codec=None,
        train: bool = False,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.max_len = int(max_seconds * TARGET_SR)
        self.rawboost = rawboost
        self.codec = codec
        self.train = train

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        wave = read_audio(row["path"], offset=int(row["offset"]), size=int(row["size"]))
        if wave.size > self.max_len:
            start = np.random.randint(0, wave.size - self.max_len + 1) if self.train else 0
            wave = wave[start : start + self.max_len]
        channel = "clean"
        if not self.train:
            np.random.seed(idx)  # fixed augmentation per clip: evaluation views are reproducible
        if self.rawboost is not None:
            wave = self.rawboost(wave, TARGET_SR)
        if self.codec is not None:
            wave, channel = self.codec(wave, TARGET_SR)
            wave = wave[: self.max_len]
        return {"wave": torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)), "idx": idx,
                "label": int(row["label"]), "channel": CODEC_NAMES.index(channel)}


def collate(items: list[dict]) -> dict:
    """Zero-pad to the longest clip in the batch; ``lengths`` lets the backbone mask the padding."""
    lengths = torch.tensor([it["wave"].numel() for it in items])
    wave = torch.zeros(len(items), int(lengths.max()))
    for i, it in enumerate(items):
        wave[i, : lengths[i]] = it["wave"]
    return {"x": wave, "lengths": lengths, **{k: torch.tensor([it[k] for it in items])
                                              for k in ("idx", "label", "channel")}}


def seed_worker(worker_id: int) -> None:
    """Distinct augmentation RNG per DataLoader worker. Module-level so spawn can pickle it."""
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


class FeatureLoader:
    """Batches of cached pooled layers [N, L, D] held in RAM. Same batch keys as ``collate``."""

    def __init__(self, x: torch.Tensor, label: np.ndarray, channel: np.ndarray, batch_size: int,
                 shuffle: bool = False) -> None:
        self.x, self.batch_size, self.shuffle = x, batch_size, shuffle
        self.label, self.channel = torch.as_tensor(label), torch.as_tensor(channel)

    def __len__(self) -> int:
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(len(self.x)) if self.shuffle else torch.arange(len(self.x))
        for i in range(0, len(order), self.batch_size):
            idx = order[i : i + self.batch_size]
            yield {"x": self.x[idx], "lengths": None, "idx": idx, "label": self.label[idx],
                   "channel": self.channel[idx]}
