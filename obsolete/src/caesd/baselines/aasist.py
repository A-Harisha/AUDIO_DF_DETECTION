"""Pretrained AASIST wrapper for Phase 0 (no training).

AASIST reaches 1.89% EER in domain and 41.85% on ML-ITW; it anchors our protocol to known
numbers. The model code and weights are not vendored -- point ``repo_dir`` at a clone of the
official AASIST repository and ``checkpoint`` at its pretrained weights. If either is missing
the wrapper raises with instructions; it never substitutes an untrained stand-in silently.

Produces a detection score where higher => bona fide, matching this package's convention.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


class AasistScorer:
    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint: str | Path,
        device: str = "cuda",
        bonafide_index: int = 1,
        nb_samp: int = 64600,
    ) -> None:
        repo_dir = Path(repo_dir)
        checkpoint = Path(checkpoint)
        if not repo_dir.exists():
            raise FileNotFoundError(f"AASIST repo not found at {repo_dir}; clone github.com/clovaai/aasist")
        if not checkpoint.exists():
            raise FileNotFoundError(f"AASIST checkpoint not found at {checkpoint}")
        self.device = device
        self.bonafide_index = bonafide_index  # AASIST uses batch_out[:, 1] as the bona-fide score
        self.nb_samp = nb_samp                 # AASIST is trained on exactly 64600 samples
        self.model = self._load(repo_dir, checkpoint).to(device).eval()

    def fix_length(self, wave: torch.Tensor) -> torch.Tensor:
        """Crop or tile each row of a [B, T] batch to nb_samp (tile, not zero-pad, like AASIST)."""
        t = wave.shape[-1]
        if t >= self.nb_samp:
            return wave[..., : self.nb_samp]
        repeats = self.nb_samp // t + 1
        return wave.repeat(1, repeats)[..., : self.nb_samp]

    def batch_from_waves(self, waves: list[np.ndarray]) -> torch.Tensor:
        """Tile/crop each 1-D clip to nb_samp and stack into a [B, nb_samp] tensor."""
        fixed = []
        for w in waves:
            w = np.asarray(w, dtype=np.float32)
            t = w.shape[0]
            w = w[: self.nb_samp] if t >= self.nb_samp else np.tile(w, self.nb_samp // t + 1)[: self.nb_samp]
            fixed.append(torch.from_numpy(w))
        return torch.stack(fixed)

    @staticmethod
    def _load(repo_dir: Path, checkpoint: Path) -> torch.nn.Module:
        spec = importlib.util.spec_from_file_location("aasist_model", repo_dir / "models" / "AASIST.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["aasist_model"] = module
        spec.loader.exec_module(module)
        import json

        config = json.loads((repo_dir / "config" / "AASIST.conf").read_text())
        model = module.Model(config["model_config"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        return model

    @torch.no_grad()
    def score(self, wave: torch.Tensor) -> np.ndarray:
        """wave [B, T] -> bona-fide score [B]. Input is length-fixed to nb_samp first."""
        _, logits = self.model(self.fix_length(wave).to(self.device))
        return logits[:, self.bonafide_index].cpu().numpy()
