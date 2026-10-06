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
    ) -> None:
        repo_dir = Path(repo_dir)
        checkpoint = Path(checkpoint)
        if not repo_dir.exists():
            raise FileNotFoundError(f"AASIST repo not found at {repo_dir}; clone github.com/clovaai/aasist")
        if not checkpoint.exists():
            raise FileNotFoundError(f"AASIST checkpoint not found at {checkpoint}")
        self.device = device
        self.bonafide_index = bonafide_index
        self.model = self._load(repo_dir, checkpoint).to(device).eval()

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
        """wave [B, T] -> bona-fide score [B]."""
        _, logits = self.model(wave.to(self.device))
        return logits[:, self.bonafide_index].cpu().numpy()
