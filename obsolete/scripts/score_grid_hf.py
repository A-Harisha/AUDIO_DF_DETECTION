"""Score our condition grid with a published transformers audio-classification model, as a
baseline under our own protocol (bona-fide score = logit of the 'real'/bona class).

    uv run python scripts/score_grid_hf.py +hf_model=MelodyMachine/Deepfake-audio-detection-V2 \
        +scores_in=... +scores_out=...

Loads plug-and-play (AutoModelForAudioClassification), so no head is guessed. The model may be
out-of-domain for ASVspoof 5; that is reported honestly, not hidden.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from caesd.utils.io import TARGET_SR, read_audio
from caesd.utils.logging import get_logger

log = get_logger("caesd.score.hf")

_BONA = {"real", "humanvoice", "bonafide", "genuine", "human"}


def _fix(w: np.ndarray, n: int) -> np.ndarray:
    return w[:n] if w.shape[0] >= n else np.tile(w, n // w.shape[0] + 1)[:n]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    fe = AutoFeatureExtractor.from_pretrained(cfg.hf_model)
    model = AutoModelForAudioClassification.from_pretrained(cfg.hf_model).eval().to(cfg.device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    bona_idx = next(i for i, name in id2label.items() if str(name).lower() in _BONA)
    log.info("model %s | id2label=%s | bona index=%d", cfg.hf_model, id2label, bona_idx)

    manifest = pd.read_parquet(cfg.scores_in)
    n = int(cfg.data.max_seconds * TARGET_SR)
    bs = cfg.data.batch_size
    scores = np.empty(len(manifest), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(manifest), bs):
            rows = manifest.iloc[start : start + bs]
            waves = [_fix(read_audio(p), n) for p in rows["path"]]
            inputs = fe(waves, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            logits = model(inputs.input_values.to(cfg.device)).logits
            scores[start : start + len(rows)] = logits[:, bona_idx].cpu().numpy()
            if start % (bs * 20) == 0:
                log.info("scored %d / %d", start + len(rows), len(manifest))

    out = manifest.copy()
    out["score"] = scores
    keep = [c for c in ["score", "label", "condition", "codec", "language", "split"] if c in out]
    Path(cfg.scores_out).parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_parquet(cfg.scores_out)
    log.info("wrote %d scores -> %s", len(out), cfg.scores_out)


if __name__ == "__main__":
    main()
