"""Score a condition grid with a trained CAESD detector (raw detection score, higher => bona fide).

    uv run python scripts/score_grid_caesd.py +checkpoint=... +scores_in=... +scores_out=...

Produces score,label,condition,codec,language,split for the Phase-0 falsification, this time
from a model that actually discriminates on the base corpus.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from caesd.models import Backbone, CaesdDetector
from caesd.utils.io import TARGET_SR, read_audio
from caesd.utils.logging import get_logger

log = get_logger("caesd.score.caesd")


def _fix(wave: np.ndarray, n: int) -> np.ndarray:
    if wave.shape[0] >= n:
        return wave[:n]
    return np.tile(wave, n // wave.shape[0] + 1)[:n]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    backbone = Backbone.from_pretrained(ckpt["backbone"], lora_rank=ckpt["lora_rank"])
    model = CaesdDetector(backbone, n_channels=ckpt.get("n_channels", 8))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(cfg.device)

    manifest = pd.read_parquet(cfg.scores_in)
    n = int(cfg.data.max_seconds * TARGET_SR)
    bs = cfg.data.batch_size
    scores = np.empty(len(manifest), dtype=np.float64)     # raw detection score
    llr_cond = np.empty(len(manifest), dtype=np.float64)   # block C condition-dependent LLR
    with torch.no_grad():
        for start in range(0, len(manifest), bs):
            rows = manifest.iloc[start : start + bs]
            batch = torch.stack([torch.from_numpy(_fix(read_audio(p), n)).float() for p in rows["path"]])
            out_b = model(batch.to(cfg.device))
            scores[start : start + len(rows)] = out_b.raw_score.cpu().numpy()
            llr_cond[start : start + len(rows)] = model.calibrated_llr(out_b).cpu().numpy()
            if start % (bs * 20) == 0:
                log.info("scored %d / %d", start + len(rows), len(manifest))

    out = manifest.copy()
    out["score"] = scores
    out["llr_cond"] = llr_cond
    keep = [c for c in ["score", "llr_cond", "label", "condition", "codec", "language", "split"] if c in out]
    Path(cfg.scores_out).parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_parquet(cfg.scores_out)
    log.info("wrote %d scores -> %s", len(out), cfg.scores_out)


if __name__ == "__main__":
    main()
