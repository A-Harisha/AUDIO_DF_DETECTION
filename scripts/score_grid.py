"""Score a manifest with a pretrained detector, producing the parquet Phase 0 reads.

    uv run python scripts/score_grid.py scores_in=<manifest.parquet> scores_out=<scores.parquet>

Output columns: score, label, condition, codec, language, split. Score convention: higher =>
bona fide. AASIST is the default detector; it length-fixes each clip to 64600 samples itself.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from caesd.baselines import AasistScorer
from caesd.utils.io import read_audio
from caesd.utils.logging import get_logger

log = get_logger("caesd.score")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    manifest = pd.read_parquet(cfg.scores_in)
    scorer = AasistScorer(cfg.data.aasist.repo_dir, cfg.data.aasist.checkpoint, device=cfg.device)

    batch_size = cfg.data.batch_size
    scores = np.empty(len(manifest), dtype=np.float64)
    for start in range(0, len(manifest), batch_size):
        rows = manifest.iloc[start : start + batch_size]
        waves = [read_audio(p) for p in rows["path"]]
        batch = scorer.batch_from_waves(waves)
        scores[start : start + len(rows)] = scorer.score(batch)
        if start % (batch_size * 20) == 0:
            log.info("scored %d / %d", start + len(rows), len(manifest))

    out = manifest.copy()
    out["score"] = scores
    keep = [c for c in ["score", "label", "condition", "codec", "language", "split"] if c in out]
    Path(cfg.scores_out).parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_parquet(cfg.scores_out)
    log.info("wrote %d scores -> %s", len(out), cfg.scores_out)


if __name__ == "__main__":
    main()
