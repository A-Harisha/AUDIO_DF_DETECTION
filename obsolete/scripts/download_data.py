"""Staged, lean downloader for Phase 0 / 1. Sources are verified in configs/data/lean.yaml.

    uv run python scripts/download_data.py

Pulls In-the-Wild (full, 8 GB) and a lean IndicSynth subset (a few language shards, not the
full 845 GB). AASIST is cloned separately. Verify every source against its primary page first.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import hydra
from omegaconf import DictConfig

from caesd.utils.logging import get_logger

log = get_logger("caesd.download")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from huggingface_hub import snapshot_download

    root = Path(cfg.paths.data_root)
    root.mkdir(parents=True, exist_ok=True)

    itw = cfg.data.in_the_wild
    itw_dir = root / "in_the_wild"
    log.info("In-the-Wild -> %s", itw_dir)
    snapshot_download(repo_id=itw.repo_id, repo_type=itw.repo_type, local_dir=str(itw_dir))
    zip_path = itw_dir / "release_in_the_wild.zip"
    if zip_path.exists() and not (itw_dir / "release_in_the_wild").exists():
        log.info("unzipping %s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(itw_dir)

    ind = cfg.data.indicsynth
    ind_dir = root / "indicsynth"
    log.info("IndicSynth (lean subset) -> %s", ind_dir)
    snapshot_download(repo_id=ind.repo_id, repo_type=ind.repo_type,
                      allow_patterns=list(ind.allow_patterns), local_dir=str(ind_dir))

    aasist = Path(cfg.data.aasist.repo_dir)
    log.info("AASIST present: %s", aasist.exists())
    log.info("done")


if __name__ == "__main__":
    main()
