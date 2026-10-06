"""Staged, lean downloader. Pulls the datasets listed in configs/data/*.yaml into data_root.

    uv run python scripts/download_data.py

Verify every repo id / URL against its primary page before relying on it -- several sources in
this area are weeks-old preprints with shifting mirrors.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import hydra
from omegaconf import DictConfig

from caesd.utils.logging import get_logger

log = get_logger("caesd.download")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.paths.data_root)
    root.mkdir(parents=True, exist_ok=True)

    for repo in cfg.data.hf_repos:
        from huggingface_hub import snapshot_download

        log.info("HF snapshot: %s", repo.repo_id)
        snapshot_download(repo_id=repo.repo_id, repo_type=repo.get("repo_type", "dataset"),
                          local_dir=str(root / repo.repo_id.split("/")[-1]))

    for item in cfg.data.urls:
        dest = root / item.name
        if dest.exists():
            log.info("skip (exists): %s", dest)
            continue
        log.info("download: %s -> %s", item.url, dest)
        urllib.request.urlretrieve(item.url, dest)

    if not cfg.data.hf_repos and not cfg.data.urls:
        log.warning("no sources configured; fill configs/data/lean.yaml with verified ids/URLs")


if __name__ == "__main__":
    main()
