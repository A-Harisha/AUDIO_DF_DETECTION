"""Assemble a lean clean manifest from ASVspoof 5 dev for the redesigned Phase 0.

    uv run python scripts/prepare_asvspoof5.py

Extracts a balanced bona/spoof subset from one dev tar. Our frozen grid is applied afterwards
(build_condition_grid) as the controlled codec axis. Phase 0 trains nothing, so only a
speaker-disjoint dev/test split is enforced; generator-disjointness is a Phase-2 concern.
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from caesd.data.manifests import assert_no_leakage, assign_split_by, build_manifest
from caesd.data.sources import parse_asvspoof5_dev
from caesd.utils.logging import get_logger

log = get_logger("caesd.prepare.a5")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    a5 = Path(cfg.paths.data_root) / "asvspoof5"
    records = parse_asvspoof5_dev(
        protocols_tar=a5 / "ASVspoof5_protocols.tar",
        flac_tar=a5 / "flac_D_aa.tar",
        out_dir=a5 / "dev_subset",
        max_per_class=cfg.get("max_per_class", 400),
    )
    log.info("extracted %d clips", len(records))

    df = build_manifest(records)
    df = assign_split_by(df, key="speaker", fractions=(0.0, 0.5, 0.5), seed=cfg.seed)
    assert_no_leakage(df, keys=("speaker",))  # Phase 0: speaker-disjoint is what matters

    out = Path(cfg.paths.experiments) / "phase0" / "clean_manifest.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    log.info("clean manifest %d rows -> %s | splits %s | classes %s",
             len(df), out, df["split"].value_counts().to_dict(), df["class_name"].value_counts().to_dict())


if __name__ == "__main__":
    main()
