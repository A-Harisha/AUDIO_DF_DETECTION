"""Assemble a lean ASVspoof 5 manifest (dev for Phase-0 eval, or train for the CM).

    uv run python scripts/prepare_asvspoof5.py +partition=train
    uv run python scripts/prepare_asvspoof5.py +partition=dev

Extracts a balanced bona/spoof subset from one tar. Our codec grid / RawBoost is applied later
to break the codec<->label confound. Phase 0 trains nothing, so only speaker-disjoint splits
are enforced; generator-disjointness is handled at training time.
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from caesd.data.manifests import assert_no_leakage, assign_split_by, build_manifest
from caesd.data.sources import parse_asvspoof5
from caesd.utils.logging import get_logger

log = get_logger("caesd.prepare.a5")

_PARTITION = {
    # partition: (protocol, flac tar, split fractions (train,dev,test), output name, default cap)
    "dev": ("ASVspoof5.dev.track_1.tsv", "flac_D_aa.tar", (0.0, 0.5, 0.5), "clean_manifest.parquet", 400),
    "train": ("ASVspoof5.train.tsv", "flac_T_aa.tar", (0.9, 0.1, 0.0), "train_manifest.parquet", 2500),
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    partition = cfg.get("partition", "dev")
    protocol, tar, fractions, out_name, default_cap = _PARTITION[partition]
    a5 = Path(cfg.paths.data_root) / "asvspoof5"

    records = parse_asvspoof5(
        protocols_tar=a5 / "ASVspoof5_protocols.tar",
        flac_tar=a5 / tar,
        out_dir=a5 / f"{partition}_subset",
        protocol=protocol,
        max_per_class=cfg.get("max_per_class", default_cap),
    )
    log.info("%s: extracted %d clips", partition, len(records))

    df = build_manifest(records)
    df = assign_split_by(df, key="speaker", fractions=fractions, seed=cfg.seed)
    assert_no_leakage(df, keys=("speaker",))

    out = Path(cfg.paths.experiments) / "phase0" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    log.info("%s manifest %d rows -> %s | splits %s | classes %s", partition, len(df), out,
             df["split"].value_counts().to_dict(), df["class_name"].value_counts().to_dict())


if __name__ == "__main__":
    main()
