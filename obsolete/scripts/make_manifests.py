"""Build a manifest from raw per-clip records and enforce disjoint splits.

    uv run python scripts/make_manifests.py records_csv=... out=... split_key=speaker

The dataset-specific scan that produces ``records_csv`` (path,label,class_name,speaker,...)
is added per corpus in M2; this step applies the split and the leakage check once, correctly.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from caesd.data.manifests import assert_no_leakage, assign_split_by, build_manifest
from caesd.utils.logging import get_logger

log = get_logger("caesd.manifest")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    records = pd.read_csv(cfg.records_csv).to_dict("records")
    df = build_manifest(records)
    df = assign_split_by(df, key=cfg.get("split_key", "speaker"), seed=cfg.seed)
    assert_no_leakage(df)  # raises on any speaker/sentence/generator overlap
    out = Path(cfg.out)
    df.to_parquet(out)
    log.info("manifest %d rows -> %s | splits: %s", len(df), out, df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
