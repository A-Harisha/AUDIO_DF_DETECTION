"""Assemble the clean Phase-0 manifest from In-the-Wild + the IndicSynth subset.

    uv run python scripts/prepare_phase0.py

In-the-Wild gives English bona-fide + spoof (both classes, so it carries the codec-shift
analysis). IndicSynth adds synthetic Indian-language clips. Splits are disjoint by speaker;
Phase 0 trains nothing, so only dev (fit calibration) and test (evaluate) are produced.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from caesd.data.manifests import assert_no_leakage, assign_split_by, build_manifest
from caesd.data.sources import parse_in_the_wild, parse_indicsynth
from caesd.utils.logging import get_logger

log = get_logger("caesd.prepare")


def _balance(df: pd.DataFrame, max_per_class: int | None, seed: int) -> pd.DataFrame:
    if max_per_class is None:
        return df
    return (df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed)))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.paths.data_root)
    max_per_class = cfg.get("max_per_class", 1500)   # keep the first run quick
    max_synth = cfg.get("max_synth_per_lang", 500)

    records = parse_in_the_wild(root / "in_the_wild")
    log.info("In-the-Wild: %d clips", len(records))

    ind_root = root / "indicsynth"
    synth_out = root / "indicsynth_wav"
    for lang_dir in sorted(p for p in ind_root.glob("*") if p.is_dir()):
        shards = sorted(str(p) for p in lang_dir.glob("*.parquet"))
        if not shards:
            continue
        recs = parse_indicsynth(shards, lang_dir.name, synth_out, max_clips=max_synth)
        log.info("IndicSynth %s: %d clips from %d shard(s)", lang_dir.name, len(recs), len(shards))
        records.extend(recs)

    df = build_manifest(records)
    df = _balance(df, max_per_class, cfg.seed)
    df = assign_split_by(df, key="speaker", fractions=(0.0, 0.5, 0.5), seed=cfg.seed)
    assert_no_leakage(df)

    out = Path(cfg.paths.experiments) / "phase0" / "clean_manifest.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    log.info("clean manifest %d rows -> %s | splits %s | classes %s",
             len(df), out, df["split"].value_counts().to_dict(), df["class_name"].value_counts().to_dict())


if __name__ == "__main__":
    main()
