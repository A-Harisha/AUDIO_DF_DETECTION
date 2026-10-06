"""Run the metric suite per group on a scored manifest.

    uv run python scripts/evaluate.py scores_parquet=... group_by=condition

Expects columns score, llr, label and the grouping column. Reports per group, never pooled.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from caesd.eval import evaluate_per_group, evaluate_scores
from caesd.utils.logging import get_logger, write_json

log = get_logger("caesd.evaluate")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    df = pd.read_parquet(cfg.scores_parquet)
    group_by = cfg.get("group_by", "condition")
    results = {"pooled": evaluate_scores(df), "per_group": evaluate_per_group(df, group_by)}
    out = Path(cfg.paths.experiments) / "eval" / f"{group_by}.json"
    write_json(out, results)
    log.info("pooled EER=%.4f actDCF=%.4f Cllr=%.4f -> %s",
             results["pooled"]["eer"], results["pooled"]["act_dcf"], results["pooled"]["cllr"], out)


if __name__ == "__main__":
    main()
