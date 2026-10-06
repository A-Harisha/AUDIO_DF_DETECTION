"""Phase 0 go/no-go. Reads precomputed detector scores on the condition grid and reports
whether the calibration premise holds.

    uv run python scripts/phase0_falsify.py experiment=phase0
"""
from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from caesd.eval import run_falsification
from caesd.utils.logging import get_logger, write_json

log = get_logger("caesd.phase0")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    exp = cfg.experiment
    scores_path = Path(exp.scores_parquet)
    if not scores_path.exists():
        raise FileNotFoundError(
            f"{scores_path} not found. Score the condition grid first "
            "(AASIST over In-the-Wild + IndicSynth) and write score,label,condition,codec,language,split."
        )
    df = pd.read_parquet(scores_path)
    result = run_falsification(
        df,
        far_target=exp.far_target,
        holdout_codecs=list(exp.holdout_codecs),
        holdout_languages=list(exp.holdout_languages),
        gap_is_large=exp.gap_is_large,
        lookup_closes_gap=exp.lookup_closes_gap,
        min_per_cell=exp.lookup_min_per_cell,
        p_target=exp.get("p_target", 0.05),
    )
    log.info("DECISION: %s", result["decision"])
    log.info("REASON:   %s", result["reason"])

    out_dir = Path(cfg.paths.experiments) / "phase0"
    write_json(out_dir / "falsification.json", result)
    write_json(out_dir / "config.json", OmegaConf.to_container(cfg, resolve=True))
    log.info("wrote %s", out_dir / "falsification.json")


if __name__ == "__main__":
    main()
