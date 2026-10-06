"""Phase 1 measurement paper tables.

    uv run python scripts/phase1_measure.py experiment=phase1 scores_parquet=<scores.parquet>

Reports the metric suite per condition and per language for a globally-calibrated detector.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from caesd.eval import run_measurement
from caesd.utils.logging import get_logger, write_json

log = get_logger("caesd.phase1")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    df = pd.read_parquet(cfg.scores_parquet)
    result = run_measurement(df, far_target=cfg.experiment.far_target)
    out = Path(cfg.paths.experiments) / "phase1" / "measurement.json"
    write_json(out, result)
    p = result["pooled"]
    log.info("pooled EER=%.4f actDCF=%.4f Cllr=%.4f calib_loss=%.4f",
             p["eer"], p["act_dcf"], p["cllr"], p["calibration_loss"])
    log.info("conditions measured: %d | languages: %d -> %s",
             len(result["per_condition"]), len(result["per_language"]), out)


if __name__ == "__main__":
    main()
