"""Phase 2 headline: does condition-dependent calibration (block C) close the held-out-codec
calibration gap that a single global calibration cannot?

    uv run python scripts/phase2_eval.py +scores_parquet=... +holdout_codec=speex8k

Reads a dev grid scored by the trained model (columns score, llr_cond, label, condition, codec,
split). Compares, per condition at a usable prior: minDCF(raw), actDCF(global cal), actDCF(block C).
"""
from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from caesd.calibration import LogisticCalibrator
from caesd.metrics import act_dcf, min_dcf
from caesd.utils.logging import get_logger, write_json

log = get_logger("caesd.phase2")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    p = cfg.get("p_target", 0.5)
    holdout = cfg.get("holdout_codec", "speex8k")
    df = pd.read_parquet(cfg.scores_parquet)

    clean_dev = df[(df["split"] == "dev") & (df["codec"] == "clean")]
    global_cal = LogisticCalibrator().fit(clean_dev["score"].to_numpy(), clean_dev["label"].to_numpy())
    test = df[df["split"] == "test"].copy()
    test["llr_global"] = global_cal.transform(test["score"].to_numpy())

    rows = {}
    log.info("%-12s %8s %8s %8s | %9s %9s", "condition", "minDCF", "aDCF_glob", "aDCF_condC",
             "gap_glob", "gap_condC")
    for cond, sub in sorted(test.groupby("condition")):
        raw, y = sub["score"].to_numpy(), sub["label"].to_numpy()
        mn = min_dcf(raw, y, p_target=p)
        a_glob = act_dcf(sub["llr_global"].to_numpy(), y, p_target=p)
        a_cond = act_dcf(sub["llr_cond"].to_numpy(), y, p_target=p)
        rows[cond] = {"min_dcf": mn, "act_dcf_global": a_glob, "act_dcf_condC": a_cond,
                      "gap_global": a_glob - mn, "gap_condC": a_cond - mn}
        log.info("%-12s %8.3f %8.3f %8.3f | %9.3f %9.3f", cond, mn, a_glob, a_cond,
                 a_glob - mn, a_cond - mn)

    held = next((c for c in rows if c.startswith(holdout)), None)
    summary = {"p_target": p, "holdout_codec": holdout, "per_condition": rows}
    if held:
        g, c = rows[held]["gap_global"], rows[held]["gap_condC"]
        summary["headline"] = {"condition": held, "gap_global": g, "gap_condC": c,
                               "gap_reduction": g - c,
                               "relative_reduction": (g - c) / g if g else 0.0}
        log.info("HELD-OUT %s: global gap %.3f -> block C gap %.3f  (%.0f%% reduction)",
                 held, g, c, 100 * (g - c) / g if g else 0.0)

    out = Path(cfg.paths.experiments) / "phase2" / "calibration_comparison.json"
    write_json(out, summary)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
