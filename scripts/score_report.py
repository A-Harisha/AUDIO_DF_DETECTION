"""Turn a run's score parquets into the verdict, with error bars, against named control runs.

    uv run python scripts/score_report.py paths.runs=experiments/runs_partial +run=lora_probe \
        "+sets=[dev_clean,dev_codec]" "+controls=[full_seed1337,full_seed2]" +split=select
    uv run python scripts/score_report.py paths.runs=experiments/runs_partial +run=lora_probe \
        "+sets=[eval_clean]" +group=codec

Claim served: fine-tuning the backbone (LoRA) closes the remaining gap to the published SOTA. It
exists because the first LoRA verdict was wrong twice -- the control was a run with a third
training view, and a 0.0038 margin was quoted with no error bar when the standard error was 0.0035.
Both faults are mechanical, so the comparison is mechanised here rather than done by hand again.

Against each control the difference is a PAIRED bootstrap over the clips BOTH runs scored: the two
systems see the same clips, so their errors are correlated and unpaired intervals would be far too
wide. A CI that straddles zero means the difference is not established, and the script says so in
the ``verdict`` field rather than leaving it to be read off a table.

Reference points printed for context, from docs/experiment_log.md: our ensemble headline is
3.80% EER / 0.1046 minDCF (three-seed mean) and T45, the published system we are chasing, is
2.59% / 0.0750. They are reported, never used for a decision.

Writes <run_dir>/report_<set>.json and logs one line per row.
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from caesd.metrics import compute_eer, min_cllr, min_dcf, paired_ci
from caesd.utils.logging import get_logger, write_json

log = get_logger("caesd.score_report")
REFERENCE = {"ours_ensemble_3seed": (0.0380, 0.1046), "T45_published": (0.0259, 0.0750)}
EER = lambda s, y: compute_eer(s, y)[0]  # noqa: E731


def load(run_dir: Path, name: str, split: str | None) -> pd.DataFrame:
    frame = pd.read_parquet(run_dir / f"scores_{name}.parquet")
    if split is not None:
        frame = frame[frame["split"] == split]
    if frame.empty:
        raise ValueError(f"{run_dir.name}/{name}: no clips left after split={split}")
    return frame.reset_index(drop=True)


def metrics(frame: pd.DataFrame) -> dict:
    s, y = frame["score"].to_numpy(), frame["label"].to_numpy()
    return {"n": len(frame), "n_bona": int((y == 1).sum()),
            "eer": EER(s, y), "min_dcf": min_dcf(s, y), "min_cllr": min_cllr(s, y)}


def against(frame: pd.DataFrame, control: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Paired comparison on the clips both runs scored. Negative difference = the run wins."""
    both = frame.merge(control[["utt", "score"]], on="utt", how="inner", suffixes=("", "_ctl"))
    if both.empty:
        raise ValueError("the run and the control share no scored clip")
    y = both["label"].to_numpy()
    a, b = both["score"].to_numpy(), both["score_ctl"].to_numpy()
    out = {"n_paired": len(both), "eer": EER(a, y), "eer_control": EER(b, y),
           "min_dcf": min_dcf(a, y), "min_dcf_control": min_dcf(b, y)}
    for key, metric in (("eer", EER), ("min_dcf", min_dcf)):
        lo, hi = paired_ci(metric, a, b, y, n_boot, seed)
        out[f"d_{key}"] = out[key] - out[f"{key}_control"]
        out[f"d_{key}/ci95"] = [lo, hi]
        out[f"d_{key}/verdict"] = ("run better" if hi < 0 else "control better" if lo > 0
                                   else "not established")
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    runs = Path(cfg.paths.runs)
    run_dir = runs / cfg.run
    split = cfg.get("split")
    group = cfg.get("group")
    n_boot, seed = int(cfg.get("n_boot", 1000)), int(cfg.seed)
    for name in cfg.sets:
        frame = load(run_dir, name, split)
        out = {"set": name, "split": split, "overall": metrics(frame), "controls": {}, "groups": {}}
        log.info("%s | %s | n=%d bona=%d | EER %.4f | minDCF %.4f | minCllr %.4f", cfg.run, name,
                 out["overall"]["n"], out["overall"]["n_bona"], out["overall"]["eer"],
                 out["overall"]["min_dcf"], out["overall"]["min_cllr"])
        # eval sets ONLY: both reference points are eval numbers, and dev is far easier here
        # (a control scores 0.0142 on dev clean against 3.89% on eval). Printing them beside a dev
        # number invites exactly the comparison that made the first LoRA verdict wrong.
        for ref, (eer, dcf) in (REFERENCE.items() if name.startswith("eval") else ()):
            log.info("   vs %-20s EER %+.4f | minDCF %+.4f", ref,
                     out["overall"]["eer"] - eer, out["overall"]["min_dcf"] - dcf)
        for control in cfg.get("controls", []):
            out["controls"][control] = against(frame, load(runs / control, name, split), n_boot, seed)
            c = out["controls"][control]
            log.info("   vs %-20s n=%6d | EER %.4f vs %.4f | d %+.4f CI [%+.4f, %+.4f] -> %s",
                     control, c["n_paired"], c["eer"], c["eer_control"], c["d_eer"],
                     *c["d_eer/ci95"], c["d_eer/verdict"])
        if group is not None:
            # A per-ATTACK cell holds spoof clips only, because the bona fide ones carry no
            # generator. The ASVspoof protocol scores each attack against ALL bona fide clips, so a
            # single-class cell is unioned with them rather than dropped -- dropping it returned an
            # empty breakdown and said nothing about it.
            bona = frame["label"].to_numpy() == 1
            for key, sub in frame.groupby(group):
                if np.unique(sub["label"].to_numpy()).size < 2:
                    sub = pd.concat([frame[bona], sub[sub["label"] == 0]], ignore_index=True)
                if np.unique(sub["label"].to_numpy()).size < 2:
                    continue  # a bona-fide-only cell: nothing to discriminate against
                out["groups"][str(key)] = metrics(sub)
            for key, row in sorted(out["groups"].items(), key=lambda kv: -kv[1]["eer"])[:12]:
                log.info("   %-12s n=%6d | EER %.4f | minDCF %.4f", key, row["n"], row["eer"], row["min_dcf"])
        write_json(run_dir / f"report_{name}.json", out)
        log.info("wrote %s", run_dir / f"report_{name}.json")


if __name__ == "__main__":
    main()
