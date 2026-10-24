"""Choose the member set and report its eval score, with the choice made ONLY on dev.

    uv run python scripts/select_members.py paths.runs=experiments/runs_partial \
        "+members=[lora_enc,lora_mhfa,lora_lfcc]" "+views=[clean,codec,encodec]"

Claim served: the headline system. It exists because on 2026-10-24 the member set was chosen by
reading an eval table -- `lora_enc` + `lora_lfcc` at 2.93% EER -- while dev select ranked that set
FOURTH and ranked the best single eval member LAST. A number picked off an eval table is not a
result, and the retired 3.80% headline had the same fault one level down, in the combiner. This
script makes the choice mechanical: dev select ranks the subsets, the top-ranked one is the answer,
and every other row is printed only so the size of the mistake is visible.

Members are standardised with statistics from the dev ``calib`` split, never from eval, and combined
by the unweighted mean, which has no parameters to fit. ``+views`` is the selection set: adding
``encodec`` puts a NEURAL codec in it, which is the condition eval punishes hardest and dev
otherwise lacks entirely.

Writes <runs>/<study>/selection.json. ``+study`` defaults to ``selection``.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from caesd.metrics import compute_eer, min_cllr, min_dcf, paired_ci
from caesd.utils.logging import get_logger, write_json

log = get_logger("caesd.select")
EER = lambda s, y: compute_eer(s, y)[0]  # noqa: E731


def calib_stats(runs: Path, member: str, views: list[str]) -> tuple[float, float]:
    """Mean and standard deviation on the dev ``calib`` split, pooled over the views."""
    frames = [pd.read_parquet(runs / member / f"scores_dev_{v}.parquet") for v in views]
    scores = pd.concat(frames)
    scores = scores[scores["split"] == "calib"]["score"]
    if len(scores) == 0:
        raise ValueError(f"{member}: no calib clips in {views}; the scale cannot be fit")
    return float(scores.mean()), float(scores.std())


def dev_select(runs: Path, members: list[str], views: list[str],
               stats: dict[str, tuple[float, float]]) -> dict[str, pd.DataFrame]:
    """Standardised member scores on the dev ``select`` split, one frame per view."""
    out = {}
    for view in views:
        frame = None
        for member in members:
            part = pd.read_parquet(runs / member / f"scores_dev_{view}.parquet")
            part = part[part["split"] == "select"][["utt", "label", "score"]]
            mean, std = stats[member]
            part[member] = (part["score"] - mean) / std
            part = part.drop(columns="score")
            frame = part if frame is None else frame.merge(part.drop(columns="label"), on="utt")
        out[view] = frame
    return out


def eval_scores(runs: Path, members: list[str], stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    frame = None
    for member in members:
        part = pd.read_parquet(runs / member / "scores_eval_clean.parquet")[["utt", "label", "score"]]
        mean, std = stats[member]
        part[member] = (part["score"] - mean) / std
        part = part.drop(columns="score")
        frame = part if frame is None else frame.merge(part.drop(columns="label"), on="utt")
    return frame


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    runs = Path(cfg.paths.runs)
    members = list(cfg.members)
    views = list(cfg.get("views", ["clean", "codec"]))
    max_size = int(cfg.get("max_size", len(members)))
    n_boot = int(cfg.get("n_boot", 300))

    stats = {m: calib_stats(runs, m, views) for m in members}
    dev = dev_select(runs, members, views, stats)
    ev = eval_scores(runs, members, stats)
    y_eval = ev["label"].to_numpy()
    log.info("members %s | selection views %s | dev select %s clips/view | eval %d clips",
             members, views, {v: len(f) for v, f in dev.items()}, len(ev))

    rows = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(members, size):
            per_view = [EER(dev[v][list(combo)].mean(axis=1).to_numpy(),
                            dev[v]["label"].to_numpy()) for v in views]
            score = ev[list(combo)].mean(axis=1).to_numpy()
            rows.append({"members": list(combo), "dev_select": float(np.mean(per_view)),
                         "dev_per_view": dict(zip(views, per_view, strict=True)),
                         "eval_eer": EER(score, y_eval), "eval_min_dcf": min_dcf(score, y_eval),
                         "eval_min_cllr": min_cllr(score, y_eval)})
    rows.sort(key=lambda r: r["dev_select"])

    best_single = min((r for r in rows if len(r["members"]) == 1), key=lambda r: r["dev_select"])
    chosen = rows[0]
    log.info("%-4s %-10s %-42s %-9s %s", "rank", "dev select", "members", "eval EER", "eval minDCF")
    for rank, row in enumerate(rows, 1):
        mark = "<== SELECTED" if row is chosen else ""
        log.info("%-4d %-10.4f %-42s %-9.4f %.4f %s", rank, row["dev_select"],
                 "+".join(m.replace("lora_", "") for m in row["members"]),
                 row["eval_eer"], row["eval_min_dcf"], mark)

    # The claim: the dev-chosen set against the dev-chosen single member, paired on eval.
    a = ev[chosen["members"]].mean(axis=1).to_numpy()
    b = ev[best_single["members"]].mean(axis=1).to_numpy()
    verdict: dict = {"members": chosen["members"], "vs_best_single": best_single["members"]}
    for key, metric in (("eer", EER), ("min_dcf", min_dcf)):
        lo, hi = paired_ci(metric, a, b, y_eval, n_boot, int(cfg.seed))
        verdict[f"d_{key}/ci95"] = [lo, hi]
        verdict[f"d_{key}/verdict"] = ("fusion better" if hi < 0 else
                                       "single better" if lo > 0 else "not established")
    log.info("SELECTED %s: eval EER %.4f / minDCF %.4f / minCllr %.4f",
             "+".join(chosen["members"]), chosen["eval_eer"], chosen["eval_min_dcf"],
             chosen["eval_min_cllr"])
    log.info("   against the dev-chosen single member %s: d_EER CI [%+.4f, %+.4f] -> %s",
             best_single["members"][0], *verdict["d_eer/ci95"], verdict["d_eer/verdict"])
    # How much the dev ranking costs, reported but NOT claimable: the best eval row may differ.
    best_eval = min(rows, key=lambda r: r["eval_eer"])
    if best_eval is not chosen:
        log.info("   dev did NOT pick the best eval row (%s at %.4f): the selection set costs "
                 "%.4f EER points here", "+".join(best_eval["members"]), best_eval["eval_eer"],
                 chosen["eval_eer"] - best_eval["eval_eer"])

    out_dir = runs / str(cfg.get("study", "selection"))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "selection.json",
               {"views": views, "members": members, "selected": chosen, "verdict": verdict,
                "best_eval_row": best_eval, "ranking": rows})
    log.info("wrote %s", out_dir / "selection.json")


if __name__ == "__main__":
    main()
