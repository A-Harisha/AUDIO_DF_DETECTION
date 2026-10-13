"""Combine the scores of several detector runs into one ensemble, and report it against its best
member on the evaluation set.

    uv run python scripts/ensemble.py +study=ens_bands "+runs=[band0_7,band7_13,band13_19,band19_25]"
    uv run python scripts/ensemble.py +study=ens_bands_plus "+runs=[...]" "+views=[clean,codec,encodec]"

Claim served: member diversity, not the fusion rule, is what decides whether an ensemble beats the
ASVspoof 5 ensemble SOTA (T45, 2.59% EER / 0.0750 minDCF, arXiv:2408.08739). Fusing the nine
near-duplicate runs of 2026-10-13 gained 0.15 EER points at an eval-selected optimum; this script
measures the correlation matrix that explains that, and the gain each combiner gets over the best
single member once the members are built to differ.

Discipline, the same as everywhere in this project:
  dev ``calib``   fits the combiner (member scaling, logistic weights)
  dev ``select``  chooses between combiners; it is unseen attacks A09-A16
  eval            scored once, never used for a decision

Combiners (member scores -> one score):
  best_single   the member with the lowest dev select EER; the number every ensemble must beat
  mean          unweighted mean of the members' z-scores (scaling fit on calib)
  logistic      logistic regression on the members' z-scores, fit on calib

Writes experiments/runs/<study>/: config.json, ensemble.json (correlations, per-combiner metrics
per condition and per attack, paired bootstrap CIs against best_single), scores_<set>.parquet.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from caesd.eval import member_scores
from caesd.metrics import compute_eer, min_dcf
from caesd.utils.logging import get_logger, log_hparams, write_json
from caesd.utils.seed import set_seed

log = get_logger("caesd.ensemble")
COMBINERS = ("best_single", "mean", "logistic")


def fit_combiners(z_calib: np.ndarray, label: np.ndarray, member_eer: np.ndarray) -> dict:
    """Fit every combiner on the dev ``calib`` rows. Returns one callable per combiner."""
    from sklearn.linear_model import LogisticRegression

    pick = int(np.argmin(member_eer))
    lr = LogisticRegression(class_weight="balanced", C=1e6).fit(z_calib, label)
    log.info("logistic weights %s bias %.3f", np.round(lr.coef_[0], 3).tolist(), float(lr.intercept_[0]))
    log.info("best_single is member %d (dev select EER %.4f)", pick, member_eer[pick])
    return {"best_single": lambda z: z[:, pick], "mean": lambda z: z.mean(axis=1),
            "logistic": lambda z: lr.decision_function(z)}


def paired_ci(metric, a: np.ndarray, b: np.ndarray, label: np.ndarray, n: int, seed: int) -> list[float]:
    """95% bootstrap CI of metric(a) - metric(b) on the same resampled trials. Negative = a wins."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, label.size, label.size)
        if label[idx].min() != label[idx].max():
            diffs.append(metric(a[idx], label[idx]) - metric(b[idx], label[idx]))
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def report(frame: pd.DataFrame, group: str, n_boot: int, seed: int) -> dict:
    """EER and minDCF of every combiner per level of ``group``, plus the paired difference against
    best_single with a 95% CI."""
    rows = {}
    for key, sub in frame.groupby(group):
        y = sub["label"].to_numpy()
        if np.unique(y).size < 2:
            continue
        row: dict = {"n": len(sub)}
        for c in COMBINERS:
            s = sub[c].to_numpy()
            row[f"eer/{c}"], row[f"min_dcf/{c}"] = compute_eer(s, y)[0], min_dcf(s, y)
        for name, metric in (("eer", lambda s, y: compute_eer(s, y)[0]), ("min_dcf", min_dcf)):
            for c in COMBINERS[1:]:
                row[f"{name}/{c}-best_single/ci95"] = paired_ci(
                    metric, sub[c].to_numpy(), sub["best_single"].to_numpy(), y, n_boot, seed)
        rows[str(key)] = row
        log.info("%-10s n=%7d | EER %s | minDCF %s | d_EER(mean) CI [%+.4f, %+.4f]", key, row["n"],
                 " ".join(f"{row[f'eer/{c}']:.4f}" for c in COMBINERS),
                 " ".join(f"{row[f'min_dcf/{c}']:.4f}" for c in COMBINERS),
                 *row["eer/mean-best_single/ci95"])
    return rows


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.paths.runs) / cfg.study
    out_dir.mkdir(parents=True)  # raises if the study exists: results are never overwritten
    resolved = OmegaConf.to_container(cfg, resolve=True)
    log_hparams(log, resolved)
    write_json(out_dir / "config.json", resolved)
    set_seed(cfg.seed)
    runs, views = list(cfg.runs), list(cfg.get("views", ["clean", "codec"]))
    n_boot, runs_dir = cfg.get("n_boot", 200), Path(cfg.paths.runs)

    # --- dev: the fit and selection data, one row block per view ---
    dev_meta, dev_raw = [], []
    for view in views:
        meta, s = member_scores(runs_dir, runs, f"dev_{view}", ["utt", "label", "split", "speaker", "generator"])
        dev_meta.append(meta.assign(view=view))
        dev_raw.append(s)
    dev_meta, dev_raw = pd.concat(dev_meta, ignore_index=True), np.concatenate(dev_raw)
    calib = (dev_meta["split"] == "calib").to_numpy()
    y_dev = dev_meta["label"].to_numpy()

    # Members differ in score scale, so every combiner reads z-scores standardised on calib only.
    mu, sd = dev_raw[calib].mean(0), dev_raw[calib].std(0).clip(1e-6)
    dev_z = (dev_raw - mu) / sd
    select = ~calib
    member_eer = np.array([compute_eer(dev_z[select, j], y_dev[select])[0] for j in range(len(runs))])
    log.info("dev: calib %d, select %d, views %s", calib.sum(), select.sum(), views)
    for run, eer in zip(runs, member_eer, strict=True):
        log.info("member %-16s dev select EER %.4f", run, eer)

    combiners = fit_combiners(dev_z[calib], y_dev[calib], member_eer)
    results: dict = {"runs": runs, "views": views, "scale": {"mean": mu.tolist(), "std": sd.tolist()},
                     "member_dev_select_eer": dict(zip(runs, member_eer.tolist(), strict=True))}

    # --- the diagnosis: how much do the members actually differ? ---
    for part, z in (("dev_select", dev_z[select]), ):
        corr = np.corrcoef(z.T)
        results[f"correlation_{part}"] = {"runs": runs, "matrix": np.round(corr, 4).tolist(),
                                          "mean_offdiag": float((corr.sum() - len(runs)) / (len(runs) ** 2 - len(runs)))}
        log.info("%s member correlation: mean off-diagonal %.3f\n%s", part,
                 results[f"correlation_{part}"]["mean_offdiag"],
                 pd.DataFrame(np.round(corr, 3), index=runs, columns=range(len(runs))).to_string())

    # --- dev select: the only place a combiner may be chosen ---
    dev_out = dev_meta.assign(**{c: f(dev_z) for c, f in combiners.items()})
    results["dev_select"] = report(dev_out[select].assign(all="all"), "all", n_boot, cfg.seed)["all"]
    results["dev_select_per_view"] = report(dev_out[select], "view", n_boot, cfg.seed)
    dev_out[select].to_parquet(out_dir / "scores_dev_select.parquet")

    # --- eval: scored once, reported pooled, per real codec and per attack ---
    ev_meta, ev_raw = member_scores(runs_dir, runs, "eval_clean", ["utt", "label", "codec", "generator"])
    ev_out = ev_meta.assign(**{c: f((ev_raw - mu) / sd) for c, f in combiners.items()})
    corr = np.corrcoef(((ev_raw - mu) / sd).T)
    results["correlation_eval"] = {"runs": runs, "matrix": np.round(corr, 4).tolist(),
                                   "mean_offdiag": float((corr.sum() - len(runs)) / (len(runs) ** 2 - len(runs)))}
    log.info("eval member correlation: mean off-diagonal %.3f", results["correlation_eval"]["mean_offdiag"])
    log.info("== eval, pooled (the benchmark numbers; SOTA to beat: 2.59%% EER / 0.0750 minDCF)")
    results["eval_pooled"] = report(ev_out.assign(all="all"), "all", n_boot, cfg.seed)["all"]
    log.info("== eval, per real codec")
    results["eval_per_codec"] = report(ev_out, "codec", n_boot, cfg.seed)
    log.info("== eval, per attack (A18, A20, A23, A27, A30, A31, A32 are the adversarial ones)")
    bona = ev_out["label"].to_numpy() == 1
    per_attack = {}
    for attack in sorted(ev_out.loc[~bona, "generator"].unique()):
        sub = ev_out[bona | (ev_out["generator"] == attack).to_numpy()]
        per_attack[str(attack)] = {"n": len(sub), **{f"eer/{c}": compute_eer(sub[c].to_numpy(), sub["label"].to_numpy())[0]
                                                     for c in COMBINERS}}
        log.info("%-5s n=%7d | EER %s", attack, len(sub),
                 " ".join(f"{per_attack[str(attack)][f'eer/{c}']:.4f}" for c in COMBINERS))
    results["eval_per_attack"] = per_attack
    ev_out.to_parquet(out_dir / "scores_eval_clean.parquet")

    write_json(out_dir / "ensemble.json", results)
    log.info("wrote %s", out_dir / "ensemble.json")


if __name__ == "__main__":
    main()
