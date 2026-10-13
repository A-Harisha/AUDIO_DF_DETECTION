"""Combine the scores of several detector runs into one ensemble, and report it against its best
member on the evaluation set.

    uv run python scripts/ensemble.py +study=ens_bands "+runs=[band0_7,band7_13,band13_19,band19_25]" \
        "+views=[clean,codec,encodec]"

Claim served: member diversity, not the fusion rule, decides whether an ensemble beats the
ASVspoof 5 ensemble SOTA (T45, 2.59% EER / 0.0750 minDCF, arXiv:2408.08739). Fusing the nine
near-duplicate runs of 2026-10-13 gained 0.15 EER points at an eval-selected optimum; this script
reports the correlation matrix that explains that, and what each combiner then wins once the
members are built to differ.

Discipline, as everywhere in this project:
  dev ``calib``   fits the combiners; a speaker-disjoint quarter of it selects the gate's step
  dev ``select``  chooses between combiners; it is the unseen attacks A09-A16
  eval            scored once, never used for a decision

Combiners (member scores -> one score):
  best_single   the member with the lowest dev select EER; the number every ensemble must beat
  mean          unweighted mean of the members' z-scores
  logistic      logistic regression on the members' z-scores
  global_mix    one condition-independent weighted mixture, fit on condition-balanced Cllr
  gated         global_mix plus a condition-dependent shift read from the channel descriptor only

``gated`` is reportable only where the descriptor is label-free, and that is NOT everywhere. The
long-term average spectrum carries an ATTACK-SPECIFIC signature: a probe on the descriptor alone
recovers the class when it is fit and tested on the same attacks (dev calib -> dev select, EER
0.26-0.33) and gets nothing across attacks (train A01-A08 -> dev A09-A16, EER 0.55)
(docs/experiment_log.md, 2026-10-08). Both probes are therefore reported:
  within_attack  fit dev calib, test dev select. EXPECTED TO FAIL. It is why the dev ``gated``
                 numbers below are structurally unusable and must not be quoted as support.
  cross_attack   fit train A01-A08, test dev select A09-A16. This is the protocol the gate actually
                 runs under on eval -- fit on dev A09-A16, applied to eval A17-A32 -- so it is the
                 precondition for every gated EVAL number, and it belongs in any table quoting one.

Writes experiments/runs/<study>/: config.json, ensemble.json, scores_dev_select.parquet,
scores_eval_clean.parquet.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from caesd.data.manifests import assign_split_by
from caesd.eval import descriptor, member_scores
from caesd.metrics import compute_eer, min_dcf
from caesd.models.gate import ConditionGate, label_probe_eer
from caesd.utils.logging import get_logger, log_hparams, write_json
from caesd.utils.seed import set_seed

log = get_logger("caesd.ensemble")
COMBINERS = ("best_single", "mean", "logistic", "global_mix", "gated")


def read_set(cfg: DictConfig, runs: list[str], name: str, meta_columns: list[str]
             ) -> tuple[pd.DataFrame, np.ndarray]:
    """Aligned member scores and metadata of one score set, with the channel descriptor joined on
    ``utt``. The join is validated one-to-one: a partial descriptor cache would otherwise silently
    drop clips from the ensemble but not from the baseline."""
    meta, raw = member_scores(Path(cfg.paths.runs), runs, name, meta_columns)
    desc = descriptor(Path(cfg.paths.features), name)
    merged = meta.assign(_row=np.arange(len(meta))).merge(desc, on="utt", how="inner", validate="one_to_one")
    if len(merged) != len(meta):
        raise RuntimeError(f"{name}: {len(meta)} scored clips, {len(merged)} with a descriptor")
    return merged.drop(columns="_row"), raw[merged["_row"].to_numpy()]


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
    """EER and minDCF of every combiner per level of ``group``, plus each one's paired difference
    against best_single with a 95% CI."""
    rows = {}
    for key, sub in frame.groupby(group):
        y = sub["label"].to_numpy()
        if np.unique(y).size < 2:
            continue
        row: dict = {"n": len(sub), "null_weight": float(sub["null_weight"].mean())}
        for c in COMBINERS:
            s = sub[c].to_numpy()
            row[f"eer/{c}"], row[f"min_dcf/{c}"] = compute_eer(s, y)[0], min_dcf(s, y)
        for name, metric in (("eer", lambda s, y: compute_eer(s, y)[0]), ("min_dcf", min_dcf)):
            for c in COMBINERS[1:]:
                row[f"{name}/{c}-best_single/ci95"] = paired_ci(
                    metric, sub[c].to_numpy(), sub["best_single"].to_numpy(), y, n_boot, seed)
        rows[str(key)] = row
        log.info("%-10s n=%7d | EER %s | minDCF %s | d_EER(gated) CI [%+.4f, %+.4f] | null_w %.2f",
                 key, row["n"], " ".join(f"{row[f'eer/{c}']:.4f}" for c in COMBINERS),
                 " ".join(f"{row[f'min_dcf/{c}']:.4f}" for c in COMBINERS),
                 *row["eer/gated-best_single/ci95"], row["null_weight"])
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
    n_boot, n_anchors, steps = cfg.get("n_boot", 200), cfg.get("n_anchors", 8), cfg.get("gate_steps", 2000)

    # --- dev: one row block per view, the fit and selection data ---
    blocks = [read_set(cfg, runs, f"dev_{v}", ["utt", "label", "split", "speaker", "generator"]) for v in views]
    dev = pd.concat([b[0].assign(view=v) for b, v in zip(blocks, views, strict=True)], ignore_index=True)
    dev_raw = np.concatenate([b[1] for b in blocks])
    dev["group"] = dev["sim_codec"]  # the simulated condition each clip was scored in
    calib = (dev["split"] == "calib").to_numpy()
    y_dev = dev["label"].to_numpy()

    # Members differ in score scale, so every combiner reads z-scores standardised on calib only.
    mu, sd = dev_raw[calib].mean(0), dev_raw[calib].std(0).clip(1e-6)
    dev_z = (dev_raw - mu) / sd
    select = ~calib
    member_eer = np.array([compute_eer(dev_z[select, j], y_dev[select])[0] for j in range(len(runs))])
    for run, eer in zip(runs, member_eer, strict=True):
        log.info("member %-16s dev select EER %.4f", run, eer)

    # The gate selects its step on speakers it did not fit on: block C overfit a fixed step count
    # on all of the calibration data (docs/experiment_log.md, 2026-10-07).
    stop = np.zeros(len(dev), dtype=bool)
    buckets = assign_split_by(dev.loc[calib, ["speaker"]], "speaker", (0.75, 0.25, 0.0), cfg.seed + 1)
    stop[calib] = (buckets["split"] == "dev").to_numpy()
    log.info("dev: calib %d (fit %d / stop %d), select %d | views %s",
             calib.sum(), (calib & ~stop).sum(), stop.sum(), select.sum(), views)

    # --- the PRECONDITION for the gate, under both protocols (see the module docstring) ---
    z_dev = np.stack(dev["z"])
    probe_within = label_probe_eer(z_dev[calib], y_dev[calib], z_dev[select], y_dev[select])
    train_desc = pd.concat([descriptor(Path(cfg.paths.features), f"train_{v}") for v in views],
                           ignore_index=True)
    train_label = pd.read_parquet(Path(cfg.paths.manifests) / "train.parquet", columns=["utt", "label"])
    train_desc = train_desc.merge(train_label, on="utt", how="inner", validate="many_to_one")
    probe_cross = label_probe_eer(np.stack(train_desc["z"]), train_desc["label"].to_numpy(),
                                  z_dev[select], y_dev[select])
    del train_desc
    log.info("PRECONDITION | descriptor label probe | within-attack (calib -> select) EER %.4f "
             "(expected to FAIL: the dev gated numbers are unusable) | cross-attack (train -> select) "
             "EER %.4f (must be ~0.5; it is the precondition for the gated EVAL numbers)",
             probe_within, probe_cross)

    # --- fit every combiner on calib ---
    from sklearn.linear_model import LogisticRegression

    pick = int(np.argmin(member_eer))
    logistic = LogisticRegression(class_weight="balanced", C=1e6).fit(dev_z[calib], y_dev[calib])
    gate = ConditionGate(z_dev.shape[1], len(runs), n_anchors=n_anchors)
    info = gate.fit(dev_z[calib], z_dev[calib], y_dev[calib], dev.loc[calib, "group"].to_numpy(),
                    stop[calib], steps=steps, seed=cfg.seed)
    log.info("best_single is %s | logistic weights %s | gate step %d/%d, holdout Cllr %.4f, "
             "global mixture %s", runs[pick], np.round(logistic.coef_[0], 3).tolist(),
             info["best_step"], info["steps"], info["best_holdout_cllr"], np.round(info["global_weights"], 3).tolist())

    def combine(z: np.ndarray, zz: np.ndarray) -> dict[str, np.ndarray]:
        gated = gate.transform(z, zz)
        return {"best_single": z[:, pick], "mean": z.mean(axis=1),
                "logistic": logistic.decision_function(z), "global_mix": gated["global"],
                "gated": gated["gated"], "null_weight": gated["null_weight"]}

    results: dict = {"runs": runs, "views": views, "best_single": runs[pick],
                     "descriptor_label_probe_eer": {"within_attack": probe_within,
                                                    "cross_attack": probe_cross},
                     "member_dev_select_eer": dict(zip(runs, member_eer.tolist(), strict=True)),
                     "gate": {k: v for k, v in info.items() if k != "curve"},
                     "logistic_weights": logistic.coef_[0].tolist(),
                     "scale": {"mean": mu.tolist(), "std": sd.tolist()}}

    # --- the diagnosis: how much do the members actually differ? ---
    for part, z in (("dev_select", dev_z[select]), ):
        corr = np.corrcoef(z.T)
        off = float((corr.sum() - len(runs)) / (len(runs) ** 2 - len(runs)))
        results[f"correlation_{part}"] = {"runs": runs, "matrix": np.round(corr, 4).tolist(), "mean_offdiag": off}
        log.info("%s member correlation: mean off-diagonal %.3f\n%s", part, off,
                 pd.DataFrame(np.round(corr, 3), index=runs, columns=range(len(runs))).to_string())

    # --- dev select: the only place a combiner may be chosen ---
    dev_out = dev.drop(columns="z").assign(**combine(dev_z, z_dev))
    results["dev_select"] = report(dev_out[select].assign(all="all"), "all", n_boot, cfg.seed)["all"]
    results["dev_select_per_view"] = report(dev_out[select], "view", n_boot, cfg.seed)
    dev_out[select].to_parquet(out_dir / "scores_dev_select.parquet")

    # --- eval: scored once, reported pooled, per real codec and per attack ---
    ev, ev_raw = read_set(cfg, runs, "eval_clean", ["utt", "label", "codec", "generator"])
    ev_z = (ev_raw - mu) / sd
    ev_out = ev.drop(columns="z").assign(**combine(ev_z, np.stack(ev["z"])))
    corr = np.corrcoef(ev_z.T)
    results["correlation_eval"] = {"runs": runs, "matrix": np.round(corr, 4).tolist(),
                                   "mean_offdiag": float((corr.sum() - len(runs)) / (len(runs) ** 2 - len(runs)))}
    log.info("eval member correlation: mean off-diagonal %.3f", results["correlation_eval"]["mean_offdiag"])
    log.info("== eval, pooled | combiners %s | SOTA to beat: 2.59%% EER / 0.0750 minDCF", COMBINERS)
    results["eval_pooled"] = report(ev_out.assign(all="all"), "all", n_boot, cfg.seed)["all"]
    log.info("== eval, per real codec (C04 and C07 are the neural ones, C11 is devices)")
    results["eval_per_codec"] = report(ev_out, "codec", n_boot, cfg.seed)

    log.info("== eval, per attack (A18, A20, A23, A27, A30, A31, A32 are the adversarial ones)")
    bona = ev_out["label"].to_numpy() == 1
    per_attack = {}
    for attack in sorted(ev_out.loc[~bona, "generator"].unique()):
        sub = ev_out[bona | (ev_out["generator"] == attack).to_numpy()]
        y = sub["label"].to_numpy()
        per_attack[str(attack)] = {"n": len(sub),
                                   **{f"eer/{c}": compute_eer(sub[c].to_numpy(), y)[0] for c in COMBINERS}}
        log.info("%-5s n=%7d | EER %s", attack, len(sub),
                 " ".join(f"{per_attack[str(attack)][f'eer/{c}']:.4f}" for c in COMBINERS))
    results["eval_per_attack"] = per_attack
    ev_out.to_parquet(out_dir / "scores_eval_clean.parquet")

    write_json(out_dir / "ensemble.json", results)
    log.info("wrote %s", out_dir / "ensemble.json")


if __name__ == "__main__":
    main()
