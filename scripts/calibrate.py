"""Calibration study: does condition-dependent calibration (block C) give better LLRs under codec
shift than one global calibration, including on conditions it never saw?

    uv run python scripts/calibrate.py +study=cal_frozen_xlsr "+runs=[frozen_xlsr]"
    uv run python scripts/calibrate.py +study=cal_fused "+runs=[frozen_xlsr,frozen_wavlm]" +eval=true

Claim served: the core paper claim, and its go/no-go premise. Everything is fit on the dev
``calib`` split and tested on data the calibrators never saw:
  - dev ``select`` split, per simulated codec (seen conditions);
  - leave-one-codec-out on dev: that codec is removed from calibration (and from the lookup_pred
    baseline's codec classifier), then tested (unseen simulated condition);
  - with ``+eval=true``: the ASVspoof 5 eval set per real codec. C04, C07 and C11 are never
    simulated, so they are unseen real conditions.

Methods (score -> LLR):
  global_clean   one logistic calibration fit on clean speech (the common practice)
  global_pooled  one logistic calibration fit on clean + all simulated codecs
  lookup_oracle  one calibration per codec, chosen with the TRUE codec label; a condition with
                 no cell uses global_pooled and is reported as uncovered
  lookup_pred    one calibration per codec, chosen by a codec classifier on the descriptor
  hypernet       block C: (a, b) generated from the channel descriptor, anchored to global

Block C and lookup_pred read ONLY the cached channel descriptor, never the SSL features the
detector uses, so neither can carry class evidence. ``descriptor_label_probe_eer`` in the output
is the precondition for every calibration claim here: it must sit near 0.5. Block C is fit on
speaker-disjoint ``fit`` / ``stop`` halves of the calib split and run over ``+hyper_seeds``, so
its variance is measured rather than assumed.

With several ``runs`` the scores are first fused by logistic regression fit on the calib split.
Metrics use the ASVspoof 5 cost model (spoof prior 0.05, C_miss 1, C_fa 10). Writes experiments/runs/<study>/.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from caesd.calibration import LogisticCalibrator, LookupTableCalibrator
from caesd.data.condition_grid import EVAL_CODEC
from caesd.data.datasets import CODEC_NAMES
from caesd.data.manifests import assign_split_by
from caesd.eval import descriptor as read_descriptor
from caesd.metrics import act_dcf, cllr, compute_eer, min_cllr, min_dcf, paired_ci
from caesd.models import CalibrationHypernet, label_probe_eer
from caesd.utils.logging import get_logger, log_hparams, write_json
from caesd.utils.seed import set_seed

log = get_logger("caesd.calibrate")
METHODS = ("global_clean", "global_pooled", "lookup_oracle", "lookup_pred", "hypernet")


def descriptor(cfg: DictConfig, name: str) -> pd.DataFrame:
    """The cached channel descriptor of set ``name``. Body shared with the ensemble study."""
    return read_descriptor(Path(cfg.paths.features), name)


def codec_classifier(z: np.ndarray, channel: np.ndarray, exclude: int | None) -> np.ndarray:
    """Predicted codec NAME per row of ``z``, from a classifier on the channel descriptor only.
    Used by the lookup_pred baseline. ``exclude`` removes one codec class from its training data
    (leave-one-codec-out)."""
    from sklearn.linear_model import LogisticRegression

    keep = np.flatnonzero(channel != exclude)
    clf = LogisticRegression(max_iter=1000).fit(z[keep], channel[keep])
    return np.asarray(CODEC_NAMES)[clf.predict(z)]


def calibrate(calib: pd.DataFrame, test: pd.DataFrame, pred_cell: np.ndarray,
              seeds: list[int], steps: int) -> pd.DataFrame:
    """Fit every method on ``calib`` and return ``test`` with one LLR column per method.

    ``calib`` carries ``z`` (the channel descriptor) and ``stop`` (the speaker-disjoint rows block
    C selects its step on). ``pred_cell`` is the lookup_pred baseline's predicted codec per test
    row. Block C is fit once per seed: ``hypernet`` is the first seed, and the rest measure the
    run-to-run variance that v1 never reported."""
    s, y, group, stop = (calib[c].to_numpy() for c in ("score", "label", "group", "stop"))
    z_calib, z_test = np.stack(calib["z"]), np.stack(test["z"])
    out, ts = test.copy(), test["score"].to_numpy()
    out["global_clean"] = LogisticCalibrator().fit(s[group == "clean"], y[group == "clean"]).transform(ts)
    out["global_pooled"] = LogisticCalibrator().fit(s, y).transform(ts)
    table = LookupTableCalibrator(min_per_cell=200).fit(s, y, group)
    for name, cells in (("lookup_oracle", test["cell"].to_numpy()), ("lookup_pred", pred_cell)):
        llr, covered = table.transform(ts, cells)
        out[name] = np.where(covered, llr, out["global_pooled"])  # explicit, and reported below
        out[f"{name}_covered"] = covered
    for seed in seeds:
        hyper = CalibrationHypernet(z_calib.shape[1])
        info = hyper.fit(s, z_calib, y, group, stop, steps=steps, seed=seed)
        log.info("block C seed %d: step %d / %d | holdout Cllr %.4f (at start %.4f)", seed,
                 info["best_step"], info["steps"], info["best_holdout_cllr"], info["curve"][0][1])
        out[f"hypernet_s{seed}"] = hyper.transform(ts, z_test)
        out[f"null_weight_s{seed}"] = hyper.null_weight(z_test)
    out["hypernet"] = out[f"hypernet_s{seeds[0]}"]
    out["null_weight"] = out[f"null_weight_s{seeds[0]}"]
    return out


def report(df: pd.DataFrame, n_boot: int, seed: int, seeds: list[int]) -> dict:
    """Per condition group: discrimination, then calibration quality of each method, the paired
    difference hypernet - global_pooled with a 95% CI (negative = block C is better), block C's
    spread over ``seeds``, and ``null_weight`` -- the mean weight on block C's null anchor, i.e.
    the share of the clip that got the plain global calibration. It says when C acted at all."""
    rows = {}
    for group, sub in df.groupby("group"):
        y, raw = sub["label"].to_numpy(), sub["score"].to_numpy()
        if np.unique(y).size < 2:
            continue
        row = {"n": len(sub), "eer": compute_eer(raw, y)[0], "min_dcf": min_dcf(raw, y), "min_cllr": min_cllr(raw, y),
               "oracle_covered": float(sub["lookup_oracle_covered"].mean()),
               "null_weight": float(sub["null_weight"].mean())}
        for m in METHODS:
            llr = sub[m].to_numpy()
            row[f"eer/{m}"] = compute_eer(llr, y)[0]  # differs from "eer" only if the method changes discrimination
            row[f"act_dcf/{m}"] = act_dcf(llr, y)
            row[f"cllr/{m}"] = cllr(llr, y)
        for name, metric in (("act_dcf", act_dcf), ("cllr", cllr)):
            per_seed = [metric(sub[f"hypernet_s{s}"].to_numpy(), y) for s in seeds]
            row[f"{name}/hypernet/seed_mean"] = float(np.mean(per_seed))
            row[f"{name}/hypernet/seed_std"] = float(np.std(per_seed))
            row[f"{name}/hypernet-global_pooled/ci95"] = paired_ci(
                metric, sub["hypernet"].to_numpy(), sub["global_pooled"].to_numpy(), y, n_boot, seed)
        rows[str(group)] = row
        log.info("%-10s n=%6d EER %.4f (hypernet LLR %.4f) minDCF %.3f | actDCF %s | Cllr %s (min %.3f) | "
                 "d_Cllr(hyper-pooled) CI [%.3f, %.3f] | Cllr(C) %.3f+-%.3f over %d seeds | null_w %.2f",
                 group, row["n"], row["eer"], row["eer/hypernet"], row["min_dcf"],
                 " ".join(f"{row[f'act_dcf/{m}']:.3f}" for m in METHODS),
                 " ".join(f"{row[f'cllr/{m}']:.3f}" for m in METHODS), row["min_cllr"],
                 *row["cllr/hypernet-global_pooled/ci95"],
                 row["cllr/hypernet/seed_mean"], row["cllr/hypernet/seed_std"], len(seeds),
                 row["null_weight"])
    return rows


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.paths.runs) / cfg.study
    out_dir.mkdir(parents=True)  # raises if the study exists: results are never overwritten
    resolved = OmegaConf.to_container(cfg, resolve=True)
    log_hparams(log, resolved)
    write_json(out_dir / "config.json", resolved)
    set_seed(cfg.seed)
    n_boot, sets = cfg.get("n_boot", 200), ["dev_clean", "dev_codec"] + (["eval_clean"] if cfg.get("eval", False) else [])

    # --- scores (fused over runs), one frame per set ---
    seeds, steps = list(cfg.get("hyper_seeds", [0, 1, 2])), cfg.get("hyper_steps", 2000)
    frames = {}
    for name in sets:
        parts = [pd.read_parquet(Path(cfg.paths.runs) / run / f"scores_{name}.parquet") for run in cfg.runs]
        frame = parts[0].drop(columns="score")
        frame["scores"] = list(np.stack([p["score"].to_numpy() for p in parts], axis=1))
        # The channel descriptor, joined on utt: block C's only input besides the score.
        frames[name] = frame.merge(descriptor(cfg, name).drop(columns="sim_codec"),
                                   on="utt", how="inner", validate="one_to_one")
        if len(frames[name]) != len(frame):
            raise RuntimeError(f"{name}: {len(frame)} scored clips, {len(frames[name])} with a descriptor")
    dev = pd.concat([frames["dev_clean"], frames["dev_codec"]], ignore_index=True)
    dev["group"] = dev["cell"] = dev["sim_codec"]
    is_calib = (dev["split"] == "calib").to_numpy()
    # Block C selects its step on speakers it did not fit on: v1 ran a fixed 2000 steps on all of
    # the calibration data and overfit it (docs/experiment_log.md, 2026-10-07).
    dev["stop"] = False
    buckets = assign_split_by(dev.loc[is_calib, ["speaker"]], "speaker", (0.75, 0.25, 0.0), cfg.seed + 1)
    dev.loc[is_calib, "stop"] = (buckets["split"] == "dev").to_numpy()
    fusion = None
    if len(cfg.runs) > 1:
        from sklearn.linear_model import LogisticRegression
        fusion = LogisticRegression(class_weight="balanced", C=1e6).fit(
            np.stack(dev.loc[is_calib, "scores"]), dev.loc[is_calib, "label"])
        log.info("fusion weights %s bias %s", fusion.coef_.tolist(), fusion.intercept_.tolist())

    def fused(frame: pd.DataFrame) -> np.ndarray:
        s = np.stack(frame["scores"])
        return s[:, 0] if fusion is None else fusion.decision_function(s)

    dev["score"] = fused(dev)
    log.info("dev: calib %d (fit %d / stop %d), select %d | groups %s", is_calib.sum(),
             (is_calib & ~dev["stop"]).sum(), dev["stop"].sum(), (~is_calib).sum(),
             dev["group"].value_counts().to_dict())

    # --- the PRECONDITION: the channel descriptor must not predict the class ---
    z_dev, y_dev = np.stack(dev["z"]), dev["label"].to_numpy()
    probe_eer = label_probe_eer(z_dev[is_calib], y_dev[is_calib], z_dev[~is_calib], y_dev[~is_calib])
    log.info("PRECONDITION | label probe on the descriptor alone: dev select EER %.4f "
             "(must be ~0.5; below that, block C carries class evidence and no claim holds)", probe_eer)
    results: dict = {"runs": list(cfg.runs), "hyper_seeds": seeds,
                     "descriptor_label_probe_eer": probe_eer}

    # --- the lookup_pred baseline's codec classifier, on the descriptor only ---
    train_desc = pd.concat([descriptor(cfg, f"train_{v}") for v in ("clean", "codec")], ignore_index=True)
    ztr = np.stack(train_desc["z"])
    ctr = train_desc["sim_codec"].map(CODEC_NAMES.index).to_numpy()
    del train_desc
    truth = dev["sim_codec"].map(CODEC_NAMES.index).to_numpy()
    pred_dev = codec_classifier(np.vstack([ztr, z_dev]), np.concatenate([ctr, truth]), None)[len(ztr):]
    accuracy = {c: float((pred_dev[truth == i] == c).mean()) for i, c in enumerate(CODEC_NAMES)}
    log.info("codec classifier accuracy on dev: %s", {k: round(v, 3) for k, v in accuracy.items()})
    results["codec_classifier_accuracy"] = accuracy

    # --- 1. seen conditions: fit on calib, test on select ---
    log.info("== dev select, seen simulated codecs | methods: %s", METHODS)
    seen = calibrate(dev[is_calib], dev[~is_calib], pred_dev[~is_calib], seeds, steps)
    seen.drop(columns=["scores", "z"]).to_parquet(out_dir / "llr_dev_seen.parquet")
    results["dev_seen"] = report(seen, n_boot, cfg.seed, seeds)

    # --- 2. leave-one-codec-out: the codec is unseen by calibration and by the codec classifier ---
    log.info("== dev select, leave-one-codec-out (unseen simulated condition)")
    loco = []
    for i, codec in enumerate(CODEC_NAMES[1:], start=1):
        if not (truth == i).any():  # a codec with no dev rows in this study's views (e.g. EnCodec)
            continue
        fit_rows = is_calib & (truth != i)
        test_rows = ~is_calib & (truth == i)
        # Only lookup_pred needs a classifier that has not seen this codec. Block C reads the
        # descriptor directly, so v1's per-codec retraining -- the source of its 0.209 / 0.517
        # swing between runs of the same study -- is gone.
        pred = codec_classifier(np.vstack([ztr, z_dev]), np.concatenate([ctr, truth]), i)[len(ztr):]
        log.info("leave-one-codec-out %s: fit %d, test %d", codec, fit_rows.sum(), test_rows.sum())
        loco.append(calibrate(dev[fit_rows], dev[test_rows], pred[test_rows], seeds, steps))
    loco = pd.concat(loco, ignore_index=True)
    loco.drop(columns=["scores", "z"]).to_parquet(out_dir / "llr_dev_loco.parquet")
    results["dev_leave_one_codec_out"] = report(loco, n_boot, cfg.seed, seeds)

    # --- 3. the eval set: real codecs, three of them never simulated ---
    if cfg.get("eval", False):
        log.info("== eval, real codecs (C04, C07, C11 are unseen conditions)")
        ev = frames["eval_clean"]
        ev["score"], ev["group"] = fused(ev), ev["codec"]
        ev["cell"] = ev["codec"].map(lambda c: EVAL_CODEC.get(c, c))  # clean -> clean; unmapped -> no cell
        z_ev = np.stack(ev["z"])
        pred_ev = codec_classifier(np.vstack([ztr, z_ev]), np.concatenate([ctr, np.full(len(z_ev), -1)]),
                                   -1)[len(ztr):]
        out = calibrate(dev[is_calib], ev, pred_ev, seeds, steps)
        out.drop(columns=["scores", "z"]).to_parquet(out_dir / "llr_eval.parquet")
        results["eval"] = report(out, n_boot, cfg.seed, seeds)
        log.info("== eval, pooled over all conditions (the benchmark numbers)")
        results["eval_pooled"] = report(out.assign(group="all"), n_boot, cfg.seed, seeds)["all"]
        results["eval_runs_eer"] = {run: compute_eer(np.stack(ev["scores"])[:, j], ev["label"].to_numpy())[0]
                                    for j, run in enumerate(cfg.runs)}
    write_json(out_dir / "calibration.json", results)
    log.info("wrote %s", out_dir / "calibration.json")


if __name__ == "__main__":
    main()
