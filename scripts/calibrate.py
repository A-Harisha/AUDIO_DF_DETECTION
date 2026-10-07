"""Calibration study: does condition-dependent calibration (block C) give better LLRs under codec
shift than one global calibration, including on conditions it never saw?

    uv run python scripts/calibrate.py +study=cal_frozen_xlsr "+runs=[frozen_xlsr]"
    uv run python scripts/calibrate.py +study=cal_fused "+runs=[frozen_xlsr,frozen_wavlm]" +eval=true

Claim served: the core paper claim, and its go/no-go premise. Everything is fit on the dev
``calib`` split and tested on data the calibrators never saw:
  - dev ``select`` split, per simulated codec (seen conditions);
  - leave-one-codec-out on dev: that codec is removed from calibration AND from the codec
    classifier, then tested (unseen simulated condition);
  - with ``+eval=true``: the ASVspoof 5 eval set per real codec. C04, C07 and C11 are never
    simulated, so they are unseen real conditions.

Methods (score -> LLR):
  global_clean   one logistic calibration fit on clean speech (the common practice)
  global_pooled  one logistic calibration fit on clean + all simulated codecs
  lookup_oracle  one calibration per codec, chosen with the TRUE codec label; a condition with
                 no cell uses global_pooled and is reported as uncovered
  lookup_pred    one calibration per codec, chosen by the codec classifier
  hypernet       block C: (a, b) generated from the codec posterior

With several ``runs`` the scores are first fused by logistic regression fit on the calib split.
Metrics use the ASVspoof 5 prior (p_target = 0.05). Writes experiments/runs/<study>/.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from caesd.calibration import LogisticCalibrator, LookupTableCalibrator
from caesd.data.condition_grid import EVAL_CODEC
from caesd.data.datasets import CODEC_NAMES, FeatureLoader
from caesd.metrics import act_dcf, cllr, compute_eer, min_cllr, min_dcf
from caesd.models import CalibrationHypernet, Head
from caesd.utils.logging import get_logger, log_hparams, write_json
from caesd.utils.seed import set_seed

log = get_logger("caesd.calibrate")
METHODS = ("global_clean", "global_pooled", "lookup_oracle", "lookup_pred", "hypernet")


def features(cfg: DictConfig, name: str) -> tuple[torch.Tensor, pd.DataFrame]:
    root = Path(cfg.paths.features) / cfg.backbone.tag
    meta = pd.read_parquet(root / f"{name}.parquet")
    return torch.from_numpy(np.load(root / f"{name}.npy")), meta


def fit_condition_classifier(x: torch.Tensor, channel: np.ndarray, exclude: int | None, cfg: DictConfig) -> Head:
    """Codec classifier on frozen features, separate from the detector. ``exclude`` removes one
    codec class from its training data (leave-one-codec-out)."""
    keep = np.flatnonzero(channel != exclude)
    loader = FeatureLoader(x[keep], np.zeros(len(keep)), channel[keep], 1024, shuffle=True)
    head = Head(x.shape[1], x.shape[2], n_out=len(CODEC_NAMES)).to(cfg.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    step, steps = 0, cfg.get("condition_steps", 1500)
    head.train()
    while step < steps:
        for batch in loader:
            loss = F.cross_entropy(head(batch["x"].to(cfg.device))["score"], batch["channel"].to(cfg.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    return head.eval()


@torch.no_grad()
def condition_vector(head: Head, x: torch.Tensor, device: str) -> np.ndarray:
    """Codec log-posterior [N, n_codecs]: the only input block C sees besides the score."""
    out = [F.log_softmax(head(x[i : i + 4096].to(device))["score"], -1).cpu() for i in range(0, len(x), 4096)]
    return torch.cat(out).numpy()


def calibrate(calib: pd.DataFrame, test: pd.DataFrame, cond_calib: np.ndarray, cond_test: np.ndarray,
              seed: int) -> pd.DataFrame:
    """Fit every method on ``calib`` and return ``test`` with one LLR column per method."""
    s, y, group = (calib[c].to_numpy() for c in ("score", "label", "group"))
    out, ts = test.copy(), test["score"].to_numpy()
    out["global_clean"] = LogisticCalibrator().fit(s[group == "clean"], y[group == "clean"]).transform(ts)
    out["global_pooled"] = LogisticCalibrator().fit(s, y).transform(ts)
    table = LookupTableCalibrator(min_per_cell=200).fit(s, y, group)
    for name, cells in (("lookup_oracle", test["cell"].to_numpy()),
                        ("lookup_pred", np.asarray(CODEC_NAMES)[cond_test.argmax(1)])):
        llr, covered = table.transform(ts, cells)
        out[name] = np.where(covered, llr, out["global_pooled"])  # explicit, and reported below
        out[f"{name}_covered"] = covered
    hyper = CalibrationHypernet(cond_calib.shape[1])
    curve = hyper.fit(s, cond_calib, y, group, seed=seed)
    log.info("hypernet loss %.4f -> %.4f", curve[0], curve[-1])
    out["hypernet"] = hyper.transform(ts, cond_test)
    return out


def paired_ci(metric, llr_a: np.ndarray, llr_b: np.ndarray, labels: np.ndarray, n: int, seed: int) -> list[float]:
    """95% bootstrap CI of metric(a) - metric(b) on the same resampled trials."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, labels.size, labels.size)
        if labels[idx].min() != labels[idx].max():
            diffs.append(metric(llr_a[idx], labels[idx]) - metric(llr_b[idx], labels[idx]))
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def report(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Per condition group: discrimination, then calibration quality of each method, and the
    paired difference hypernet - global_pooled with a 95% CI (negative = block C is better)."""
    rows = {}
    for group, sub in df.groupby("group"):
        y, raw = sub["label"].to_numpy(), sub["score"].to_numpy()
        if np.unique(y).size < 2:
            continue
        row = {"n": len(sub), "eer": compute_eer(raw, y)[0], "min_dcf": min_dcf(raw, y), "min_cllr": min_cllr(raw, y),
               "oracle_covered": float(sub["lookup_oracle_covered"].mean())}
        for m in METHODS:
            llr = sub[m].to_numpy()
            row[f"eer/{m}"] = compute_eer(llr, y)[0]  # differs from "eer" only if the method changes discrimination
            row[f"act_dcf/{m}"] = act_dcf(llr, y)
            row[f"cllr/{m}"] = cllr(llr, y)
        for name, metric in (("act_dcf", act_dcf), ("cllr", cllr)):
            row[f"{name}/hypernet-global_pooled/ci95"] = paired_ci(
                metric, sub["hypernet"].to_numpy(), sub["global_pooled"].to_numpy(), y, n_boot, seed)
        rows[str(group)] = row
        log.info("%-10s n=%6d EER %.4f (hypernet LLR %.4f) minDCF %.3f | actDCF %s | Cllr %s (min %.3f) | "
                 "d_Cllr(hyper-pooled) CI [%.3f, %.3f]", group, row["n"], row["eer"], row["eer/hypernet"], row["min_dcf"],
                 " ".join(f"{row[f'act_dcf/{m}']:.3f}" for m in METHODS),
                 " ".join(f"{row[f'cllr/{m}']:.3f}" for m in METHODS), row["min_cllr"],
                 *row["cllr/hypernet-global_pooled/ci95"])
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
    frames = {}
    for name in sets:
        parts = [pd.read_parquet(Path(cfg.paths.runs) / run / f"scores_{name}.parquet") for run in cfg.runs]
        frame = parts[0].drop(columns="score")
        frame["scores"] = list(np.stack([p["score"].to_numpy() for p in parts], axis=1))
        frames[name] = frame
    dev = pd.concat([frames["dev_clean"], frames["dev_codec"]], ignore_index=True)
    dev["group"] = dev["cell"] = dev["sim_codec"]
    is_calib = (dev["split"] == "calib").to_numpy()
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
    log.info("dev: calib %d, select %d | groups %s", is_calib.sum(), (~is_calib).sum(), dev["group"].value_counts().to_dict())

    # --- condition vectors from a stand-alone codec classifier (trained on the train partition) ---
    train = [features(cfg, f"train_{v}") for v in ("clean", "codec")]
    xt = torch.cat([x for x, _ in train])
    ct = np.concatenate([meta["sim_codec"].map(CODEC_NAMES.index).to_numpy() for _, meta in train])
    del train
    xd = torch.cat([features(cfg, n)[0] for n in ("dev_clean", "dev_codec")])
    clf = fit_condition_classifier(xt, ct, None, cfg)
    cond_dev = condition_vector(clf, xd, cfg.device)
    truth = dev["sim_codec"].map(CODEC_NAMES.index).to_numpy()
    accuracy = {c: float((cond_dev[truth == i].argmax(1) == i).mean()) for i, c in enumerate(CODEC_NAMES)}
    log.info("codec classifier accuracy on dev: %s", {k: round(v, 3) for k, v in accuracy.items()})
    results: dict = {"runs": list(cfg.runs), "codec_classifier_accuracy": accuracy}

    # --- 1. seen conditions: fit on calib, test on select ---
    log.info("== dev select, seen simulated codecs | methods: %s", METHODS)
    seen = calibrate(dev[is_calib], dev[~is_calib], cond_dev[is_calib], cond_dev[~is_calib], cfg.seed)
    seen.drop(columns="scores").to_parquet(out_dir / "llr_dev_seen.parquet")
    results["dev_seen"] = report(seen, n_boot, cfg.seed)

    # --- 2. leave-one-codec-out: the codec is unseen by calibration and by the codec classifier ---
    log.info("== dev select, leave-one-codec-out (unseen simulated condition)")
    loco = []
    for i, codec in enumerate(CODEC_NAMES[1:], start=1):
        cond = condition_vector(fit_condition_classifier(xt, ct, i, cfg), xd, cfg.device)
        fit_rows = is_calib & (truth != i)
        test_rows = ~is_calib & (truth == i)
        loco.append(calibrate(dev[fit_rows], dev[test_rows], cond[fit_rows], cond[test_rows], cfg.seed))
    loco = pd.concat(loco, ignore_index=True)
    loco.drop(columns="scores").to_parquet(out_dir / "llr_dev_loco.parquet")
    results["dev_leave_one_codec_out"] = report(loco, n_boot, cfg.seed)
    del xt

    # --- 3. the eval set: real codecs, three of them never simulated ---
    if cfg.get("eval", False):
        log.info("== eval, real codecs (C04, C07, C11 are unseen conditions)")
        ev = frames["eval_clean"]
        ev["score"], ev["group"] = fused(ev), ev["codec"]
        ev["cell"] = ev["codec"].map(lambda c: EVAL_CODEC.get(c, c))  # clean -> clean; unmapped -> no cell
        cond_ev = condition_vector(clf, features(cfg, "eval_clean")[0], cfg.device)
        out = calibrate(dev[is_calib], ev, cond_dev[is_calib], cond_ev, cfg.seed)
        out.drop(columns="scores").to_parquet(out_dir / "llr_eval.parquet")
        results["eval"] = report(out, n_boot, cfg.seed)
        log.info("== eval, pooled over all conditions (the benchmark numbers)")
        results["eval_pooled"] = report(out.assign(group="all"), n_boot, cfg.seed)["all"]
        results["eval_runs_eer"] = {run: compute_eer(np.stack(ev["scores"])[:, j], ev["label"].to_numpy())[0]
                                    for j, run in enumerate(cfg.runs)}
    write_json(out_dir / "calibration.json", results)
    log.info("wrote %s", out_dir / "calibration.json")


if __name__ == "__main__":
    main()
