"""Measure adversarial robustness of one detector or one ensemble, white-box and black-box.

    # the single-model control
    uv run python scripts/attack.py +study=adv_single "+runs=[fullenc_seed1337]"
    # the ensemble, same attack set and budgets
    uv run python scripts/attack.py +study=adv_bands "+runs=[band0_7,band7_13,band13_19,band19_25]"
    # transfer: craft on the surrogate, score on the system under test
    uv run python scripts/attack.py +study=adv_transfer "+runs=[band...]" "+surrogate=[fullenc_seed1337]"

Claim served: the ensemble's robustness is its own measured result, never an assumption
(docs/experiment_log.md, project goal of 2026-10-09). The two halves of the threat model are
reported separately, as decided on 2026-10-13:

  white-box  PGD on the waveform against the exact system under test, including its combiner
             weights. ``+eot`` raises the expectation count for a stochastic system.
  black-box  SPSA with a stated query budget, and transfer from ``+surrogate``. Plus, at no extra
             cost, the Malafide / Malacopula attacks already in the eval set (A18, A20, A23, A27,
             A30, A31, A32; arXiv:2408.08739 Table 2), which are real black-box attacks on a
             detector the attacker did not have.

Attack set. A gradient attack needs the backbone in the loop, so the full 680,774-clip eval set is
out of reach: the attacks run on a stratified subsample (``+n_per_cell`` clips per codec and class,
plus every clip of the adversarial attacks up to the same cap), cropped to ``+attack_seconds``. The
clean metrics of that subsample are reported beside the attacked ones, so the drop is read against
the right baseline and never against the full-eval number.

Reported per budget: EER, and the attacker's success rate at the threshold that the CLEAN subsample
puts at 0.1% and 1% false-accept -- the rate at which a spoof clip crosses a fixed operating point
is what a deployment cares about. The realised SNR of every perturbation is reported too, so the
budget is measured rather than assumed.

Writes experiments/runs/<study>/: config.json, attack.json, scores_attack.parquet.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from caesd.attacks import l2_budget, pgd, snr_db, spsa
from caesd.data.datasets import AudioDataset, collate
from caesd.eval import calib_scale
from caesd.metrics import compute_eer, far_frr_at_threshold, min_dcf, threshold_at_far
from caesd.models.ensemble import ScoreEnsemble
from caesd.utils.logging import get_logger, log_hparams, write_json
from caesd.utils.seed import set_seed

log = get_logger("caesd.attack")
ADVERSARIAL_ATTACKS = ("A18", "A20", "A23", "A27", "A30", "A31", "A32")


def attack_set(manifest: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    """A stratified subsample of the eval manifest: ``n_per_cell`` clips per (codec, class), and
    every adversarial-attack clip up to the same cap per codec. Small enough for PGD, and it still
    covers every condition the full-eval table reports."""
    rng = np.random.default_rng(seed)

    def take(sub: pd.DataFrame, n: int) -> pd.DataFrame:
        return sub if len(sub) <= n else sub.iloc[rng.choice(len(sub), n, replace=False)]

    parts = []
    for (_codec, _label), sub in manifest.groupby(["codec", "label"]):
        parts.append(take(sub, n_per_cell))
    # The adversarial attacks are sampled more thinly per codec: they span every codec, so the same
    # cap would make them four fifths of the set and leave no PGD budget for the ordinary conditions.
    for attack in ADVERSARIAL_ATTACKS:
        sub = manifest[manifest["generator"] == attack]
        if sub.empty:
            raise RuntimeError(f"{attack} is absent from the eval manifest: the attack list is wrong")
        for _codec, by_codec in sub.groupby("codec"):
            parts.append(take(by_codec, max(1, n_per_cell // 4)))
    out = pd.concat(parts).drop_duplicates(subset="utt").reset_index(drop=True)
    log.info("attack set: %d clips | bona fide %d | codecs %d | attacks %d", len(out),
             int((out["label"] == 1).sum()), out["codec"].nunique(), out["generator"].nunique())
    return out


def score_all(score_fn, loader: DataLoader, device: str, attack=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every batch, optionally after ``attack(wave, lengths, goal)``. Returns the clip index,
    the score and the realised SNR (``inf`` where nothing was perturbed)."""
    idx, score, snr = [], [], []
    for batch in loader:
        wave, lengths = batch["x"].to(device), batch["lengths"].to(device)
        if attack is None:
            with torch.no_grad():
                out = score_fn(wave, lengths)
            realised = torch.full_like(out, float("inf"))
        else:
            # The attacker pushes each clip towards the WRONG class: a spoof clip up, a bona fide
            # clip down. One pass per direction keeps the per-clip goal exact.
            goal = torch.where(batch["label"].to(device) == 1, -1, 1)
            attacked = wave.clone()
            for direction in (1, -1):
                rows = goal == direction
                if rows.any():
                    attacked[rows] = attack(wave[rows], lengths[rows], int(direction))
            with torch.no_grad():
                out = score_fn(attacked, lengths)
            realised = snr_db(wave, attacked, lengths)
        idx.append(batch["idx"].numpy())
        score.append(out.float().cpu().numpy())
        snr.append(realised.float().cpu().numpy())
    order = np.argsort(np.concatenate(idx))
    return np.concatenate(idx)[order], np.concatenate(score)[order], np.concatenate(snr)[order]


def report(label: np.ndarray, clean: np.ndarray, attacked: np.ndarray, snr: np.ndarray) -> dict:
    """EER under attack, and the share of spoof clips that cross the CLEAN thresholds at 0.1% and
    1% false-accept. The thresholds come from the clean scores, because a deployment fixes its
    operating point before it is attacked."""
    out = {"n": int(label.size), "eer": compute_eer(attacked, label)[0], "min_dcf": min_dcf(attacked, label),
           "median_snr_db": float(np.median(snr[np.isfinite(snr)])) if np.isfinite(snr).any() else None}
    spoof = label == 0
    for far in (0.001, 0.01):
        thr = threshold_at_far(clean, label, far)
        out[f"success_rate_at_far{far}"] = float((attacked[spoof] > thr).mean())
        out[f"frr_at_far{far}"] = far_frr_at_threshold(attacked, label, thr)[1]
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.paths.runs) / cfg.study
    out_dir.mkdir(parents=True)  # raises if the study exists: results are never overwritten
    resolved = OmegaConf.to_container(cfg, resolve=True)
    log_hparams(log, resolved)
    write_json(out_dir / "config.json", resolved)
    log.info("seed=%d", set_seed(cfg.seed))
    runs, runs_dir = list(cfg.runs), Path(cfg.paths.runs)
    views = list(cfg.get("views", ["clean", "codec"]))
    device, n_per_cell = cfg.device, cfg.get("n_per_cell", 60)
    steps, eot, snrs = cfg.get("steps", 20), cfg.get("eot", 1), list(cfg.get("snr_db", [30, 20]))

    # --- the system under test: an unweighted mean of standardised members (the combiner that won
    #     on dev; see the ensemble study). A single run is the same code path with one member. ---
    mean, std = calib_scale(runs_dir, runs, views)
    weights = torch.full((len(runs),), 1.0 / len(runs))
    model = ScoreEnsemble.from_runs(runs_dir, runs, torch.tensor(mean), torch.tensor(std), weights, device)
    log.info("system under test: %d member(s) %s | backbones %s", len(runs), runs, list(model.backbones))

    manifest = pd.read_parquet(Path(cfg.paths.manifests) / "eval.parquet")
    meta = attack_set(manifest, n_per_cell, cfg.seed)
    dataset = AudioDataset(meta, cfg.get("attack_seconds", 4.0))
    loader = DataLoader(dataset, batch_size=cfg.get("attack_batch_size", 8), collate_fn=collate,
                        num_workers=cfg.data.num_workers)
    label = meta["label"].to_numpy()

    _, clean, _ = score_all(model.score_fn(), loader, device)
    results: dict = {"runs": runs, "n_per_cell": n_per_cell, "steps": steps, "eot": eot,
                     "clean": report(label, clean, clean, np.full_like(clean, np.inf))}
    log.info("clean subsample: EER %.4f minDCF %.4f", results["clean"]["eer"], results["clean"]["min_dcf"])
    frame = meta[["utt", "label", "codec", "generator"]].assign(clean=clean)

    # --- black-box, free: the Malafide / Malacopula attacks already in the eval set ---
    bona = label == 1
    results["benchmark_adversarial_attacks"] = {}
    for attack in ADVERSARIAL_ATTACKS:
        rows = bona | (meta["generator"] == attack).to_numpy()
        results["benchmark_adversarial_attacks"][attack] = report(
            label[rows], clean[rows], clean[rows], np.full(int(rows.sum()), np.inf))
        log.info("benchmark attack %s: EER %.4f", attack, results["benchmark_adversarial_attacks"][attack]["eer"])

    # --- white-box PGD, one budget at a time ---
    surrogate = None
    if cfg.get("surrogate"):
        s_runs = list(cfg.surrogate)
        s_mean, s_std = calib_scale(runs_dir, s_runs, views)
        surrogate = ScoreEnsemble.from_runs(runs_dir, s_runs, torch.tensor(s_mean), torch.tensor(s_std),
                                            torch.full((len(s_runs),), 1.0 / len(s_runs)), device)
        log.info("transfer surrogate: %s", s_runs)

    for snr_target in snrs:
        for name, crafter in (("pgd_whitebox", model), ("pgd_transfer", surrogate)):
            if crafter is None:
                continue
            fn = crafter.score_fn()

            def craft(wave, lengths, goal, fn=fn, snr_target=snr_target):
                return pgd(fn, wave, lengths, goal, l2_budget(wave, lengths, snr_target),
                           norm="l2", steps=steps, eot=eot)

            _, adv_score, snr = score_all(model.score_fn(), loader, device, attack=craft)
            key = f"{name}_snr{snr_target}"
            results[key] = report(label, clean, adv_score, snr)
            frame[key] = adv_score
            log.info("%-22s | EER %.4f (clean %.4f) | success@FAR0.1%% %.4f | median SNR %.1f dB", key,
                     results[key]["eer"], results["clean"]["eer"],
                     results[key]["success_rate_at_far0.001"], results[key]["median_snr_db"])

    # --- black-box SPSA, one budget, with its query budget stated ---
    if cfg.get("spsa", True):
        snr_target = snrs[-1]  # the largest perturbation only: SPSA is the expensive half
        n_samples = cfg.get("spsa_samples", 8)

        def craft_spsa(wave, lengths, goal):
            attacked, _ = spsa(model.score_fn(), wave, lengths, goal,
                               l2_budget(wave, lengths, snr_target), norm="l2", steps=steps,
                               n_samples=n_samples)
            return attacked

        _, adv_score, snr = score_all(model.score_fn(), loader, device, attack=craft_spsa)
        key = f"spsa_blackbox_snr{snr_target}"
        results[key] = {**report(label, clean, adv_score, snr), "queries_per_clip": steps * 2 * n_samples}
        frame[key] = adv_score
        log.info("%-22s | EER %.4f | success@FAR0.1%% %.4f | %d queries/clip", key, results[key]["eer"],
                 results[key]["success_rate_at_far0.001"], results[key]["queries_per_clip"])

    frame.to_parquet(out_dir / "scores_attack.parquet")
    write_json(out_dir / "attack.json", results)
    log.info("wrote %s", out_dir / "attack.json")


if __name__ == "__main__":
    main()
