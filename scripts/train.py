"""Train one detector run, select it on unseen attacks, and score the evaluation sets.

    uv run python scripts/train.py experiment.name=frozen_xlsr
    uv run python scripts/train.py experiment.name=lora_xlsr experiment.features=false experiment.lora_rank=16

Claim served: a frozen SSL backbone with a learned layer weighting and a small head generalises
to unseen attacks, and each added component (LoRA, FiLM / block A, full fine-tuning) must earn
its place on unseen-attack EER. Model selection uses the dev ``select`` split (attacks A09-A16);
the eval set is only scored, never used for a decision.

Writes experiments/runs/<name>/: config.json, metrics.jsonl (one line per epoch), model.pt (best
epoch), best.json, scores_<set>.parquet.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from caesd.data.augment import CodecAugment, RawBoost
from caesd.data.datasets import CODEC_NAMES, AudioDataset, FeatureLoader, collate, seed_worker
from caesd.models import Backbone, CaesdDetector, Head
from caesd.train.loop import fit, predict
from caesd.utils.logging import get_logger, log_hparams, write_json
from caesd.utils.seed import set_seed

log = get_logger("caesd.train.cli")
_SCORE_COLUMNS = ["utt", "label", "split", "speaker", "codec", "codec_q", "generator", "sim_codec"]


def feature_set(cfg: DictConfig, name: str, split: str | None = None) -> tuple[torch.Tensor, pd.DataFrame]:
    """Cached pooled layers [N, L, D] and their metadata, optionally one split only."""
    root = Path(cfg.paths.features) / cfg.backbone.tag
    meta = pd.read_parquet(root / f"{name}.parquet")  # written last by extract_features: complete
    x = np.load(root / f"{name}.npy", mmap_mode="r")
    if split is not None:
        keep = (meta["split"] == split).to_numpy()
        x, meta = x[keep], meta[keep].reset_index(drop=True)
    return torch.from_numpy(np.ascontiguousarray(x)), meta


def feature_loader(x: torch.Tensor, meta: pd.DataFrame, batch_size: int, shuffle: bool = False) -> FeatureLoader:
    channel = meta["sim_codec"].map(CODEC_NAMES.index).to_numpy()
    return FeatureLoader(x, meta["label"].to_numpy(), channel, batch_size, shuffle)


def audio_set(cfg: DictConfig, name: str, split: str | None = None, n: int | None = None) -> tuple[DataLoader, pd.DataFrame]:
    """Evaluation loader over raw audio for the set ``<partition>_<view>`` (full length, fixed view)."""
    partition, view = name.split("_", 1)
    meta = pd.read_parquet(Path(cfg.paths.manifests) / f"{partition}.parquet")
    if split is not None:
        meta = meta[meta["split"] == split]
    if n is not None:
        meta = meta.sample(n, random_state=cfg.seed)
    meta = meta.reset_index(drop=True)
    dataset = AudioDataset(meta, cfg.data.max_seconds, codec=CodecAugment() if view == "codec" else None)
    order = np.argsort(-meta["size"].to_numpy(), kind="stable")
    bs = cfg.data.extract_batch_size
    loader = DataLoader(dataset, batch_sampler=[order[i : i + bs].tolist() for i in range(0, len(order), bs)],
                        collate_fn=collate, num_workers=cfg.data.num_workers, pin_memory=True)
    return loader, meta


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    exp = cfg.experiment
    run_dir = Path(cfg.paths.runs) / exp.name
    run_dir.mkdir(parents=True)  # raises if the run exists: results are never overwritten
    resolved = OmegaConf.to_container(cfg, resolve=True)
    log_hparams(log, resolved)
    write_json(run_dir / "config.json", resolved)
    log.info("seed=%d", set_seed(cfg.seed))
    n_channels = len(CODEC_NAMES) if exp.film else 0
    band = None if exp.band is None else tuple(exp.band)

    if exp.features:
        parts = [feature_set(cfg, f"train_{v}") for v in exp.views]
        x = torch.cat([p[0] for p in parts])
        train_meta = pd.concat([p[1] for p in parts], ignore_index=True)
        train_loader = feature_loader(x, train_meta, exp.batch_size, shuffle=True)
        val = {}
        for v in exp.val_views:  # independent of the train views, so runs stay comparable
            vx, vmeta = feature_set(cfg, f"dev_{v}", split="select")
            val[f"dev_{v}"] = (feature_loader(vx, vmeta, 4096), vmeta)
        model = CaesdDetector(None, Head(x.shape[1], x.shape[2], dropout=exp.dropout,
                                         n_channels=n_channels, band=band))
    else:
        train_meta = pd.read_parquet(Path(cfg.paths.manifests) / "train.parquet")
        dataset = AudioDataset(train_meta, exp.crop_seconds, rawboost=RawBoost() if exp.rawboost else None,
                               codec=CodecAugment(p=exp.codec_p), train=True)
        train_loader = DataLoader(dataset, batch_size=exp.batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate, num_workers=cfg.data.num_workers, pin_memory=True,
                                  persistent_workers=True, worker_init_fn=seed_worker)
        val = {name: audio_set(cfg, name, split="select", n=exp.val_n) for name in ("dev_clean", "dev_codec")}
        backbone = Backbone.from_pretrained(cfg.backbone.name, freeze=not exp.unfreeze, lora_rank=exp.lora_rank)
        model = CaesdDetector(backbone, Head(backbone.n_layers, backbone.feature_dim, dropout=exp.dropout,
                                             n_channels=n_channels, band=band))
    log.info("train clips: %d | label %s | val sets: %s", len(train_meta),
             train_meta["label"].value_counts().to_dict(), {k: len(v[1]) for k, v in val.items()})

    best = fit(model, train_loader, val, run_dir, epochs=exp.epochs, lr=exp.lr, backbone_lr=exp.backbone_lr,
               weight_decay=exp.weight_decay, patience=exp.patience, cond_weight=exp.cond_weight,
               device=cfg.device, select_window=exp.select_window)
    # band_mask included: without it the logged weights are not the ones the head used.
    best["layer_weights"] = torch.softmax(model.head.layer_logits + model.head.band_mask, 0).tolist()
    write_json(run_dir / "best.json", best)
    log.info("best epoch %d | select EER %.4f", best["epoch"], best["select"])

    for name in exp.score_sets:
        if exp.features:
            sx, meta = feature_set(cfg, name)
            loader = feature_loader(sx, meta, 4096)
        else:
            loader, meta = audio_set(cfg, name)
        scores = predict(model, loader, cfg.device)
        meta["sim_codec"] = np.asarray(CODEC_NAMES)[scores.pop("channel").to_numpy()]
        out = meta[_SCORE_COLUMNS].assign(**scores.drop(columns="idx"))
        out.to_parquet(run_dir / f"scores_{name}.parquet")
        log.info("scored %s: %d clips -> %s", name, len(out), run_dir / f"scores_{name}.parquet")


if __name__ == "__main__":
    main()
