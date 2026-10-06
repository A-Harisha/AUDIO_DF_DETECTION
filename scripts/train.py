"""Train the CAESD detector.

    uv run python scripts/train.py experiment=phase2 data.manifest=...

Builds the backbone + all four blocks, then runs the observable training loop.
"""
from __future__ import annotations

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from caesd.data.augment import RawBoost
from caesd.data.datasets import AudioDataset
from caesd.models import Backbone, CaesdDetector
from caesd.train.loop import fit
from caesd.utils.logging import get_logger, log_hparams
from caesd.utils.seed import set_seed

log = get_logger("caesd.train.cli")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log_hparams(log, OmegaConf.to_container(cfg, resolve=True))
    log.info("seed=%d", set_seed(cfg.seed))

    manifest = pd.read_parquet(cfg.data.manifest)
    train_df = manifest[manifest["split"] == "train"]
    augment = RawBoost() if cfg.experiment.get("train", {}).get("augment") == "rawboost" else None
    dataset = AudioDataset(train_df, max_seconds=cfg.data.max_seconds, augment=augment, train=True)
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=4)

    backbone = Backbone.from_pretrained(cfg.backbone.name, cfg.backbone.freeze, cfg.backbone.lora_rank)
    model = CaesdDetector(backbone)
    tcfg = cfg.experiment.train
    fit(model, loader, epochs=tcfg.epochs, lr=tcfg.lr, device=cfg.device,
        loss_weights=OmegaConf.to_container(cfg.experiment.loss_weights, resolve=True))


if __name__ == "__main__":
    main()
