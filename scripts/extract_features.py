"""Cache the pooled layer features of a frozen backbone for one partition and one view.

    uv run python scripts/extract_features.py +partition=train +view=clean
    uv run python scripts/extract_features.py +partition=dev +view=codec "+backbones=[xlsr,wavlm]"

Claim served: the frozen SSL backbone already separates unseen attacks (linear probe ~1% dev
EER), so the detector baseline must start from frozen features. Views: ``clean`` (the clip as
released), ``codec`` (one simulated ASVspoof 5 codec per clip), ``rb_codec`` (RawBoost, then the
codec), ``encodec`` (EnCodec, or MP3 then EnCodec: the neural codecs of eval C04 / C07). Clips are
read at full length up to ``data.max_seconds``.

Several backbones (``+backbones``, names of configs/backbone/*.yaml) share one pass over the
audio, so they see the identical decoded and coded waveform.

Writes <features>/<backbone tag>/<partition>_<view>.npy  [N, L, D] float16, and a .parquet with
the manifest rows plus ``sim_codec``. The parquet is written last: its presence means complete.
A stopped run resumes from the last flushed batch.

Also writes <features>/descriptor/<partition>_<view>.npy [N, DESCRIPTOR_DIM] float32: block C's
channel descriptor of the same final waveform. It is backbone-independent, so it lives in its own
directory beside the per-backbone ones and is computed once. With ``+descriptor_only=true`` the backbones are
never loaded and only the descriptor is written -- that is the CPU-only pass for feature sets that
were extracted before block C needed a leak-free condition input:

    uv run python scripts/extract_features.py +partition=dev +view=codec +descriptor_only=true
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from caesd.data.augment import CodecAugment, RawBoost
from caesd.data.condition_grid import DESCRIPTOR_DIM, NEURAL_CODECS
from caesd.data.datasets import CODEC_NAMES, AudioDataset, collate
from caesd.models import Backbone
from caesd.utils.logging import get_logger, log_hparams

log = get_logger("caesd.extract")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log_hparams(log, OmegaConf.to_container(cfg, resolve=True))
    partition, view = cfg.partition, cfg.get("view", "clean")
    manifest = pd.read_parquet(Path(cfg.paths.manifests) / f"{partition}.parquet")
    descriptor_only = cfg.get("descriptor_only", False)
    names = [] if descriptor_only else cfg.get("backbones", [HydraConfig.get().runtime.choices.backbone])
    confs = [OmegaConf.load(Path(__file__).parent.parent / "configs" / "backbone" / f"{n}.yaml") for n in names]
    outs = [Path(cfg.paths.features) / c.tag / f"{partition}_{view}" for c in confs]
    desc_out = Path(cfg.paths.features) / "descriptor" / f"{partition}_{view}"
    for out in [*outs, desc_out]:
        out.parent.mkdir(parents=True, exist_ok=True)
    for out in outs:
        if out.with_suffix(".parquet").exists():
            raise FileExistsError(f"{out}.parquet exists: this feature set is complete")
    # The descriptor is backbone-independent, so a second backbone's pass reuses the first one's.
    write_desc = not desc_out.with_suffix(".parquet").exists()
    if descriptor_only and not write_desc:
        raise FileExistsError(f"{desc_out}.parquet exists: this descriptor set is complete")
    if not write_desc:
        log.info("descriptor already complete, not recomputed: %s.parquet", desc_out)

    backbones = [Backbone.from_pretrained(c.name).eval().to(cfg.device) for c in confs]
    state_path = (outs[0] if outs else desc_out).with_suffix(".state.npz")
    resume = state_path.exists()
    feats = []
    for out, backbone in zip(outs, backbones, strict=True):
        shape = (len(manifest), backbone.n_layers, backbone.feature_dim)
        if resume:
            feats.append(np.lib.format.open_memmap(out.with_suffix(".npy"), mode="r+"))
            if feats[-1].shape != shape:
                raise RuntimeError(f"cannot resume: {out}.npy has shape {feats[-1].shape}, need {shape}")
        else:
            feats.append(np.lib.format.open_memmap(out.with_suffix(".npy"), mode="w+", dtype=np.float16, shape=shape))
        log.info("shape | %s = %s", out, shape)
    desc = None
    if write_desc:
        shape = (len(manifest), DESCRIPTOR_DIM)
        mode = "r+" if resume and desc_out.with_suffix(".npy").exists() else "w+"
        desc = np.lib.format.open_memmap(desc_out.with_suffix(".npy"), mode=mode,
                                         dtype=np.float32, shape=None if mode == "r+" else shape)
        if desc.shape != shape:
            raise RuntimeError(f"cannot resume: {desc_out}.npy has shape {desc.shape}, need {shape}")
        log.info("shape | %s = %s", desc_out, shape)
    if resume:
        state = np.load(state_path)
        done, channel = state["done"], state["channel"]
    else:
        done, channel = np.zeros(len(manifest), bool), np.zeros(len(manifest), np.int64)
    log.info("clips already done: %d / %d", int(done.sum()), len(done))

    dataset = AudioDataset(
        manifest, cfg.data.max_seconds,
        rawboost=RawBoost() if view == "rb_codec" else None,
        codec=(CodecAugment() if view in ("codec", "rb_codec")
               else CodecAugment(codecs=NEURAL_CODECS) if view == "encodec" else None),
        descriptor=write_desc,
    )
    # Read in tar order (sequential disk access); inside each window of 100 batches, sort by size
    # so a batch holds clips of similar length.
    bs = cfg.data.extract_batch_size
    order = np.lexsort((manifest["offset"].to_numpy(), manifest["path"].to_numpy()))
    order = order[~done[order]]
    size = manifest["size"].to_numpy()
    windows = [order[i : i + 100 * bs] for i in range(0, len(order), 100 * bs)]
    order = np.concatenate([w[np.argsort(-size[w], kind="stable")] for w in windows]) if windows else order
    batches = [order[i : i + bs].tolist() for i in range(0, len(order), bs)]
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate,
                        num_workers=cfg.data.num_workers, pin_memory=True)

    for step, batch in enumerate(loader):
        idx = batch["idx"].numpy()
        if backbones:  # descriptor_only never touches the GPU
            x, lengths = batch["x"].to(cfg.device), batch["lengths"].to(cfg.device)
        for feat, backbone in zip(feats, backbones, strict=True):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                feat[idx] = backbone(x, lengths).float().cpu().numpy().astype(np.float16)
        if desc is not None:
            desc[idx] = batch["descriptor"].numpy()
        channel[idx] = batch["channel"].numpy()
        done[idx] = True
        if step % 500 == 0:
            for feat in [*feats, *([desc] if desc is not None else [])]:
                feat.flush()
            np.savez(state_path, done=done, channel=channel)
            log.info("batch %d / %d | clips done %d / %d", step, len(batches), int(done.sum()), len(done))
    manifest["sim_codec"] = np.asarray(CODEC_NAMES)[channel]
    written = []
    for feat, out in [*zip(feats, outs, strict=True), *([(desc, desc_out)] if desc is not None else [])]:
        feat.flush()
        if not np.isfinite(feat[:: max(1, len(feat) // 2000)].astype(np.float32)).all():
            raise RuntimeError(f"non-finite features in {out}")
        manifest.to_parquet(out.with_suffix(".parquet"))
        written.append(str(out))
    state_path.unlink(missing_ok=True)
    log.info("wrote %s (%d clips) | sim_codec %s", written, len(manifest),
             manifest["sim_codec"].value_counts().to_dict())


if __name__ == "__main__":
    main()
