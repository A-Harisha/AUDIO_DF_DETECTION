"""Index ASVspoof 5 (train, dev, eval) into manifests with random access into the flac tars.

    uv run python scripts/prepare_asvspoof5.py

Claim served: every claim. This fixes the protocol: train on train (attacks A01-A08), select
models and fit calibration on dev (A09-A16, speaker-disjoint halves ``select`` / ``calib``), and
report on eval only (A17-A32, codecs C01-C11). No clip is extracted to disk.
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from caesd.data.manifests import assert_no_leakage, assign_split_by
from caesd.data.sources import index_asvspoof5
from caesd.utils.logging import get_logger

log = get_logger("caesd.prepare.a5")

# partition: (protocol, tar glob, split fractions for assign_split_by, or the single split name)
_PARTITION = {
    "train": ("ASVspoof5.train.tsv", "flac_T_*.tar", "train"),
    "dev": ("ASVspoof5.dev.track_1.tsv", "flac_D_*.tar", None),
    "eval": ("ASVspoof5.eval.track_1.tsv", "flac_E_*.tar", "test"),
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.paths.manifests)
    out_dir.mkdir(parents=True, exist_ok=True)
    for partition in cfg.get("partitions", list(_PARTITION)):
        protocol, pattern, split = _PARTITION[partition]
        tar_dir = Path(cfg.paths.asvspoof5_eval if partition == "eval" else cfg.paths.asvspoof5)
        tars = sorted(tar_dir.glob(pattern))
        df = index_asvspoof5(Path(cfg.paths.asvspoof5) / "ASVspoof5_protocols.tar", tars, protocol)
        coverage = len(df) / int(df["n_protocol"].iloc[0])
        log.info("%s: %d tars, %d clips, %.1f%% of the protocol", partition, len(tars), len(df), 100 * coverage)
        if coverage < cfg.get("min_coverage", 1.0):
            raise RuntimeError(f"{partition}: only {coverage:.1%} of the protocol is on disk "
                               f"(pass +min_coverage=<fraction> to index a partial download)")
        if split is None:  # dev: model selection and calibration never share a speaker
            df = assign_split_by(df, key="speaker", fractions=(0.5, 0.5, 0.0), seed=cfg.seed)
            df["split"] = df["split"].map({"train": "select", "dev": "calib"})
            assert_no_leakage(df, keys=("speaker",))
        else:
            df["split"] = split
        df = df.drop(columns="n_protocol")
        df.to_parquet(out_dir / f"{partition}.parquet")
        log.info("%s -> %s | splits %s | label %s | codecs %s", partition, out_dir / f"{partition}.parquet",
                 df["split"].value_counts().to_dict(), df["label"].value_counts().to_dict(),
                 df["codec"].value_counts().to_dict())


if __name__ == "__main__":
    main()
