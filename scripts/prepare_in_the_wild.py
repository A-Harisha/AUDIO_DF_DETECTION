"""Index the In-the-Wild deepfake set as a manifest our loaders can read.

    uv run python scripts/prepare_in_the_wild.py paths.manifests=experiments/data_full

Claim served: generalisation. Every number this project reports is in-domain ASVspoof 5, where train,
dev and eval share a recording pipeline and the spoofing systems are known families A01-A32. A
headline of 3.21% there says nothing about audio found in the world. In-the-Wild is 31,779 clips of
real and synthesised speech of public figures, collected from the internet: different speakers,
different channels, no codec labels, and spoofing systems we have no list of.

It is a REPORTING set only, like eval. Nothing is selected or calibrated on it. The expectation is
that the numbers are much worse than 3.21% -- published systems typically lose an order of magnitude
moving to this set -- and the point is to measure that rather than leave it unstated.

``offset`` is -1 for every row: these are loose .wav files, not byte ranges inside a tar, and
``AudioDataset`` takes a negative offset to mean exactly that.

Writes <manifests>/itw.parquet, so the score set is named ``itw_clean``.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from caesd.utils.logging import get_logger

log = get_logger("caesd.prepare_itw")
DEFAULT_ROOT = "F:/datasets/audio/in_the_wild/release_in_the_wild"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.get("itw_root", DEFAULT_ROOT))
    meta = pd.read_csv(root / "meta.csv")
    if not {"file", "speaker", "label"} <= set(meta.columns):
        raise ValueError(f"{root / 'meta.csv'}: expected file, speaker, label; got {list(meta.columns)}")

    paths = [root / f for f in meta["file"]]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} clips in meta.csv are not on disk, e.g. {missing[:3]}")

    labels = meta["label"].str.strip().str.lower()
    unknown = set(labels) - {"bona-fide", "bonafide", "spoof"}
    if unknown:
        raise ValueError(f"unexpected labels {unknown}: the mapping to our 1=bona fide would be a guess")

    out = pd.DataFrame({
        "path": [str(p) for p in paths],
        "offset": -1,                 # a loose file, not a tar member
        "size": [p.stat().st_size for p in paths],
        "utt": [Path(f).stem for f in meta["file"]],
        "label": (labels != "spoof").astype(int),   # 1 = bona fide, as everywhere else
        "speaker": meta["speaker"].astype(str),
        "gender": "unknown",          # not distributed with the set
        "codec": "unknown",           # found audio: the channel is whatever the internet did to it
        "codec_q": 0,
        "generator": labels.map(lambda v: "itw_spoof" if v == "spoof" else "-"),
        "split": "eval",              # reporting only, never selected or calibrated on
    })
    if out["utt"].duplicated().any():
        raise ValueError("duplicate utt ids: the score alignment joins on utt")

    path = Path(cfg.paths.manifests) / "itw.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    log.info("wrote %s: %d clips | label %s | %d speakers | %.1f GB of audio", path, len(out),
             out["label"].value_counts().to_dict(), out["speaker"].nunique(),
             out["size"].sum() / 2**30)
    log.info("score it with experiment.score_sets=[itw_clean]")


if __name__ == "__main__":
    main()
