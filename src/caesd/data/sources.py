"""Dataset-specific parsers that turn a downloaded corpus into manifest records.

Each parser returns a list of dicts with the manifest columns it can fill. The split and the
leakage check are applied afterwards by scripts/make_manifests.py, once, correctly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# In-the-Wild label strings -> (binary label, 3-class name). Bona fide is the positive class.
_ITW_LABEL = {"bona-fide": (1, "bona_fide"), "spoof": (0, "synthetic")}

_A5_KEY = {"bonafide": 1, "spoof": 0}

# IndicSynth language folder -> ISO 639 code (explicit, so Marathi/Malayalam do not collide).
_INDIC_LANG = {
    "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn", "Malayalam": "ml",
    "Marathi": "mr", "Odia": "or", "Punjabi": "pa", "Sanskrit": "sa", "Tamil": "ta",
    "Telugu": "te", "Urdu": "ur",
}


def parse_in_the_wild(root: str | Path) -> list[dict]:
    """In-the-Wild (Mueller et al., 2022): a folder of .wav files + meta.csv(file,speaker,label)."""
    root = Path(root)
    meta_path = next(root.rglob("meta.csv"))
    meta = pd.read_csv(meta_path)
    audio_dir = meta_path.parent
    records = []
    for _, row in meta.iterrows():
        label, class_name = _ITW_LABEL[str(row["label"]).strip()]
        records.append({
            "path": str(audio_dir / row["file"]),
            "label": label,
            "class_name": class_name,
            "speaker": str(row["speaker"]).strip(),
            "language": "en",
            # In-the-Wild gives no generator labels (deepfakes from varied unknown sources).
            # Leave it None so generator-disjointness is enforced only where it is meaningful;
            # speaker-disjointness is what we control for this corpus.
            "generator": None,
            "dataset": "in_the_wild",
            "condition": "clean",
        })
    return records


def index_asvspoof5(protocols_tar: str | Path, flac_tars: list[str | Path], protocol: str) -> pd.DataFrame:
    """ASVspoof 5 track-1 manifest with random access into the (uncompressed) flac tars.

    Protocol columns (README.txt): SPEAKER_ID FLAC_FILE_NAME GENDER CODEC CODEC_Q CODEC_SEED
    ATTACK_TAG ATTACK_LABEL KEY TMP. CODEC is ``-`` for every train and dev clip; only the
    evaluation set is coded (C01-C11). ATTACK_TAG (AC1-AC3) is not a codec.

    Returns one row per protocol clip found in ``flac_tars``: path (the tar), offset, size, utt,
    label, speaker, gender, codec (``clean`` or C01-C11), codec_q, generator. Clips whose tar is
    not in ``flac_tars`` are absent; the caller checks coverage.
    """
    import tarfile

    with tarfile.open(protocols_tar) as t:
        proto = pd.read_csv(
            t.extractfile(protocol), sep=" ", header=None, dtype=str,
            names=["speaker", "utt", "gender", "codec", "codec_q", "codec_seed", "attack_tag",
                   "generator", "key", "tmp"],
        )
    members = []
    for tar in flac_tars:
        with tarfile.open(tar) as tf:
            members += [(Path(m.name).stem, str(tar), m.offset_data, m.size) for m in tf if m.isfile()]
    index = pd.DataFrame(members, columns=["utt", "path", "offset", "size"])
    df = proto.merge(index, on="utt", how="inner")
    df["label"] = df["key"].map(_A5_KEY).astype(int)
    df["codec"] = df["codec"].replace("-", "clean")
    df["codec_q"] = df["codec_q"].replace("-", "0").astype(int)
    df["n_protocol"] = len(proto)
    return df[["path", "offset", "size", "utt", "label", "speaker", "gender", "codec", "codec_q",
               "generator", "n_protocol"]]


def parse_indicsynth(parquet_paths: list[str | Path], language: str, out_dir: str | Path,
                     max_clips: int | None = None) -> list[dict]:
    """IndicSynth (ACL 2025): language-sharded parquet. All clips are synthetic (spoof).

    Extracts the embedded wav bytes to ``out_dir/<language>/`` and returns manifest records.
    ``max_clips`` caps extraction per call, for the lean subset.
    """
    import pyarrow.parquet as pq

    out_dir = Path(out_dir) / language
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for parquet_path in parquet_paths:
        for batch in pq.ParquetFile(str(parquet_path)).iter_batches(batch_size=256):
            for i, row in enumerate(batch.to_pylist()):
                if max_clips is not None and len(records) >= max_clips:
                    return records
                wav_path = out_dir / f"{len(records):06d}.wav"
                wav_path.write_bytes(row["audio"]["bytes"])
                records.append({
                    "path": str(wav_path),
                    "label": 0,
                    "class_name": "synthetic",
                    "speaker": str(row["Target Speaker ID"]),
                    "language": _INDIC_LANG[language],
                    "generator": row["Generative Model"],
                    "dataset": "indicsynth",
                    "condition": "clean",
                })
    return records
