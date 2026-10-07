"""Tar-indexed audio access, signal ops, manifest splits and leakage control."""
import io
import tarfile

import numpy as np
import pytest
import soundfile as sf

from caesd.data.condition_grid import CODECS, EVAL_CODEC, add_noise, apply_reverb
from caesd.data.datasets import CODEC_NAMES, AudioDataset, collate
from caesd.data.manifests import assert_no_leakage, assign_split_by, build_manifest
from caesd.data.sources import index_asvspoof5


def _fake_asvspoof5(tmp_path):
    """Two clips in a flac tar and a matching protocol tar, in the ASVspoof 5 layout."""
    rng = np.random.default_rng(0)
    waves = {"T_0000000001": rng.normal(0, 0.1, 16000).astype(np.float32),
             "T_0000000002": rng.normal(0, 0.1, 24000).astype(np.float32)}
    with tarfile.open(tmp_path / "flac_T_aa.tar", "w") as tf:
        for utt, wave in waves.items():
            buf = io.BytesIO()
            sf.write(buf, wave, 16000, format="FLAC", subtype="PCM_16")
            info = tarfile.TarInfo(f"flac_T/{utt}.flac")
            info.size = len(buf.getvalue())
            buf.seek(0)
            tf.addfile(info, buf)
    proto = (b"T_01 T_0000000001 F - - - AC3 A05 spoof -\n"
             b"T_02 T_0000000002 M - - - - bonafide bonafide -\n"
             b"T_03 T_0000000003 M - - - - bonafide bonafide -\n")
    with tarfile.open(tmp_path / "protocols.tar", "w") as tf:
        info = tarfile.TarInfo("train.tsv")
        info.size = len(proto)
        tf.addfile(info, io.BytesIO(proto))
    return waves


def test_index_reads_clips_straight_from_the_tar(tmp_path):
    waves = _fake_asvspoof5(tmp_path)
    df = index_asvspoof5(tmp_path / "protocols.tar", [tmp_path / "flac_T_aa.tar"], "train.tsv")
    assert list(df["utt"]) == list(waves) and list(df["label"]) == [0, 1]
    assert df["n_protocol"].iloc[0] == 3  # the third clip is not on disk: coverage is 2/3
    assert set(df["codec"]) == {"clean"}
    batch = collate([AudioDataset(df, max_seconds=4.0)[i] for i in range(2)])
    assert batch["x"].shape == (2, 24000) and batch["lengths"].tolist() == [16000, 24000]
    assert np.allclose(batch["x"][0, :16000].numpy(), waves["T_0000000001"], atol=1e-4)


def test_codec_tables_are_consistent():
    assert CODEC_NAMES[0] == "clean" and set(EVAL_CODEC.values()) == set(CODECS)
    assert all(len(spec[3]) == 5 for spec in CODECS.values())  # CODEC_Q 1..5


def test_add_noise_hits_target_snr():
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 1, 16000)
    noise = rng.normal(0, 1, 16000)
    out = add_noise(wave, noise, snr_db=10.0)
    added = out - wave
    measured = 10 * np.log10(np.mean(wave**2) / np.mean(added**2))
    assert measured == pytest.approx(10.0, abs=0.5)


def test_reverb_preserves_length():
    wave = np.random.randn(8000).astype(np.float32)
    ir = np.random.randn(512).astype(np.float32)
    assert apply_reverb(wave, ir).shape == wave.shape


def test_split_is_disjoint_by_key():
    records = [{"path": f"p{i}", "label": i % 2, "speaker": f"spk{i % 20}"} for i in range(400)]
    df = assign_split_by(build_manifest(records), key="speaker", seed=0)
    assert_no_leakage(df, keys=("speaker",))  # must not raise


def test_leakage_is_detected():
    df = build_manifest([{"path": "a", "speaker": "s1"}, {"path": "b", "speaker": "s1"}])
    df.loc[0, "split"], df.loc[1, "split"] = "train", "test"
    with pytest.raises(AssertionError):
        assert_no_leakage(df, keys=("speaker",))
