"""Condition-grid labels, signal ops, manifest splits and leakage control."""
import numpy as np
import pytest

from caesd.data.condition_grid import Condition, add_noise, apply_reverb
from caesd.data.manifests import assert_no_leakage, assign_split_by, build_manifest


def test_condition_id_is_stable():
    assert Condition(codec="mp3", bitrate_kbps=16, snr_db=10.0).cond_id == "mp3_16k_snr10"
    assert Condition().cond_id == "clean"


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
