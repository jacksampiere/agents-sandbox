"""Gate 2 — the on-disk cache round-trips embeddings + index and keys pooling correctly."""

import numpy as np
import pandas as pd
import pytest

from eeg_explorer import cache


def _toy_index(n):
    return pd.DataFrame(
        {
            "epoch_id": [f"E{i:03d}" for i in range(n)],
            "record_id": ["R00"] * n,
            "subject_id": ["S00"] * n,
        }
    )


def test_epoch_roundtrip(tmp_path):
    emb = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    index = _toy_index(10)
    cache.write_cache("enc", "epoch", emb, index, root=tmp_path)
    back_emb, back_index = cache.read_cache("enc", "epoch", root=tmp_path)
    np.testing.assert_array_equal(emb, back_emb)
    pd.testing.assert_frame_equal(index, back_index)


def test_record_pooling_in_path(tmp_path):
    emb = np.zeros((3, 4), dtype=np.float32)
    index = pd.DataFrame({"record_id": ["R00", "R01", "R02"], "subject_id": ["S00", "S00", "S01"]})
    cache.write_cache("enc", "record", emb, index, pooling="mean", root=tmp_path)
    expected = tmp_path / "enc" / "record" / "mean" / "embeddings.npy"
    assert expected.exists()
    back_emb, _ = cache.read_cache("enc", "record", pooling="mean", root=tmp_path)
    assert back_emb.shape == (3, 4)


def test_pooling_required_for_coarse_grain(tmp_path):
    with pytest.raises(ValueError, match="cache key"):
        cache.cache_dir("enc", "record", pooling=None, root=tmp_path)


def test_manifest_and_listing(tmp_path):
    cache.write_cache("enc", "epoch", np.zeros((2, 2), np.float32), _toy_index(2), root=tmp_path)
    cache.write_manifest("enc", {"encoder": "enc", "embedding_dim": 2}, root=tmp_path)
    assert cache.list_encoders(tmp_path) == ["enc"]  # only dirs with a manifest count
    assert cache.read_manifest("enc", tmp_path)["embedding_dim"] == 2
    assert "epoch" in cache.available_granularities("enc", tmp_path)


def test_available_poolings(tmp_path):
    idx = pd.DataFrame({"subject_id": ["S00"]})
    for pooling in ("mean", "max"):
        cache.write_cache(
            "enc", "subject", np.zeros((1, 2), np.float32), idx, pooling=pooling, root=tmp_path
        )
    assert cache.available_poolings("enc", "subject", tmp_path) == ["max", "mean"]
    assert cache.available_poolings("enc", "epoch", tmp_path) == []


def test_env_var_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("EEG_CACHE_ROOT", str(tmp_path / "c"))
    monkeypatch.setenv("EEG_LABELS_PATH", str(tmp_path / "labels.parquet"))
    assert cache.cache_root() == tmp_path / "c"
    assert cache.labels_path() == tmp_path / "labels.parquet"
