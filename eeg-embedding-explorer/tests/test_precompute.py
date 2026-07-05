"""Gate 2 — precompute writes the full cache layout, via both the synthetic and adapter paths."""

import numpy as np
import pandas as pd
import pytest

from eeg_explorer import cache
from eeg_explorer.adapters import dummy_adapter
from eeg_explorer.precompute import build_synthetic_caches, precompute_from_adapter
from eeg_explorer.registry import load_registry
from eeg_explorer.synthetic import SyntheticConfig

SMALL = SyntheticConfig(n_subjects=4, epochs_per_record=20)


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_build_synthetic_caches_full_layout(tmp_path, registry):
    labels_out = tmp_path / "labels.parquet"
    build_synthetic_caches(root=tmp_path, labels_out=labels_out, config=SMALL)

    assert sorted(cache.list_encoders(tmp_path)) == sorted(registry)
    assert labels_out.exists()

    for name, facts in registry.items():
        # Epoch grain: dim matches the registry, N = subjects * epochs_per_record.
        emb, index = cache.read_cache(name, "epoch", root=tmp_path)
        assert emb.shape == (SMALL.n_subjects * SMALL.epochs_per_record, facts.embedding_dim)
        assert list(index.columns) == ["epoch_id", "record_id", "subject_id"]
        # Coarser grains for each pooling.
        for pooling in ("mean", "max"):
            rec, _ = cache.read_cache(name, "record", pooling=pooling, root=tmp_path)
            sub, _ = cache.read_cache(name, "subject", pooling=pooling, root=tmp_path)
            assert rec.shape == (SMALL.n_subjects, facts.embedding_dim)  # 1 record/subject
            assert sub.shape == (SMALL.n_subjects, facts.embedding_dim)
        manifest = cache.read_manifest(name, tmp_path)
        assert manifest["source_dataset"] == "synthetic"
        assert manifest["pretraining_datasets"]  # non-empty (feeds the overlap flag)
        assert manifest["poolings"] == ["mean", "max"]


def test_precompute_from_dummy_adapter(tmp_path, registry):
    adapter = dummy_adapter("cbramod", registry=registry, seed=1)
    n_epochs, n_channels, n_times = 6, 2, 6000  # 30s @ 200Hz, 2 SHHS channels
    raw = (
        np.random.default_rng(0).standard_normal((n_epochs, n_channels, n_times)).astype(np.float32)
    )
    index = pd.DataFrame(
        {
            "epoch_id": [f"R00-E{i:03d}" for i in range(n_epochs)],
            "record_id": ["R00"] * n_epochs,
            "subject_id": ["S00"] * n_epochs,
        }
    )
    emb = precompute_from_adapter(adapter, raw, ["C3", "C4"], index, root=tmp_path)
    assert emb.shape == (n_epochs, registry["cbramod"].embedding_dim)
    assert np.isfinite(emb).all()

    back, _ = cache.read_cache("cbramod", "epoch", root=tmp_path)
    np.testing.assert_array_equal(emb, back)
    # Pooled caches exist for the adapter path too.
    rec, _ = cache.read_cache("cbramod", "record", pooling="mean", root=tmp_path)
    assert rec.shape == (1, registry["cbramod"].embedding_dim)


def test_dummy_adapter_is_deterministic_and_correct_dim(registry):
    adapter = dummy_adapter("bendr", registry=registry, seed=7)
    raw = np.zeros((3, 90, 7680), dtype=np.float32)  # BENDR-shaped no-op input
    a = adapter.embed_epochs(raw, ["C3", "C4"])
    b = adapter.embed_epochs(raw, ["C3", "C4"])
    assert a.shape == (3, 512)
    np.testing.assert_array_equal(a, b)
