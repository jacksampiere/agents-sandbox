"""Gate 2 — rule-based pooling up the granularity hierarchy."""

import numpy as np
import pytest

from eeg_explorer.pooling import pool_embeddings
from eeg_explorer.synthetic import DEFAULT_CONFIG, generate_fixture


@pytest.fixture(scope="module")
def fixture():
    return generate_fixture(embedding_dim=32)


def test_record_mean_matches_groupby(fixture):
    pooled, index = pool_embeddings(fixture.embeddings, fixture.epoch_index, "record", "mean")
    assert pooled.shape == (DEFAULT_CONFIG.n_subjects, 32)  # 1 record/subject
    assert list(index.columns) == ["record_id", "subject_id"]
    # Row i must equal the mean of that record's epochs.
    for i, rid in enumerate(index["record_id"]):
        mask = (fixture.epoch_index["record_id"] == rid).to_numpy()
        np.testing.assert_allclose(pooled[i], fixture.embeddings[mask].mean(axis=0), rtol=1e-5)


def test_subject_pooling_shape_and_alignment(fixture):
    pooled, index = pool_embeddings(fixture.embeddings, fixture.epoch_index, "subject", "mean")
    assert pooled.shape == (DEFAULT_CONFIG.n_subjects, 32)
    assert list(index.columns) == ["subject_id"]
    first = index["subject_id"].iloc[0]
    mask = (fixture.epoch_index["subject_id"] == first).to_numpy()
    np.testing.assert_allclose(pooled[0], fixture.embeddings[mask].mean(axis=0), rtol=1e-5)


def test_max_pooling(fixture):
    pooled, index = pool_embeddings(fixture.embeddings, fixture.epoch_index, "record", "max")
    rid = index["record_id"].iloc[0]
    mask = (fixture.epoch_index["record_id"] == rid).to_numpy()
    np.testing.assert_allclose(pooled[0], fixture.embeddings[mask].max(axis=0), rtol=1e-5)


def test_epoch_grain_rejected(fixture):
    with pytest.raises(ValueError, match="base grain"):
        pool_embeddings(fixture.embeddings, fixture.epoch_index, "epoch", "mean")


def test_unknown_pooling_rejected(fixture):
    with pytest.raises(ValueError, match="unknown pooling"):
        pool_embeddings(fixture.embeddings, fixture.epoch_index, "record", "median")
