"""Gate 2 — projections: z-score→PCA (deterministic guardrail) and seeded UMAP, fit per grain."""

import numpy as np
import pytest
from sklearn.metrics import silhouette_score

from eeg_explorer.labels import broadcast_labels
from eeg_explorer.pooling import pool_embeddings
from eeg_explorer.projections import project_pca, project_umap, zscore
from eeg_explorer.synthetic import generate_fixture


@pytest.fixture(scope="module")
def fixture():
    return generate_fixture(embedding_dim=128)


def test_zscore_standardizes(fixture):
    z = zscore(fixture.embeddings)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-6)


def test_pca_shape_and_determinism(fixture):
    a = project_pca(fixture.embeddings, n_components=2)
    b = project_pca(fixture.embeddings, n_components=2)
    assert a.shape == (fixture.embeddings.shape[0], 2)
    np.testing.assert_array_equal(a, b)  # PCA is deterministic
    assert project_pca(fixture.embeddings, n_components=3).shape[1] == 3


def test_umap_shape_and_reproducibility(fixture):
    a = project_umap(fixture.embeddings, n_components=2, n_neighbors=15, min_dist=0.1)
    b = project_umap(fixture.embeddings, n_components=2, n_neighbors=15, min_dist=0.1)
    assert a.shape == (fixture.embeddings.shape[0], 2)
    np.testing.assert_allclose(a, b)  # fixed seed → reproducible


def test_umap_small_pointset_clamps_neighbors(fixture):
    # Subject grain has few points (< default n_neighbors); must clamp, not crash.
    pooled, _ = pool_embeddings(fixture.embeddings, fixture.epoch_index, "subject", "mean")
    coords = project_umap(pooled, n_components=2, n_neighbors=15, min_dist=0.1)
    assert coords.shape == (pooled.shape[0], 2)


def test_pca_recovers_subject_disease_structure(fixture):
    """Injected subject-level disease shift separates after pooling — under deterministic PCA."""
    pooled, index = pool_embeddings(fixture.embeddings, fixture.epoch_index, "subject", "mean")
    disease = broadcast_labels(fixture.labels, index, "disease").to_numpy()
    sex = broadcast_labels(fixture.labels, index, "sex").to_numpy()
    coords = project_pca(pooled, n_components=2)
    assert silhouette_score(coords, disease) > 0.5  # structured
    assert silhouette_score(coords, sex) < 0.2  # negative control
