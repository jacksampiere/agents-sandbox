"""Gate 1 — the synthetic fixture is deterministic and carries recoverable injected structure.

PCA is run here with plain sklearn (independent of our own projections module) so the test validates
the *fixture*, not the code that will later consume it.
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from eeg_explorer.labels import LABEL_COLUMNS, broadcast_labels, native_granularity
from eeg_explorer.synthetic import DEFAULT_CONFIG, generate_fixture

N_EPOCH = (
    DEFAULT_CONFIG.n_subjects
    * DEFAULT_CONFIG.records_per_subject
    * DEFAULT_CONFIG.epochs_per_record
)


def _pca_silhouette(embeddings, labels, n_components=2):
    """z-score → PCA → silhouette, mirroring the honesty-guardrail transform (z-score before PCA)."""
    z = StandardScaler().fit_transform(embeddings)
    pcs = PCA(n_components=n_components, random_state=0).fit_transform(z)
    return silhouette_score(pcs, labels)


@pytest.fixture(scope="module")
def fixture():
    return generate_fixture(embedding_dim=200)


def test_shapes(fixture):
    assert fixture.embeddings.shape == (N_EPOCH, 200)
    assert fixture.embeddings.dtype == np.float32
    assert len(fixture.epoch_index) == N_EPOCH
    assert len(fixture.record_index) == DEFAULT_CONFIG.n_subjects  # 1 record/subject in the default
    assert len(fixture.subject_index) == DEFAULT_CONFIG.n_subjects
    assert list(fixture.epoch_index.columns) == ["epoch_id", "record_id", "subject_id"]


def test_labels_schema_and_native_grains(fixture):
    assert list(fixture.labels.columns) == LABEL_COLUMNS
    assert native_granularity(fixture.labels, "sleep_stage") == "epoch"
    assert native_granularity(fixture.labels, "arousal") == "epoch"
    assert native_granularity(fixture.labels, "ahi") == "record"
    assert native_granularity(fixture.labels, "disease") == "subject"
    assert native_granularity(fixture.labels, "sex") == "subject"
    assert native_granularity(fixture.labels, "age") == "subject"
    # value column is generic string long-format (holds arbitrary outcomes later)
    assert fixture.labels["value"].map(type).eq(str).all()


def test_determinism_same_seed(fixture):
    again = generate_fixture(embedding_dim=200)
    assert np.array_equal(fixture.embeddings, again.embeddings)
    assert fixture.labels.equals(again.labels)


def test_per_encoder_seed_changes_embeddings_not_labels(fixture):
    other = generate_fixture(embedding_dim=200, seed=DEFAULT_CONFIG.seed + 7)
    assert not np.array_equal(fixture.embeddings, other.embeddings)
    assert fixture.labels.equals(other.labels)  # metadata is dim/seed-independent


def test_metadata_identical_across_dims(fixture):
    big = generate_fixture(embedding_dim=512)
    assert big.embeddings.shape == (N_EPOCH, 512)
    assert fixture.labels.equals(big.labels)
    assert fixture.epoch_index.equals(big.epoch_index)


def test_pca_recovers_injected_structure(fixture):
    """Structured epoch label (sleep_stage) separates under PCA; the negative control (sex) does not."""
    stage = broadcast_labels(fixture.labels, fixture.epoch_index, "sleep_stage").to_numpy()
    sex = broadcast_labels(fixture.labels, fixture.epoch_index, "sex").to_numpy()

    stage_sil = _pca_silhouette(fixture.embeddings, stage)
    sex_sil = _pca_silhouette(fixture.embeddings, sex)

    assert stage_sil > 0.25, (
        f"injected sleep_stage structure not recovered (silhouette={stage_sil:.3f})"
    )
    assert sex_sil < 0.10, f"negative-control sex should not separate (silhouette={sex_sil:.3f})"
    assert stage_sil - sex_sil > 0.20
