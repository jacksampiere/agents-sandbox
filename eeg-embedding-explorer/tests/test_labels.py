"""Gate 1 — the long-format label store and the broadcast-down rule."""

import pytest

from eeg_explorer.labels import (
    available_labels_for_granularity,
    broadcast_labels,
    native_granularity,
)
from eeg_explorer.synthetic import generate_fixture


@pytest.fixture(scope="module")
def fixture():
    return generate_fixture(embedding_dim=64)


def test_available_labels_respect_broadcast_direction(fixture):
    labels = fixture.labels
    # Epoch (finest) can show every label — all broadcast down to it.
    assert available_labels_for_granularity(labels, "epoch") == sorted(
        ["sleep_stage", "arousal", "ahi", "disease", "sex", "age"]
    )
    # Record can show record- and subject-native labels, never epoch-native ones.
    assert available_labels_for_granularity(labels, "record") == sorted(
        ["ahi", "disease", "sex", "age"]
    )
    # Subject (coarsest) can show only subject-native labels.
    assert available_labels_for_granularity(labels, "subject") == sorted(["disease", "sex", "age"])


def test_broadcast_subject_label_to_epochs(fixture):
    values = broadcast_labels(fixture.labels, fixture.epoch_index, "disease")
    assert len(values) == len(fixture.epoch_index)
    assert values.notna().all()

    # Every epoch inherits exactly its subject's disease value.
    subj_disease = (
        fixture.labels.query("attribute_name == 'disease'").set_index("id")["value"].to_dict()
    )
    expected = fixture.epoch_index["subject_id"].map(subj_disease).to_numpy()
    assert (values.to_numpy() == expected).all()


def test_broadcast_epoch_label_at_native_grain(fixture):
    values = broadcast_labels(fixture.labels, fixture.epoch_index, "sleep_stage")
    assert set(values.unique()) <= {"W", "N1", "N2", "N3", "REM"}
    assert len(values) == len(fixture.epoch_index)


def test_cannot_broadcast_up_to_coarser_grain(fixture):
    # sleep_stage is epoch-native; the subject index lacks epoch_id → cannot apply without aggregation.
    with pytest.raises(ValueError, match="coarser grain"):
        broadcast_labels(fixture.labels, fixture.subject_index, "sleep_stage")


def test_native_granularity_single_valued(fixture):
    assert native_granularity(fixture.labels, "ahi") == "record"
