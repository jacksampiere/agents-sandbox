"""Synthetic embedding fixture — the verification substrate (DESIGN §6, BUILD_SPEC Gate 1).

Generates embeddings *directly* at the epoch grain (bypassing any encoder) with controllable
injected cluster structure tied to label sets, plus Gaussian noise, deterministically from a fixed
seed. This lets tests assert that PCA recovers the injected structure (and, as a negative control,
finds none where none was injected). Commit the generator + seed, not large ``.npy`` files.

Structure injected:
- ``sleep_stage`` (epoch-level, 5 classes): one centroid per stage → strong epoch-grain clusters.
- ``disease`` (subject-level, binary): a per-subject centroid shift → separates at *subject* grain
  once epochs are pooled (the stage component averages to a shared constant across subjects).
- ``sex`` (subject-level, binary): assigned independently of the embedding → a **negative control**
  that PCA should NOT separate. Guards against the tool manufacturing structure.
- ``age`` (subject), ``ahi`` (record), ``arousal`` (epoch): plain metadata across all three grains,
  exercising label variety and the broadcast rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eeg_explorer.labels import LABEL_COLUMNS

GRANULARITIES = ("subject", "record", "epoch")

SLEEP_STAGES = ("W", "N1", "N2", "N3", "REM")
SLEEP_STAGE_PROBS = (0.10, 0.10, 0.40, 0.20, 0.20)  # same for every subject → cancels on pooling
SEX_CATEGORIES = ("F", "M")
DISEASE_CATEGORIES = ("control", "case")


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 20260705
    n_subjects: int = 12
    records_per_subject: int = 1
    epochs_per_record: int = 200
    class_sep: float = 6.0  # sleep-stage centroid scale (epoch-grain structure)
    subject_sep: float = 4.0  # disease centroid scale (subject-grain structure)
    noise_std: float = 1.0


DEFAULT_CONFIG = SyntheticConfig()


@dataclass
class Metadata:
    """Dimension-independent structure: ids, label table, and per-epoch class assignments."""

    epoch_index: pd.DataFrame  # [epoch_id, record_id, subject_id], one row per epoch
    record_index: pd.DataFrame  # [record_id, subject_id]
    subject_index: pd.DataFrame  # [subject_id]
    labels: pd.DataFrame  # long-format [id, granularity, attribute_name, value]
    stage_idx: np.ndarray  # (N_epoch,) sleep-stage class index per epoch
    disease_idx: np.ndarray  # (N_epoch,) disease class index per epoch (broadcast from subject)


@dataclass
class SyntheticFixture:
    embeddings: np.ndarray  # (N_epoch, D) float32, epoch grain
    epoch_index: pd.DataFrame
    record_index: pd.DataFrame
    subject_index: pd.DataFrame
    labels: pd.DataFrame
    embedding_dim: int


def _label_rows(ids, granularity, attribute_name, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(ids),
            "granularity": granularity,
            "attribute_name": attribute_name,
            "value": [str(v) for v in values],
        }
    )


def generate_metadata(config: SyntheticConfig = DEFAULT_CONFIG) -> Metadata:
    """Build ids, labels, and per-epoch class assignments. Deterministic from ``config.seed`` and
    independent of embedding dimensionality, so every encoder shares identical labels."""
    rng = np.random.default_rng(config.seed)

    n_subjects = config.n_subjects
    subject_ids = np.array([f"S{i:02d}" for i in range(n_subjects)])

    # Subject-level labels.
    disease = rng.integers(0, len(DISEASE_CATEGORIES), size=n_subjects)  # structured
    sex = rng.integers(0, len(SEX_CATEGORIES), size=n_subjects)  # negative control
    age = rng.integers(30, 80, size=n_subjects)  # metadata scalar

    # Records: subject index for each record.
    rec_subject_pos = np.repeat(np.arange(n_subjects), config.records_per_subject)
    n_records = rec_subject_pos.size
    record_ids = np.array([f"R{i:02d}" for i in range(n_records)])
    record_subject_ids = subject_ids[rec_subject_pos]
    ahi = rng.uniform(0.0, 40.0, size=n_records).round(1)  # record-level metadata

    # Epochs: record (and thus subject) index for each epoch.
    epoch_record_pos = np.repeat(np.arange(n_records), config.epochs_per_record)
    n_epochs = epoch_record_pos.size
    epoch_subject_pos = rec_subject_pos[epoch_record_pos]
    within_record = np.tile(np.arange(config.epochs_per_record), n_records)
    epoch_ids = np.array(
        [f"{record_ids[r]}-E{e:03d}" for r, e in zip(epoch_record_pos, within_record)]
    )
    epoch_record_ids = record_ids[epoch_record_pos]
    epoch_subject_ids = subject_ids[epoch_subject_pos]

    stage_idx = rng.choice(len(SLEEP_STAGES), size=n_epochs, p=SLEEP_STAGE_PROBS)
    arousal = rng.integers(0, 2, size=n_epochs)  # epoch-level metadata
    disease_idx = disease[epoch_subject_pos]  # broadcast to epoch for embedding generation

    epoch_index = pd.DataFrame(
        {"epoch_id": epoch_ids, "record_id": epoch_record_ids, "subject_id": epoch_subject_ids}
    )
    record_index = pd.DataFrame({"record_id": record_ids, "subject_id": record_subject_ids})
    subject_index = pd.DataFrame({"subject_id": subject_ids})

    labels = pd.concat(
        [
            _label_rows(epoch_ids, "epoch", "sleep_stage", [SLEEP_STAGES[i] for i in stage_idx]),
            _label_rows(epoch_ids, "epoch", "arousal", arousal),
            _label_rows(record_ids, "record", "ahi", ahi),
            _label_rows(
                subject_ids, "subject", "disease", [DISEASE_CATEGORIES[i] for i in disease]
            ),
            _label_rows(subject_ids, "subject", "sex", [SEX_CATEGORIES[i] for i in sex]),
            _label_rows(subject_ids, "subject", "age", age),
        ],
        ignore_index=True,
    )[LABEL_COLUMNS]

    return Metadata(
        epoch_index=epoch_index,
        record_index=record_index,
        subject_index=subject_index,
        labels=labels,
        stage_idx=stage_idx,
        disease_idx=disease_idx,
    )


def generate_embeddings(
    meta: Metadata, embedding_dim: int, config: SyntheticConfig, seed: int
) -> np.ndarray:
    """Epoch embeddings = sleep-stage centroid + disease shift + isotropic Gaussian noise.

    Deterministic from ``seed``; pass distinct per-encoder seeds so equal-dim encoders differ."""
    rng = np.random.default_rng(seed)
    d = embedding_dim
    stage_centroids = rng.standard_normal((len(SLEEP_STAGES), d)) * config.class_sep
    disease_centroids = rng.standard_normal((len(DISEASE_CATEGORIES), d)) * config.subject_sep
    n = meta.stage_idx.size
    emb = stage_centroids[meta.stage_idx] + disease_centroids[meta.disease_idx]
    emb = emb + rng.standard_normal((n, d)) * config.noise_std
    return emb.astype(np.float32)


def generate_fixture(
    embedding_dim: int, config: SyntheticConfig = DEFAULT_CONFIG, seed: int | None = None
) -> SyntheticFixture:
    """Compose metadata (dim-independent) with embeddings (dim/seed-dependent)."""
    meta = generate_metadata(config)
    emb = generate_embeddings(
        meta, embedding_dim, config, seed if seed is not None else config.seed
    )
    return SyntheticFixture(
        embeddings=emb,
        epoch_index=meta.epoch_index,
        record_index=meta.record_index,
        subject_index=meta.subject_index,
        labels=meta.labels,
        embedding_dim=embedding_dim,
    )
