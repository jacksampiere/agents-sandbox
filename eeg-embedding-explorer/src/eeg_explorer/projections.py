"""Projections: PCA (the honesty guardrail) and UMAP (DESIGN §3, §4).

Fit **per (encoder, granularity)** — callers pass one embedding set, so each fit is independent;
embedding dimensionality never needs to match across encoders. **Z-score per dimension before any
PCA** (encoders differ in scale; a few high-variance dims would otherwise dominate). UMAP is
pre-reduced with a ``min(50, D)`` PCA, then embedded with a fixed random_state for reproducibility.

PCA is deterministic and cheap. UMAP/t-SNE distances and cluster sizes are **not** meaningful — the
dashboard always shows PCA alongside. t-SNE is deferred (DESIGN marks it optional/redundant).
"""

from __future__ import annotations

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PCA_PRE_REDUCE = 50  # components fed into UMAP (DESIGN: fixed min(50, D) pre-reduction)
UMAP_SEED = 42
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1


def zscore(embeddings: np.ndarray) -> np.ndarray:
    """Standardize per dimension. Constant dims (zero variance) are left at zero, not NaN."""
    return StandardScaler().fit_transform(np.asarray(embeddings, dtype=np.float64))


def project_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    z = zscore(embeddings)
    k = min(n_components, z.shape[1], z.shape[0])
    return PCA(n_components=k, random_state=0).fit_transform(z)


def project_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    seed: int = UMAP_SEED,
    pca_pre_reduce: int = PCA_PRE_REDUCE,
) -> np.ndarray:
    z = zscore(embeddings)
    n_samples, dim = z.shape
    pre = min(pca_pre_reduce, dim, n_samples)
    reduced = PCA(n_components=pre, random_state=0).fit_transform(z) if pre < dim else z
    # UMAP requires n_neighbors < n_samples; clamp for small point sets (e.g. subject grain).
    neighbors = int(np.clip(n_neighbors, 2, max(2, n_samples - 1)))
    reducer = umap.UMAP(
        n_components=min(n_components, max(1, n_samples - 1)),
        n_neighbors=neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return np.asarray(reducer.fit_transform(reduced))


def project(embeddings: np.ndarray, method: str, n_components: int = 2, **params) -> np.ndarray:
    """Dispatch by method name. ``PCA`` ignores UMAP-only params (n_neighbors/min_dist)."""
    if method.upper() == "PCA":
        return project_pca(embeddings, n_components=n_components)
    if method.upper() == "UMAP":
        return project_umap(embeddings, n_components=n_components, **params)
    raise ValueError(f"unknown projection method: {method!r}")
