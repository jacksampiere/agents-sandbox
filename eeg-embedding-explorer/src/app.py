"""EEG Embedding Explorer — Streamlit + Plotly dashboard (DESIGN §3-§4).

Reads precomputed caches and renders an interactive scatter, colored by a selectable label at a
selectable granularity/pooling/projection. **PCA is always shown alongside** the nonlinear
projection (the honesty guardrail); projection params, point counts, and the pretraining-overlap
flag are surfaced. The tool shows *where* representations separate — it never asserts a discovery.

Run: ``uv run streamlit run src/app.py`` (after ``python -m eeg_explorer.precompute`` populates caches).
All analysis lives in the ``eeg_explorer`` package; this file is thin presentation only.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from eeg_explorer.cache import (
    available_granularities,
    available_poolings,
    cache_root,
    labels_path,
    list_encoders,
    read_cache,
    read_manifest,
)
from eeg_explorer.labels import available_labels_for_granularity, broadcast_labels, read_labels
from eeg_explorer.projections import (
    DEFAULT_MIN_DIST,
    DEFAULT_N_NEIGHBORS,
    project_pca,
    project_umap,
)
from eeg_explorer.registry import pretraining_overlap

st.set_page_config(page_title="EEG Embedding Explorer", layout="wide")

NONLINEAR_METHODS = ["UMAP"]  # t-SNE deferred (DESIGN: optional/redundant with UMAP)
PROJECTION_METHODS = NONLINEAR_METHODS + ["PCA"]


@st.cache_data(show_spinner=False)
def _load_labels(path_str: str) -> pd.DataFrame:
    return read_labels(path_str)


@st.cache_data(show_spinner=False)
def _load_cache(root_str: str, encoder: str, grain: str, pooling: str | None):
    return read_cache(encoder, grain, pooling, root=root_str)


@st.cache_data(show_spinner=False)
def _pca_coords(root_str: str, encoder: str, grain: str, pooling: str | None, n_components: int):
    emb, _ = read_cache(encoder, grain, pooling, root=root_str)
    return project_pca(emb, n_components=n_components)


@st.cache_data(show_spinner=False)
def _umap_coords(
    root_str: str,
    encoder: str,
    grain: str,
    pooling: str | None,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
):
    emb, _ = read_cache(encoder, grain, pooling, root=root_str)
    return project_umap(emb, n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)


def _color_series(values: pd.Series) -> pd.Series:
    """Continuous labels (age, ahi) get a numeric colorbar; categorical ones get discrete colors."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric if numeric.notna().all() else values.astype(str)


def _scatter(coords, color, title: str, n_components: int):
    df = pd.DataFrame(coords, columns=[f"dim{i + 1}" for i in range(coords.shape[1])])
    df["color"] = color.to_numpy()
    common = dict(color="color", title=title, opacity=0.75)
    if n_components >= 3 and coords.shape[1] >= 3:
        fig = px.scatter_3d(df, x="dim1", y="dim2", z="dim3", **common)
        fig.update_traces(marker_size=3)
    else:
        fig = px.scatter(df, x="dim1", y="dim2", **common)
        fig.update_traces(marker_size=5)
    fig.update_layout(legend_title_text="", margin=dict(l=0, r=0, t=40, b=0), height=520)
    return fig


def main() -> None:
    st.title("EEG Embedding Explorer")
    st.caption(
        "Separation is not signal. This tool shows *where* representations separate to guide "
        "investigation — it does not assert that any separation is meaningful. UMAP inter-cluster "
        "distances and cluster sizes are not meaningful; always read the PCA panel alongside."
    )

    root = cache_root()
    root_str = str(root)
    encoders = list_encoders(root)
    if not encoders:
        st.error(
            f"No caches found under {root}. Run "
            "`PYTHONPATH=src uv run python -m eeg_explorer.precompute` to build the synthetic caches."
        )
        return

    labels = _load_labels(str(labels_path()))

    with st.sidebar:
        st.header("Controls")
        encoder = st.selectbox("Encoder", encoders, key="encoder")
        grains = available_granularities(encoder, root)
        granularity = st.selectbox("Granularity", grains, key="granularity")

        if granularity == "epoch":
            pooling = None
            st.caption("Pooling: n/a at epoch grain (intra-epoch pooling is an adapter detail).")
        else:
            poolings = available_poolings(encoder, granularity, root)
            pooling = st.selectbox("Pooling (part of the cache key)", poolings, key="pooling")

        label_options = available_labels_for_granularity(labels, granularity)
        label_attr = st.selectbox("Color by label", label_options, key="label")

        method = st.selectbox("Projection method (right panel)", PROJECTION_METHODS, key="method")
        dims = st.radio("Dimensions", ["2D", "3D"], horizontal=True, key="dims")
        n_components = 3 if dims == "3D" else 2

        n_neighbors, min_dist = DEFAULT_N_NEIGHBORS, DEFAULT_MIN_DIST
        if method == "UMAP":
            n_neighbors = st.slider(
                "UMAP n_neighbors", 2, 50, DEFAULT_N_NEIGHBORS, key="n_neighbors"
            )
            min_dist = st.slider("UMAP min_dist", 0.0, 0.99, DEFAULT_MIN_DIST, 0.01, key="min_dist")

    _, index = _load_cache(root_str, encoder, granularity, pooling)
    color = _color_series(broadcast_labels(labels, index, label_attr))

    manifest = read_manifest(encoder, root)
    viz_dataset = manifest.get("source_dataset", "")
    pretraining = manifest.get("pretraining_datasets", [])
    overlaps = pretraining_overlap(viz_dataset, pretraining)

    c1, c2, c3 = st.columns(3)
    c1.metric("Points", f"{len(index):,}", help=f"{granularity} grain")
    c2.metric("Granularity", granularity)
    c3.metric("Embedding dim", manifest.get("embedding_dim", "?"))

    if overlaps:
        st.warning(
            f"⚠️ Pretraining overlap: the visualization dataset ({viz_dataset!r}) appears in "
            f"{encoder}'s pretraining corpus {pretraining}. Apparent separation may reflect "
            "memorization, not genuine structure."
        )
    else:
        st.info(
            f"No pretraining overlap: visualization dataset = {viz_dataset!r}; "
            f"{encoder} pretraining = {pretraining}."
        )

    left, right = st.columns(2)
    with left:
        pca = _pca_coords(root_str, encoder, granularity, pooling, n_components)
        st.plotly_chart(
            _scatter(pca, color, "PCA (linear sanity check)", n_components),
            use_container_width=True,
        )
        st.caption("PCA is deterministic; distances are meaningful. This is the honesty guardrail.")
    with right:
        if method == "UMAP":
            coords = _umap_coords(
                root_str, encoder, granularity, pooling, n_components, n_neighbors, min_dist
            )
            title = f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist:g}, seed=42)"
            note = "UMAP distances/sizes are NOT meaningful; params shown above."
        else:
            coords = pca
            title = "PCA (secondary panel)"
            note = "Same linear PCA as the left panel."
        st.plotly_chart(_scatter(coords, color, title, n_components), use_container_width=True)
        st.caption(note)


main()
