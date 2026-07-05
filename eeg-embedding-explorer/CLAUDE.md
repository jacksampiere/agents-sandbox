# CLAUDE.md — EEG Embedding Explorer

Streamlit + Plotly dashboard for qualitatively exploring pretrained EEG foundation-model embeddings, colored by labels/metadata at epoch/record/subject granularity. Full design in `docs/DESIGN.md`; build process in `docs/BUILD_SPEC.md`.

<!-- Uncomment to inherit global agents-sandbox conventions:
@../CLAUDE.md
-->

## Invariants (do not violate)

- **Separation is not signal.** The tool never asserts a discovery; it shows where to look. Never phrase UI or output as if a cluster means something is proven.
- **Epochs are normalized to a 30s unit** across all encoders (tile the native window + pool). This is the comparable base unit.
- **Projections are refit per (encoder, granularity).** Never reuse a projection across encoders or grains.
- **Z-score per dimension before any PCA.**
- **Always render PCA alongside UMAP/t-SNE** and surface projection params. It is the honesty guardrail.
- **No network for weights, no dependency upgrades, no fabricated data.** If blocked, stop and report.

When docs conflict, prefer the active gate scope in `docs/BUILD_SPEC.md`, then `docs/DESIGN.md`, then `weights/MANIFEST.md` for encoder-specific facts. Do not silently resolve ambiguous encoder windowing granularity behavior: either ask the user if it could affect shapes, registry facts, adapter outputs, or dependency scope, or document any assumptions in `docs/HANDOFF.md` if the current gate can continue safely. If you make an assumption that allows the current gate to continue, record it in docs/HANDOFF.md and update the relevant docs.

## Structure

- `src/` — pipeline + dashboard (created by the build)
- `tests/` — pytest suite; the test-gate hook runs this before any stop
- `weights/MANIFEST.md` — encoder provenance + facts; source of the adapter registry
- `cache/` — precomputed embeddings + per-encoder `manifest.json` (synthetic committable)
- `labels.parquet` — long-format `(id, granularity, attribute_name, value)`

## Commands

- Sync/lock env: `uv sync` / `uv lock`
- Run tests: `uv run pytest -q`
- Run dashboard: `uv run streamlit run src/app.py`
- Precompute (once real data staged): `uv run python -m eeg_explorer.precompute …`

## Cache schema (keys)

`cache/<encoder>/<granularity>/[<pooling>/]embeddings.npy` + `index.parquet`. Epoch grain has no pooling subdir (intra-epoch pooling is an adapter detail). Pooling is part of the cache key at record/subject grain. Treat the per-encoder key as a generic embedding-space id (forward-compat with reasoning over latent representations).
