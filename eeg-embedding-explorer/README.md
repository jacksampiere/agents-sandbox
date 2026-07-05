# EEG Embedding Explorer

Streamlit + Plotly dashboard for qualitatively exploring pretrained EEG foundation-model embeddings,
colored by labels/metadata at epoch/record/subject granularity.

- Design: `docs/DESIGN.md`
- Build process (autonomous, gated): `docs/BUILD_SPEC.md`
- Handoff / next steps (written by the build): `docs/HANDOFF.md`

## Quick start

```bash
# from inside this directory
uv sync                                                   # install pinned deps (creates the venv)
uv run pytest -q                                          # run the test suite

# Build the synthetic embedding caches (deterministic, from a fixed seed).
# PYTHONPATH=src because the package lives under src/ and is not installed ([tool.uv] package=false).
PYTHONPATH=src uv run python -m eeg_explorer.precompute

uv run streamlit run src/app.py                           # launch the dashboard → http://localhost:8501
```

The synthetic `cache/` and `labels.parquet` are **generated artifacts (git-ignored)** — they
regenerate deterministically from the seed in `src/eeg_explorer/synthetic.py`, so a fresh clone just
runs the precompute step above before launching the dashboard.

## Model weights

Not committed (see weights/MANIFEST.md for full provenance, versions, and load quirks). Sources used:
- CBraMod — weighting666/CBraMod (Hugging Face) — pretrained_weights.pth
- BENDR — SPOClab-ca/BENDR — release v0.1-alpha, encoder.pt + contextualizer.pt
- LaBraM — braindecode/Labram-Braindecode (Hugging Face) — braindecode_labram_base.pt
