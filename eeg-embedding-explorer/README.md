# EEG Embedding Explorer

Streamlit + Plotly dashboard for qualitatively exploring pretrained EEG foundation-model embeddings,
colored by labels/metadata at epoch/record/subject granularity.

- Design: `docs/DESIGN.md`
- Build process (autonomous, gated): `docs/BUILD_SPEC.md`
- Handoff / next steps (written by the build): `docs/HANDOFF.md`

## Quick start

```bash

# from inside this directory
uv sync
uv run pytest -q
uv run streamlit run src/app.py   # after the build
```

The precompute pipeline and real-data runbook will be documented here once Gate 4 is built.

## Model weights

Not committed (see weights/MANIFEST.md for full provenance, versions, and load quirks). Sources used:
- CBraMod — weighting666/CBraMod (Hugging Face) — pretrained_weights.pth
- BENDR — SPOClab-ca/BENDR — release v0.1-alpha, encoder.pt + contextualizer.pt
- LaBraM — braindecode/Labram-Braindecode (Hugging Face) — braindecode_labram_base.pt
