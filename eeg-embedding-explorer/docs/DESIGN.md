# EEG Embedding Explorer — Design Brief

> Intended location: `docs/DESIGN.md` in the `eeg-embedding-explorer` subproject under `agents-sandbox`.
> Single source of truth for **design** (architecture, data model, contracts). `docs/BUILD_SPEC.md` governs the **build process** and defers to this doc on design. It is written to be self-contained so a fresh Claude Code instance can pick up the work without the originating conversation.

## 1. Goal and scope

This is a Python data-visualization / ML-tooling engineering task: build a Streamlit+Plotly dashboard and a NumPy/scikit-learn pipeline over **precomputed embedding vectors (opaque float32 arrays)**. There is no clinical inference and no diagnostic or wet-lab component — the vectors are treated as generic numeric data for dimensionality reduction, plotting, and caching.

Build a dashboard that lets a human qualitatively explore representations produced by pretrained time series foundation models, colored by downstream-task labels and subject/record metadata. The purpose is **qualitative inspection of embedding geometry** — a visualization/QA aid, not benchmarking: surface where representations separate by some label so the user knows what to investigate further.

In scope for v1:
- A Streamlit + Plotly dashboard that loads precomputed embeddings and renders an interactive 2D scatter (3D as a secondary toggle), colored by a selectable label set, at a selectable granularity, with a selectable pooling rule and projection method.
- Support for at least two encoders (target three: CBraMod, BENDR, LaBraM) so the multi-encoder code path exists from day one.
- A synthetic-embedding test fixture and a small real demo dataset.

Explicitly out of scope for v1 (deferred to keep the build honest and small):
- Agent-as-runtime-component (e.g., encoders as tools an agent selects, reasoning over latent representations). Clean v2 boundary.
- Synthetic raw-signal generation and synthetic encoders. Latent geometry of real encoders can't be forecast from synthetic signals, so this earns its place only later, for testing specific encoder invariances.
- Runtime reconciliation of arbitrary model artifact formats. Reconcile up front via the adapter registry (§5).
- kNN prediction/uncertainty overlays and cross-encoder comparison/neighbor agreement. Cross-encoder comparison is its own method, not a trivial extension: encoders produce embeddings of different dimensionality and incommensurable spaces (different dims, scales, bases), so vectors can't be directly compared or co-plotted. It requires representational-similarity methods (CKA, RSA) or alignment (CCA/Procrustes), not naive distance. v1 supports only side-by-side independent plots, each labeled as a separate space.
- Any model training or formal evaluation. Nothing is trained, so there is no train/test split and no leakage to prevent — unless a kNN feature is later added, at which point subject-level splits become necessary.

## 2. Core data model

The conceptual crux. Do not treat the dashboard as "one scatter, recolor by label." Different label granularities produce genuinely different point sets, and the design surfaces that rather than hiding it.

**Granularity hierarchy:** `epoch` → `record` → `subject`.
- `epoch` is normalized to a 30-second unit across all encoders. Aligns with sleep-stage labels and makes encoders comparable.
- `record` is one recording (one night).
- `subject` is one person. With ~1 record per subject in the demo set, record and subject nearly coincide, but keep them distinct in the model.

**30s epoch normalization:** every adapter emits exactly one embedding vector per normalized
30s epoch. Distinguish the adapter's project-level epoch input from the model's internal patch
or pretraining crop length:

- `epoch_input_s` is the raw time span consumed by the project adapter to produce one epoch
  embedding. For the starting trio, this is 30s.
- `patch_s` is an internal model patch/sub-window size, when applicable. CBraMod and LaBraM
  operate over 1s patches, but the adapter should construct the model's full 30-patch input
  and run it in one forward call, not loop over 30 independent model calls.
- `pretraining_crop_s` records the original crop/window length used in pretraining when that
  differs from the project adapter input. This is provenance only unless the adapter explicitly
  requires it.

For models that return multiple patch/sub-window embeddings for a 30s epoch, the adapter pools
those outputs to one 30s vector, defaulting to mean pooling. This intra-epoch pooling is an
adapter implementation detail, not a user-facing control. Do not center-crop a shorter model
window from the epoch; tile/pack the full 30s epoch according to the model's expected forward
input.

**Encoder-declared finest granularity:** each encoder declares, in the registry, the finest
granularity it can serve. An encoder whose required `epoch_input_s` exceeds 30s is registered
as record-level and does not offer epoch-grain views; the UI only exposes (granularity, label)
combinations it can serve. The starting trio all support epoch grain because their project
adapters emit one vector per 30s epoch.

**Label granularity and the broadcast rule:** every label carries a native granularity — sleep stage / arousal / apnea event are epoch-level; recording metadata (e.g. AHI) is record-level; age / sex / disease class / medication are subject-level. A label may color points at its native granularity **or any finer granularity by broadcasting down** (every epoch inherits its subject's age). It may not be applied at a coarser granularity without aggregation (generally meaningless for categoricals). The point set is determined by the chosen granularity; the projection is refit per (encoder, granularity); and the UI states the point count explicitly (e.g. "8,142 epoch points" vs "10 subject points") so the granularity change is visible rather than disguised as a recolor.

**Pooling as a rule-based toggle:** aggregation up the hierarchy (epoch → record → subject) is user-selectable among rule-based options (start with mean; add max and, for encoders emitting a summary/CLS token, that token; mean+std concat possible later). Pooling is part of the cache key. Start with mean only through the synthetic phase.

## 3. Architecture

**Two-phase compute.** The only expensive, model-dependent step is the forward pass; run it once at build time and cache. Projections are cheap on these sizes and run on the fly during interaction. This is why no GPU is needed: base models are small (4–25M params) and CPU inference over ~10 recordings is a slow-but-trivial one-time job.

**Repo layout (illustrative):**
```
eeg-embedding-explorer/
  .claude/
    hooks/
      test-gate.sh
    settings.json
  cache/                        # created by precompute (synthetic committable; real gated by DUA)
    <encoder>/
      epoch/
        embeddings.npy            # (N_epoch, D) float32, intra-epoch pooled
        index.parquet             # row -> subject_id, record_id, epoch_id, native epoch labels
      record/
        <pooling>/
          embeddings.npy # epochs pooled per record
          index.parquet
      subject/
        <pooling>/
          embeddings.npy
          index.parquet
      manifest.json                   # encoder version + weights hash, preprocessing params,
                                      # source dataset + subset, pretraining datasets, date
  docs/
    BUILD_SPEC.md
    DESIGN.md
    HANDOFF.md                  # produced by the build
  src/…                         # created by the build
  tests/…                       # created by the build
  weights/                      # staged model artifacts (NOT committed; needed only at Gate 3)
    <encoder>/…                 #   checkpoint files, loaded only from these pinned paths
    MANIFEST.md                 #   per-encoder provenance + facts (source of §5 registry facts)
  CLAUDE.md
  labels.parquet                      # long-format: (id, granularity, attribute_name, value)
  pyproject.toml
  README.md
```
Embeddings join to labels by id. Pooling only varies the record/subject caches (epoch embeddings come from the adapter's intra-epoch pooling). Note the two distinct manifests: `weights/MANIFEST.md` (provenance + facts staged by hand) vs. `cache/<encoder>/manifest.json` (generated by precompute).

**Projections:** PCA and UMAP are core; t-SNE optional/stretch (largely redundant with UMAP, slower). Project to 2 or 3 components. Standardize features (z-score per dimension) before any PCA, since encoders differ in scale and a few high-variance dims would otherwise dominate. For pre-reduction before UMAP/t-SNE, start with a fixed `min(50, D)` PCA; later, select PC count per embedding set by an explained-variance threshold (~90–95%) on the fly with a scree readout. PCA is deterministic and cheap (cache optional). UMAP/t-SNE are stochastic — cache a default-params, fixed-seed projection per (encoder, granularity, method) for instant first load and reproducible artifacts, and recompute on the fly (in-session cache) when the user changes params.

**Dashboard:** Streamlit + Plotly. Default 2D scatter; 3D as a secondary toggle (emit up to 3 components from the projection step so 3D is a plotting branch, not a rewrite). Controls: encoder, granularity, label set, pooling, projection method + params, 2D/3D.

## 4. Methodological guardrails (low-lift, build in from the start)

Honesty annotations, not evaluation. The tool's value depends on the leads it generates being real.

- **PCA shown alongside the nonlinear projection.** UMAP/t-SNE routinely manufacture or destroy apparent structure; inter-cluster distances and cluster sizes are not meaningful. A linear PCA view is a near-free sanity check and the single most important guardrail. Surface projection params (perplexity / n_neighbors) in the UI.
- **Pretraining-dataset annotation and overlap flag.** Each encoder's registry entry records its pretraining datasets; the UI flags when the visualization dataset overlaps. Material: several open EEG models were pretrained on public PSG corpora used here (e.g. a BIOT checkpoint pretrained on SHHS; CBraMod/LaBraM on large multi-dataset corpora), so apparent in-distribution separation may reflect memorization.
- **Class/point counts per label** so imbalance is visible.
- **Invariant — separation is not signal.** The tool never asserts a discovery; it shows where to look. Stated as an invariant in CLAUDE.md.

## 5. Models and the adapter contract

Starting trio (open weights, CPU-feasible):
- [**CBraMod**](https://arxiv.org/abs/2412.07236) (ICLR 2025) — 30s-native at 200 Hz, masked-reconstruction; expose embeddings by replacing the projection head with identity.
- [**BENDR**](https://arxiv.org/abs/2101.12037) (2021) — contrastive; provides the loss-type contrast against the reconstruction models.
- [**LaBraM**](https://arxiv.org/abs/2405.18765) (ICLR 2024) — masked neural-code prediction; short channel-patch segments pooled up to 30s.

**Adapter contract (the interface every encoder implements):** given a raw EEG segment, an adapter must resample to the model's sample rate, select/reorder channels to the model's montage, window to the encoder's native window tiling the 30s epoch, run the forward pass, and pool sub-windows to a single 30s epoch vector of shape `(D,)`. Adapters register with: name, embedding dim, native window length, windowing approach (single-pass vs. tile+pool), sampling rate, expected montage, finest supported granularity, loss type, and pretraining datasets. **These registry facts are sourced from `weights/MANIFEST.md`** (staged by hand; see BUILD_SPEC §1). A no-op dummy adapter returning random embeddings of correct shape is the first one built (tests the contract and the dashboard before any real model).

**What is required vs. open:** the adapter contract, the registry, and the precompute pipeline are required and prescribed here. Whether the reusable pieces are expressed as Claude Code skills/subagents or as plain code is left open — for a single autonomous run, plain code is fine; the harness authoring is a separate exercise (§7).

## 6. Data

Decouple "does the tool work" from "is there real separation." Success criterion is correct rendering at the correct granularity with correct pooling and joins — never "clusters appear." Separation is the user's discovery.

- **Test fixture — synthetic embeddings (not signals).** Generate embeddings directly at epoch/record/subject grain with controllable injected cluster structure tied to label sets, plus noise. Bypasses the encoder and lets tests assert structure recovery (e.g. label-A separates from label-B under PCA) deterministically. Verification substrate and the first thing built. **Prefer generating from a fixed seed and committing the generator + seed, not large `.npy` files** — this keeps the synthetic phase off Git LFS entirely.
- **Demo data — a small real subset.** ~10 recordings from ~10 subjects from a public NSRR dataset (SHHS easiest, rich metadata), chosen for label coverage and balance (all five sleep stages; a spread of each record/subject label) so every toggle visibly does something. Deferred to post-handoff (BUILD_SPEC §6).
- **DUA caveat:** verify that data-use agreements permit committing derived embeddings before pushing caches publicly.

**Forward-compatibility schema note.** This project is expected to share its substrate (adapter registry + embedding store) with a possible future project involving cross-modal / retrieval-oriented reasoning agents over these representations. Two cheap choices now avoid a schema migration later: (1) treat `labels.parquet` as generic long-format `(id, granularity, attribute_name, value)` rather than fixed dashboard columns, so it can hold arbitrary outcomes later; (2) treat the per-encoder cache key as a generic embedding-space identifier in code, not hardcoded to a single-model single-modality signal encoder, so a future aligned or multi-modal space stores the same way. Nothing else changes — no retrieval index, agent loop, or alignment logic belongs here. Any crucial info that needs to be passed along to the future project should be included in the `HANDOFF.md` written after Gate 2.

Real demo caches are committed via Git LFS post-handoff (subject to the DUA check); the synthetic fixture is not (generated from seed).
