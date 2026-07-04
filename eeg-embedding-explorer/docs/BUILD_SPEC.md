# Autonomous Build Spec — EEG Embedding Explorer

> Location: `docs/BUILD_SPEC.md`. This is the prompt/contract for an autonomous build in the Claude Code instance. It layers execution rules on top of `docs/DESIGN.md`, which remains the source of truth for architecture and data model. Where the two disagree, DESIGN.md wins on *design* and this document wins on *build process*.

## 0. Before you do anything else

**First action — verify prerequisites, before any planning or reasoning.** Read this whole document and `docs/DESIGN.md`, then confirm every item in §1 exists. If anything is missing, contradictory, or ambiguous, **stop and surface it as questions for me to answer interactively** (in-session), not as a wall of text — do not guess, do not work around it, do not start planning against a missing prerequisite. Only once §1 checks out, proceed to §5 (plan mode).

## 1. Prerequisites (fixed; do not modify)

This run is **synthetic-only** (Gates 0–2). It needs no model weights, no torch, and no braindecode — those are for the Gate 3 adapter work, done later in separate sessions. What must be in place now:

- **Repo skeleton, launched as its own project root.** Claude Code is launched from inside `eeg-embedding-explorer/`, so that directory is the project root. Present: `docs/DESIGN.md`, this spec, `README.md`, `CLAUDE.md` (invariants + run/test commands), `pyproject.toml`, and `.claude/settings.json` with the test-gate hook (`.claude/hooks/test-gate.sh`, executable). The hook applies to exactly this project — no cross-subproject scoping needed.
- **Python environment.** Python 3.12, managed by `uv`. `pyproject.toml` lists the light dependency set (numpy, pandas, pyarrow, scikit-learn, umap-learn, streamlit, plotly; pytest + ruff for dev). **Gate 0 resolves and freezes it** (`uv sync`; `uv lock` if no lockfile). After Gate 0, the lock is fixed — do not upgrade dependencies, and do not add torch/braindecode (they belong to Gate 3).
- **Encoder facts — `weights/MANIFEST.md`.** For each of CBraMod, BENDR, LaBraM: embedding dim, native window length, sampling rate, expected montage, finest supported granularity, loss type, pretraining datasets — plus provenance fields (source URL, version, file format, local path, load quirk) for later use. Facts must be filled in now; weight files and paths are used only at Gate 3. Load facts from this file; do not fetch anything over the network.

## 2. Design source of truth

Follow `docs/DESIGN.md` for: the granularity hierarchy and 30s epoch normalization (tile + pool), the label-broadcast rule, rule-based pooling as a cache-keyed toggle, two-phase compute + cache layout, PCA+UMAP projections (t-SNE optional) fit per-(encoder, granularity) with z-score-before-PCA, the Streamlit+Plotly dashboard (2D default, 3D toggle), the adapter contract, and the honesty guardrails. Implement as specified; do not re-derive.

## 3. Non-negotiable invariants

- Projections are fit per (encoder, granularity); embedding dimensionality never needs to match across encoders because only one encoder is projected at a time.
- Standardize (z-score per dimension) before any PCA.
- Separation is not signal. The UI never asserts a discovery; it shows where to look. Always render PCA alongside the nonlinear projection and surface projection params.
- No network access; no dependency upgrades; no fabricated data. If blocked, stop and report — do not stub a real adapter as fake or invent input data to pass a gate.
- The adapter contract and precompute pipeline are required; whether reusable pieces are expressed as skills/subagents is your call for this single run (plain code is fine — see DESIGN §5/§7).

## 4. Ordered verification gates (this run: 0 → 2)

Work in order; do not proceed past a failing gate; run the test suite after each. The test-gate hook enforces this at every stop.

**Gate 0 — environment.** Confirm the pinned env imports cleanly; resolve/freeze it (`uv sync`, `uv lock` if needed). Confirm the repo skeleton and `weights/MANIFEST.md` (facts filled) are present. **Do not require weight files to exist** — this run is synthetic-only. Fail fast on anything else missing.

**Gate 0.5 — encoder facts into the registry.** Read `weights/MANIFEST.md` and populate the adapter registry (dim, window, rate, montage, finest granularity, loss, pretraining datasets) for the trio, so the dummy adapter emits correct shapes and the pretraining-overlap flag has data.

**Gate 1 — synthetic fixture and its tests.** Build the synthetic-embedding generator: embeddings at epoch/record/subject grain with controllable injected cluster structure tied to label sets, plus noise, **from a fixed seed** (commit generator + seed, not large `.npy`). Add pytest tests asserting PCA recovers the injected structure. This is the verification substrate; build it before the dashboard.

**Gate 2 — dashboard on synthetic embeddings (CHECKPOINT + END OF THIS RUN).** Implement the full analysis + dashboard pipeline against the synthetic fixture with the no-op dummy adapter (random embeddings of correct shape). Every control must function end to end: encoder, granularity, label, pooling, projection method + params, 2D/3D, PCA-alongside view, point-count display, pretraining-overlap flag. Then **write `docs/HANDOFF.md`** (§6) and **stop for review**. Do not commit.

Gates 3 (real adapters) and 4 (real-data path) are **defined below for the handoff**, not executed in this run.

**Gate 3 — real adapters, plumbing only (per adapter; separate sessions).** For each of CBraMod, LaBraM, BENDR: implement the adapter and a smoke test feeding a **random tensor** shaped to that encoder's declared montage / rate / window, asserting: loads from the pinned path with no network, forward pass runs on CPU, output is finite with the declared embedding dim, and 30s tiling + pooling yields one vector per epoch. The random tensor carries no structure — a plumbing check, distinct from the synthetic-signal generation DESIGN.md excludes. Pass one adapter before starting the next. (Add torch/braindecode to the env here.) For BENDR, use a smoke test tensor shape of (batch, 90, n_times), post-zero-padding.

**Gate 4 — real-data path, built but unrun.** Implement the precompute pipeline and real-data loader so that, given a staged SHHS subset, one command produces caches at all grains and the dashboard renders them. Do not run against real data. Include the post-handoff runbook (§6) in the README.

## 5. Autonomy protocol

Start in plan mode: produce a plan mapping to Gates 0–2 and wait for approval (first checkpoint). Then work gate by gate, running tests after each. Keep the test-gate hook active throughout so "autonomous" means "self-verifying," not "blind." After each gate, give a one-line progress report. Honor the single end checkpoint at Gate 2 (write `HANDOFF.md`, stop, no commit). If a gate cannot be passed within its intended scope, stop and report the blocker rather than working around it.

## 6. HANDOFF.md — the terminal deliverable

At the Gate 2 checkpoint, write `docs/HANDOFF.md`. Keep it bounded to **what was built / what's next / what to flag** — not a speculative roadmap. Include:
- **What was built and where** — a short map of the pipeline; explicitly where the adapter contract, registry, and dummy adapter live.
- **Gate status** — 0–2 green (with the test-gate confirmation); 3–4 defined, not started.
- **Per-adapter work orders** for CBraMod / BENDR / LaBraM — each a paste-ready brief for a separate focused session: the locked adapter contract, that encoder's declared facts from the manifest, the smoke-test template (random tensor → shape/finite/dim/pooling assertions), and the weight path to stage. These are independent and parallelizable.
- **Real-data path plan (Gate 4)** and a stub SHHS runbook: the ~10-recording label-balanced selection, the precompute command, and the DUA check on committing derived embeddings.
- **Follow-up (bounded)** — adding more encoders, deferred v2 features (cross-encoder comparison, kNN overlays), and any refinements, assumptions, or blockers you hit during the run.

## 7. First post-handoff steps (build Gate 4 to expect this)

These steps are for a human user to follow after the handoff. These are NOT instuctions for an agent to follow.

After the run: (1) execute the Gate 3 work orders in separate focused sessions, reviewing each adapter by hand; (2) stage the credentialed SHHS subset locally, run precompute, confirm the dashboard renders real embeddings. Build the Gate 4 loader, cache paths, and precompute command to be drop-in ready for real data with no code changes — only a data path and a run. Note DUAs before committing any real caches (Git LFS).

## 8. Where to focus review

As in (§7), these steps are for a human user to follow after the handoff.

Concentrate on the adapter/extraction layer (Gate 3). It is the domain-critical, error-prone part closest to real EEG work — whether an embedding (not logits) was extracted, whether channel order and resampling are correct, and whether 30s tiling/pooling is right is knowable only by inspection plus the forward pass. Doing Gate 3 as separate hands-on sessions is deliberate: "it runs" can mask "I can't debug or extend it." Trust the Gates 0–2 output more freely; it is validated by the synthetic-fixture tests.
