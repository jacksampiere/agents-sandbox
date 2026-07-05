# Model Weights Manifest

Provenance and encoder facts for the starting trio. **The build reads this file to populate the
adapter registry** (DESIGN §5) and to know where to load weights from (BUILD_SPEC §1).

Rules:
- Every `TODO` should be filled in before handoff. **Facts** (dim, window, rate, montage, granularity, loss,
  pretraining datasets) are needed for the synthetic Gates 0–2 run.
- **Provenance** (source, version, format, path, load quirk) and the actual weight files are only
  needed at Gate 3 (real adapters). Note that paths may point at not-yet-downloaded files for now.
- The build loads weights **only** from the `local_path` below and **must not** fetch over the network unless `braindecode` is used.

**Data-compatibility note (read before Gate 4):** the demo dataset (SHHS) supplies only 2 EEG
channels — C3 (referenced to A2) and C4 (referenced to A1) — not a full 10-20 montage. All three
encoders below are channel-count-flexible by design (CBraMod's adaptive positional encoding,
LaBraM's flexible channel patches, BENDR's Deep1010 zero-padding for missing channels), so this is
expected to work, not a blocker — but do not hardcode a fixed full-montage assumption anywhere in
the adapter; pass the channel list actually present in the source data at call time.

---

## cbramod

- embedding_dim: 200
- epoch_input_s: 30
- native_window_s: 30  # backward-compatible alias for epoch_input_s
- patch_s: 1
- sampling_rate_hz: 200
- windowing: single-pass
- windowing_note: feed as (batch, n_channels, 30, 200) — 30 one-second patches in one forward call, not 30 separate calls
- expected_montage: channel-adaptive (ACPE); pass the channel-name list present in the source data at call time.
SHHS supplies only C3, C4 — feed those two. Reference full-10-20 list (not required, non-SHHS use only):
[Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2]
- finest_granularity: epoch
- loss_type: masked-reconstruction
- pretraining_datasets: [TUEG]
- source_url: https://github.com/wjq-learning/CBraMod   # keep original repo — braindecode's CBraMod page ships architecture only, no weights
- version: HF hash at the time of artifact download is 500543c7e30bda1b22bfd51a49301b238dee21fd
- file_format: PyTorch .pth checkpoint; state_dict
- local_path: weights/cbramod/pretrained_weights.pth
- load_quirk: load via model.load_state_dict(...), then model.proj_out = nn.Identity() to expose embeddings instead of the reconstruction head (this is the repo's own documented recipe)

## bendr

- embedding_dim: 512  # extract the transformer's contextualized output (start-token / aggregate), not the raw conv-stage vectors. OPEN ITEM: confirm at Gate 3 whether this output is 512-dim (== encoder_h) or the internal transformer width — don't assume; let the smoke test's shape assertion catch a mismatch.
- epoch_input_s: 30
- native_window_s: 30  # backward-compatible alias for epoch_input_s
- patch_s: null
- windowing: single-pass
- windowing_note: original BENDR pretraining used longer crops, but this adapter consumes 30s epochs
- sampling_rate_hz: 256
- expected_montage: DN3 Deep1010 fixed channel-slot mapping — VERIFIED, 90 channels total, fixed order:
indices 0-76 EEG (77 slots, unifies old/new 10-20 naming e.g. both T3 and T7 present),
77-80 EOG (VEOGL, VEOGR, HEOGL, HEOGR), 81-83 reference (A1, A2, REF), 84 SCALE, 85-89 EX1-EX5.
Full ordered list (index: name) — 0 NZ, 1 FP1, 2 FPZ, 3 FP2, 4 AF7, 5 AF3, 6 AFZ, 7 AF4, 8 AF8,
9 F9, 10 F7, 11 F5, 12 F3, 13 F1, 14 FZ, 15 F2, 16 F4, 17 F6, 18 F8, 19 F10, 20 FT9, 21 FT7,
22 FC5, 23 FC3, 24 FC1, 25 FCZ, 26 FC2, 27 FC4, 28 FC6, 29 FT8, 30 FT10, 31 T9, 32 T7, 33 T3,
34 C5, 35 C3, 36 C1, 37 CZ, 38 C2, 39 C4, 40 C6, 41 T4, 42 T8, 43 T10, 44 TP9, 45 TP7, 46 CP5,
47 CP3, 48 CP1, 49 CPZ, 50 CP2, 51 CP4, 52 CP6, 53 TP8, 54 TP10, 55 P9, 56 P7, 57 T5, 58 P5,
59 P3, 60 P1, 61 PZ, 62 P2, 63 P4, 64 P6, 65 T6, 66 P8, 67 P10, 68 PO7, 69 PO3, 70 POZ, 71 PO4,
72 PO8, 73 O1, 74 OZ, 75 O2, 76 IZ, 77 VEOGL, 78 VEOGR, 79 HEOGL, 80 HEOGR, 81 A1, 82 A2, 83 REF,
84 SCALE, 85 EX1, 86 EX2, 87 EX3, 88 EX4, 89 EX5.
Surplus channels ignored, missing channels zero-padded by design (this is what makes SHHS's
2-channel input workable). SHHS maps to indices 35 (C3) and 39 (C4); everything else, including
A1/A2 (81/82), is zero.
CAVEAT: Deep1010 expects individually-referenced signals with separate reference slots, but SHHS
delivers pre-computed bipolar derivations (C3-A2, C4-A1) — the referencing is already baked in
and can't be decomposed back out. Adapter decision (stated here, not left implicit): feed the
bipolar signal directly into the C3/C4 slots and leave A1/A2 at zero. This is an approximation,
not what Deep1010 architecturally expects — acceptable for a hypothesis-generation tool, but
worth remembering if BENDR's SHHS embeddings look off relative to CBraMod/LaBraM on the same data.
- finest_granularity: epoch
- loss_type: contrastive
- pretraining_datasets: [TUEG v1.1, TUEG v1.2]
- source_url: https://github.com/SPOClab-ca/BENDR
- version: release tag v0.1-alpha (commit 5a6f49365a729852a5666aa744938aa4238f7ecd)
- file_format: two separate PyTorch state_dicts (not a single combined checkpoint)
- encoder_local_path: weights/bendr/encoder.pt
- contextualizer_local_path: weights/bendr/contextualizer.pt
- load_quirk: two-stage load — encoder.pt into the conv encoder class, contextualizer.pt into the transformer contextualizer class. Loading the original DN3 classes pulls in the dn3 package as a dependency (add at Gate 3, alongside torch/braindecode); alternative is a minimal from-scratch reimplementation of just the two classes needed, avoiding the full DN3 dependency — decide which before starting the adapter, don't let the agent choose silently.

## labram

- embedding_dim: 200  # LaBraM-Base; Large=400, Huge=800
- epoch_input_s: 30
- native_window_s: 30  # backward-compatible alias for project adapter input; do not smoke-test only 1s
- patch_s: 1
- windowing: single-pass
- windowing_note: feed 30 one-second patches / 6000 samples in one forward call for a 30s epoch; pool per-patch outputs to one 30s vector if needed
- sampling_rate_hz: 200
- expected_montage: channel-adaptive (flexible channel patches with spatial embeddings); pass the channel-name list present in the source data at call time. SHHS supplies only C3, C4.
- finest_granularity: epoch
- loss_type: masked prediction of vector-quantized neural tokens
- pretraining_datasets:
  - BCI Competition IV-1
  - Emobrain
  - Grasp and Lift EEG Challenge
  - Inria BCI Challenge
  - EEG Motor Movement/Imagery
  - Raw EEG Data
  - Resting State EEG Data
  - SEED series
  - Siena Scalp EEG Database
  - SPIS Resting State Dataset
  - Target Versus Non-Target
  - TUAR
  - TUEP
  - TUSZ
  - TUSL
  - self-collected EEG data
- source_url: https://huggingface.co/braindecode/Labram-Braindecode; original repo for reference: https://github.com/935963004/LaBraM
- version: HF commit at the time of artifact download is 1f261c98703d4c6a67a0c9e5cdac1661736eee37
- file_format: PyTorch state_dict
- local_path: weights/labram/braindecode_labram_base.pt
- load_quirk: instantiate braindecode.models.Labram(...), load state_dict from local_path with torch.load(..., map_location="cpu"), then model.load_state_dict(state). Do not call load_state_dict_from_url during this project.