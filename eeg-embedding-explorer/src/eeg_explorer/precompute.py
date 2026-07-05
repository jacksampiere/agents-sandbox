"""Precompute — the build-time forward pass + pooling that populates the cache (DESIGN §3).

Run once to materialize embeddings; the dashboard then reads caches and projects on the fly. Two
entry points write the *same* cache layout:

- ``build_synthetic_caches`` — materialize the structured synthetic fixture per encoder (dims from
  the registry). This is the dashboard's demo data for the Gates 0-2 run.
- ``precompute_from_adapter`` — run any :class:`~eeg_explorer.adapters.Adapter` over raw 30s epochs.
  Exercised here with the ``DummyAdapter`` to prove the real-data path (Gate 4 swaps in real data +
  real adapters, no code change).

Config lives in module constants (edit here; the repo convention favors config vars over CLI args).
Invoke with ``PYTHONPATH=src uv run python -m eeg_explorer.precompute``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import date, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_explorer.adapters import Adapter
from eeg_explorer.cache import cache_root, labels_path, write_cache, write_manifest
from eeg_explorer.labels import write_labels
from eeg_explorer.pooling import DEFAULT_POOLING, pool_embeddings
from eeg_explorer.registry import EncoderFacts, load_registry
from eeg_explorer.synthetic import DEFAULT_CONFIG, SyntheticConfig, generate_fixture

# --- config (edit here) ---
POOLINGS_TO_BUILD: tuple[str, ...] = ("mean", "max")
VIZ_DATASET = "synthetic"  # the dataset being visualized; drives the pretraining-overlap flag
_EPOCH_INDEX_COLS = ["epoch_id", "record_id", "subject_id"]


def _encoder_seed(base_seed: int, name: str) -> int:
    """Stable per-encoder offset so equal-dim encoders (CBraMod & LaBraM are both 200-D) differ."""
    return base_seed + (sum(ord(c) for c in name) % 997)


def _write_grain_caches(
    encoder: str,
    epoch_embeddings: np.ndarray,
    epoch_index: pd.DataFrame,
    root: Path,
    poolings: Sequence[str],
) -> None:
    write_cache(encoder, "epoch", epoch_embeddings, epoch_index[_EPOCH_INDEX_COLS], root=root)
    for grain in ("record", "subject"):
        for pooling in poolings:
            pooled, index = pool_embeddings(epoch_embeddings, epoch_index, grain, pooling)
            write_cache(encoder, grain, pooled, index, pooling=pooling, root=root)


def _manifest(facts: EncoderFacts, *, poolings: Sequence[str], extra: dict | None = None) -> dict:
    manifest = {
        "encoder": facts.name,
        "embedding_dim": facts.embedding_dim,
        "source_dataset": VIZ_DATASET,
        "pretraining_datasets": list(facts.pretraining_datasets),
        "preprocessing": {
            "epoch_input_s": facts.epoch_input_s,
            "sampling_rate_hz": facts.sampling_rate_hz,
            "expected_montage": facts.expected_montage,
        },
        "poolings": list(poolings),
        "weights_hash": None,  # synthetic / dummy — no real weights loaded
        "date": date.today().isoformat(),
        "generated_utc": pd.Timestamp.now(tz=timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def build_synthetic_caches(
    root: Path | str | None = None,
    labels_out: Path | str | None = None,
    config: SyntheticConfig = DEFAULT_CONFIG,
    poolings: Sequence[str] = POOLINGS_TO_BUILD,
) -> dict[str, EncoderFacts]:
    """Materialize the synthetic fixture into the cache for every registered encoder."""
    root = Path(root) if root is not None else cache_root()
    labels_out = Path(labels_out) if labels_out is not None else labels_path()
    registry = load_registry()

    labels_written = False
    for name, facts in registry.items():
        fixture = generate_fixture(
            facts.embedding_dim, config, seed=_encoder_seed(config.seed, name)
        )
        _write_grain_caches(name, fixture.embeddings, fixture.epoch_index, root, poolings)
        write_manifest(
            name,
            _manifest(
                facts,
                poolings=poolings,
                extra={
                    "synthetic": True,
                    "subset": "generated-from-seed",
                    "synthetic_config": asdict(config),
                    "seed": _encoder_seed(config.seed, name),
                },
            ),
            root=root,
        )
        if not labels_written:  # labels are shared across encoders (same fixture metadata)
            labels_out.parent.mkdir(parents=True, exist_ok=True)
            write_labels(fixture.labels, labels_out)
            labels_written = True
    return registry


def precompute_from_adapter(
    adapter: Adapter,
    raw_epochs: np.ndarray,
    channels: Sequence[str],
    epoch_index: pd.DataFrame,
    root: Path | str | None = None,
    poolings: Sequence[str] = POOLINGS_TO_BUILD,
    manifest_extra: dict | None = None,
) -> np.ndarray:
    """Run an adapter over raw 30s epochs and write all grain caches. Returns epoch embeddings."""
    root = Path(root) if root is not None else cache_root()
    epoch_embeddings = adapter.embed_epochs(raw_epochs, channels)
    _write_grain_caches(adapter.name, epoch_embeddings, epoch_index, root, poolings)
    write_manifest(
        adapter.name, _manifest(adapter.facts, poolings=poolings, extra=manifest_extra), root=root
    )
    return epoch_embeddings


def main() -> None:
    build_synthetic_caches()
    print(f"Synthetic caches written to {cache_root()}")
    print(f"Labels written to {labels_path()}")
    print(f"Default pooling: {DEFAULT_POOLING}; poolings built: {list(POOLINGS_TO_BUILD)}")


if __name__ == "__main__":
    main()
