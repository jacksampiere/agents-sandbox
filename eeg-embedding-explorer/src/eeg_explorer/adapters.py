"""Adapter contract and the no-op dummy adapter (DESIGN §5).

The **adapter contract** is the interface every encoder implements: given raw 30s EEG epochs, resample
to the model's rate, select/reorder channels to its montage, window/tile the 30s epoch per the model's
native window, run the forward pass, and intra-epoch pool sub-windows to one ``(D,)`` vector per epoch.

This synthetic run ships only the ``DummyAdapter`` (random embeddings of the correct shape). It tests
the contract and the whole precompute→cache→dashboard path before any real model exists. The real
CBraMod / BENDR / LaBraM adapters (Gate 3) subclass ``Adapter`` and implement ``embed_epochs``; their
work orders and this locked contract are carried in docs/HANDOFF.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from eeg_explorer.registry import EncoderFacts, load_registry


class Adapter(ABC):
    """The interface every encoder implements. Facts come from the registry (weights/MANIFEST.md)."""

    def __init__(self, facts: EncoderFacts):
        self.facts = facts

    @property
    def name(self) -> str:
        return self.facts.name

    @property
    def embedding_dim(self) -> int:
        return self.facts.embedding_dim

    @abstractmethod
    def embed_epochs(self, raw_epochs: np.ndarray, channels: Sequence[str]) -> np.ndarray:
        """Map ``(N, C, T)`` raw 30s epochs to ``(N, D)`` float32 — one pooled vector per epoch.

        ``channels`` names the C axis so the adapter can select/reorder to the model's montage. The
        30s epoch must be tiled/packed to the model's expected forward input (never center-cropped);
        multiple sub-window outputs are intra-epoch pooled (default mean) to a single ``(D,)`` vector.
        """


class DummyAdapter(Adapter):
    """No-op adapter: emits seeded random embeddings of shape ``(N, D)``, ignoring raw content.

    Not a fake real model — a deliberate contract/pipeline probe. It carries no injected structure
    (that is the synthetic *fixture*'s job); it only proves shapes and plumbing.
    """

    def __init__(self, facts: EncoderFacts, seed: int = 0):
        super().__init__(facts)
        self._seed = seed

    def embed_epochs(self, raw_epochs: np.ndarray, channels: Sequence[str]) -> np.ndarray:
        n = int(np.asarray(raw_epochs).shape[0])
        rng = np.random.default_rng(self._seed)
        return rng.standard_normal((n, self.embedding_dim)).astype(np.float32)


def dummy_adapter(
    encoder_name: str, registry: dict[str, EncoderFacts] | None = None, seed: int = 0
) -> DummyAdapter:
    reg = registry if registry is not None else load_registry()
    return DummyAdapter(reg[encoder_name], seed=seed)
