"""Two-phase cache: read/write the on-disk embedding store (DESIGN §3).

Layout (the encoder key is treated as a *generic embedding-space id*, per the forward-compat note):

    cache/<encoder>/epoch/embeddings.npy + index.parquet          # no pooling subdir
    cache/<encoder>/<record|subject>/<pooling>/embeddings.npy + index.parquet
    cache/<encoder>/manifest.json                                 # provenance for this space

Roots are resolved from ``EEG_CACHE_ROOT`` / ``EEG_LABELS_PATH`` (defaulting to the repo) so tests
can redirect them to a temp dir without touching code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]


def cache_root() -> Path:
    return Path(os.environ.get("EEG_CACHE_ROOT", _REPO_ROOT / "cache"))


def labels_path() -> Path:
    return Path(os.environ.get("EEG_LABELS_PATH", _REPO_ROOT / "labels.parquet"))


def _root(root: Path | str | None) -> Path:
    return Path(root) if root is not None else cache_root()


def cache_dir(
    encoder: str, granularity: str, pooling: str | None = None, root: Path | str | None = None
) -> Path:
    d = _root(root) / encoder / granularity
    if granularity != "epoch":
        if pooling is None:
            raise ValueError(f"pooling is part of the cache key at {granularity!r} grain")
        d = d / pooling
    return d


def write_cache(
    encoder: str,
    granularity: str,
    embeddings: np.ndarray,
    index: pd.DataFrame,
    pooling: str | None = None,
    root: Path | str | None = None,
) -> Path:
    d = cache_dir(encoder, granularity, pooling, root)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    index.reset_index(drop=True).to_parquet(d / "index.parquet", index=False)
    return d


def read_cache(
    encoder: str, granularity: str, pooling: str | None = None, root: Path | str | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    d = cache_dir(encoder, granularity, pooling, root)
    return np.load(d / "embeddings.npy"), pd.read_parquet(d / "index.parquet")


def write_manifest(encoder: str, manifest: dict, root: Path | str | None = None) -> Path:
    d = _root(root) / encoder
    d.mkdir(parents=True, exist_ok=True)
    path = d / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def read_manifest(encoder: str, root: Path | str | None = None) -> dict:
    return json.loads((_root(root) / encoder / "manifest.json").read_text())


def list_encoders(root: Path | str | None = None) -> list[str]:
    r = _root(root)
    if not r.exists():
        return []
    return sorted(p.name for p in r.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def available_granularities(encoder: str, root: Path | str | None = None) -> list[str]:
    base = _root(root) / encoder
    return [g for g in ("epoch", "record", "subject") if (base / g).is_dir()]


def available_poolings(encoder: str, granularity: str, root: Path | str | None = None) -> list[str]:
    if granularity == "epoch":
        return []
    d = _root(root) / encoder / granularity
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "embeddings.npy").exists())
