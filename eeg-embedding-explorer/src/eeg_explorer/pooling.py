"""Rule-based pooling up the granularity hierarchy: epoch → record → subject (DESIGN §2).

Pooling is a user-selectable, cache-keyed toggle. This synthetic run enables ``mean`` (the DESIGN
default) and ``max`` so the pooling control is a genuine multi-option toggle end to end (BUILD_SPEC
Gate 2); CLS-token and mean+std pooling are deferred (they need per-patch model outputs — Gate 3+).
Intra-epoch pooling (sub-windows → one 30s vector) is an *adapter* detail and lives in adapters.py,
not here — epoch-grain embeddings arrive already pooled.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

POOLINGS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": lambda a: a.mean(axis=0),
    "max": lambda a: a.max(axis=0),
}
DEFAULT_POOLING = "mean"

# Column that identifies the group at each coarser grain.
_GROUP_COL = {"record": "record_id", "subject": "subject_id"}
# Columns kept in the pooled index frame per target grain.
_INDEX_COLS = {"record": ["record_id", "subject_id"], "subject": ["subject_id"]}


def pool_embeddings(
    epoch_embeddings: np.ndarray,
    epoch_index: pd.DataFrame,
    target_granularity: str,
    pooling: str = DEFAULT_POOLING,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Aggregate epoch-grain embeddings up to ``record`` or ``subject`` grain.

    Returns ``(pooled (M, D) float32, index frame)`` with rows aligned. Group order is first-seen.
    """
    if target_granularity == "epoch":
        raise ValueError("epoch is the base grain (intra-epoch pooling is an adapter detail)")
    if target_granularity not in _GROUP_COL:
        raise ValueError(f"unknown granularity: {target_granularity!r}")
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; available: {sorted(POOLINGS)}")

    group_col = _GROUP_COL[target_granularity]
    fn = POOLINGS[pooling]

    index = epoch_index.drop_duplicates(group_col, keep="first").reset_index(drop=True)
    groups = epoch_index[group_col].to_numpy()
    pooled = np.zeros((len(index), epoch_embeddings.shape[1]), dtype=np.float32)
    for i, g in enumerate(index[group_col].to_numpy()):
        pooled[i] = fn(epoch_embeddings[groups == g])
    return pooled, index[_INDEX_COLS[target_granularity]].reset_index(drop=True)
