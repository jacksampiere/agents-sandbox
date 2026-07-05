"""Labels — generic long-format ``(id, granularity, attribute_name, value)`` and the broadcast rule.

Long-format (not fixed dashboard columns) so the store can hold arbitrary outcomes later, per the
forward-compat note in DESIGN §6. Every label has a native granularity; it may color points at that
grain or **broadcast down to any finer grain** (every epoch inherits its subject's age). It may not
be applied at a coarser grain without aggregation. The point set is fixed by the chosen grain; the
label just recolors it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_COLUMNS = ["id", "granularity", "attribute_name", "value"]

# Finer granularity → higher rank. A label broadcasts down to any grain of >= its native rank.
GRAN_RANK = {"subject": 0, "record": 1, "epoch": 2}
# Column in an index frame holding the id at each native granularity.
_ID_COL = {"subject": "subject_id", "record": "record_id", "epoch": "epoch_id"}


def write_labels(labels: pd.DataFrame, path: Path | str) -> None:
    labels[LABEL_COLUMNS].to_parquet(path, index=False)


def read_labels(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def native_granularity(labels: pd.DataFrame, attribute: str) -> str:
    grans = labels.loc[labels["attribute_name"] == attribute, "granularity"].unique()
    if len(grans) != 1:
        raise ValueError(f"attribute {attribute!r} has {len(grans)} native granularities: {grans}")
    return str(grans[0])


def available_labels_for_granularity(labels: pd.DataFrame, target: str) -> list[str]:
    """Labels that can color points at ``target`` grain: native rank <= target rank (broadcast down)."""
    t = GRAN_RANK[target]
    attrs = [
        attr
        for attr in labels["attribute_name"].unique()
        if GRAN_RANK[native_granularity(labels, attr)] <= t
    ]
    return sorted(attrs)


def broadcast_labels(labels: pd.DataFrame, index: pd.DataFrame, attribute: str) -> pd.Series:
    """Return the label value for each row of ``index``, broadcasting from the label's native grain.

    ``index`` must carry the native-grain id column (e.g. ``subject_id`` for a subject label).
    Raises if asked to apply a label at a grain coarser than its native one (no aggregation here).
    """
    native = native_granularity(labels, attribute)
    id_col = _ID_COL[native]
    if id_col not in index.columns:
        raise ValueError(
            f"cannot broadcast {native}-level {attribute!r} onto an index lacking {id_col!r} "
            f"(applying a label at a coarser grain requires aggregation)"
        )
    sub = labels.loc[labels["attribute_name"] == attribute, ["id", "value"]]
    merged = index.reset_index(drop=True).merge(
        sub, left_on=id_col, right_on="id", how="left", validate="many_to_one"
    )
    return merged["value"]
