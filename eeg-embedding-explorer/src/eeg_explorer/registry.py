"""Adapter registry — encoder facts parsed from ``weights/MANIFEST.md``.

The manifest (staged by hand) is the single source of truth for per-encoder facts (DESIGN §5,
BUILD_SPEC §1 / Gate 0.5). Facts are *read from the file*, never hardcoded, so the dummy adapter can
emit correct shapes and the pretraining-overlap flag has real data. No network access.

Manifest format: one ``## <encoder>`` section per encoder, then ``- key: value`` lines. Values may
carry a trailing ``# comment``, span multiple continuation lines (e.g. the montage description), or
be a list — either inline (``[TUEG v1.1, TUEG v1.2]``) or indented bullets (LaBraM's dataset list).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Repo root is two levels up from this file: src/eeg_explorer/registry.py -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = _REPO_ROOT / "weights" / "MANIFEST.md"

_SECTION_RE = re.compile(r"^##\s+(?P<name>\S+)\s*$")
_FIELD_RE = re.compile(r"^-\s+(?P<key>\w+):\s*(?P<value>.*)$")


@dataclass
class EncoderFacts:
    """Registry facts for one encoder. Facts drive Gates 0-2; provenance is staged for Gate 3."""

    name: str
    embedding_dim: int
    epoch_input_s: float
    patch_s: float | None
    sampling_rate_hz: int
    expected_montage: str  # short descriptor (first line of the manifest value)
    finest_granularity: str
    loss_type: str
    pretraining_datasets: tuple[str, ...]
    native_window_s: float  # backward-compatible alias for epoch_input_s
    provenance: dict[str, str] = field(default_factory=dict)


def _split_sections(text: str) -> dict[str, list[str]]:
    """Group manifest lines under their ``## <encoder>`` heading, preserving order."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            current = m.group("name").lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def _parse_fields(lines: list[str]) -> dict[str, str]:
    """Parse ``- key: value`` lines; continuation lines fold into the preceding field's raw value."""
    fields: dict[str, str] = {}
    key: str | None = None
    buf: list[str] = []
    for line in lines:
        m = _FIELD_RE.match(line)
        if m:
            if key is not None:
                fields[key] = "\n".join(buf).strip()
            key = m.group("key")
            buf = [m.group("value")]
        elif key is not None:
            buf.append(line)
    if key is not None:
        fields[key] = "\n".join(buf).strip()
    return fields


def _first_line(raw: str) -> str:
    return raw.splitlines()[0].strip() if raw.strip() else ""


def _strip_comment(raw: str) -> str:
    """First line only, with any trailing ``# ...`` markdown comment removed."""
    first = _first_line(raw)
    if "#" in first:
        first = first.split("#", 1)[0]
    return first.strip()


def _parse_int(raw: str) -> int:
    return int(_strip_comment(raw))


def _parse_opt_number(raw: str) -> float | None:
    val = _strip_comment(raw)
    if val.lower() in ("", "null", "none", "n/a"):
        return None
    num = float(val)
    return int(num) if num.is_integer() else num


def _parse_list(raw: str) -> tuple[str, ...]:
    """List values: inline ``[a, b]`` or indented ``- item`` bullets on following lines."""
    stripped = raw.strip()
    inline = _first_line(raw)
    if inline.startswith("[") and inline.endswith("]"):
        inner = inline[1:-1]
        return tuple(x.strip() for x in inner.split(",") if x.strip())
    items = [
        line.strip()[2:].strip() for line in stripped.splitlines() if line.strip().startswith("- ")
    ]
    return tuple(items)


_PROVENANCE_KEYS = (
    "windowing",
    "windowing_note",
    "source_url",
    "version",
    "file_format",
    "local_path",
    "encoder_local_path",
    "contextualizer_local_path",
    "load_quirk",
)


def _build_facts(name: str, fields: dict[str, str]) -> EncoderFacts:
    epoch_input_s = _parse_opt_number(fields["epoch_input_s"])
    native_window_s = _parse_opt_number(fields.get("native_window_s", fields["epoch_input_s"]))
    provenance = {k: fields[k] for k in _PROVENANCE_KEYS if k in fields}
    # Keep the full multi-line montage text for Gate-3 use; expose a short descriptor here.
    provenance["expected_montage_detail"] = fields["expected_montage"].strip()
    return EncoderFacts(
        name=name,
        embedding_dim=_parse_int(fields["embedding_dim"]),
        epoch_input_s=epoch_input_s,
        patch_s=_parse_opt_number(fields.get("patch_s", "null")),
        sampling_rate_hz=_parse_int(fields["sampling_rate_hz"]),
        expected_montage=_first_line(fields["expected_montage"]),
        finest_granularity=_strip_comment(fields["finest_granularity"]),
        loss_type=_strip_comment(fields["loss_type"]),
        pretraining_datasets=_parse_list(fields["pretraining_datasets"]),
        native_window_s=native_window_s,
        provenance=provenance,
    )


def pretraining_overlap(viz_dataset: str, pretraining_datasets) -> bool:
    """Honesty guardrail (DESIGN §4): does the visualization dataset appear in the encoder's
    pretraining corpus? If so, apparent in-distribution separation may reflect memorization.

    Case-insensitive substring match either direction (e.g. ``"SHHS"`` vs ``"SHHS-1"``)."""
    v = viz_dataset.strip().lower()
    if not v:
        return False
    return any(v in d.lower() or d.lower() in v for d in pretraining_datasets)


def load_registry(manifest_path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, EncoderFacts]:
    """Parse the manifest into ``{encoder_name: EncoderFacts}``. Reads the file only; no network."""
    path = Path(manifest_path)
    text = path.read_text()
    registry: dict[str, EncoderFacts] = {}
    for name, lines in _split_sections(text).items():
        fields = _parse_fields(lines)
        # A real encoder section has facts; skip prose-only sections (none expected, but be safe).
        if "embedding_dim" not in fields:
            continue
        registry[name] = _build_facts(name, fields)
    return registry
