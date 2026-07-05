"""Gate 0.5 — the registry parses encoder facts from weights/MANIFEST.md correctly."""

import pytest

from eeg_explorer.registry import EncoderFacts, load_registry


@pytest.fixture(scope="module")
def registry() -> dict[str, EncoderFacts]:
    return load_registry()


def test_three_encoders_present(registry):
    assert set(registry) == {"cbramod", "bendr", "labram"}


def test_embedding_dims(registry):
    assert registry["cbramod"].embedding_dim == 200
    assert registry["bendr"].embedding_dim == 512
    assert registry["labram"].embedding_dim == 200


def test_epoch_input_is_30s_and_alias_matches(registry):
    for facts in registry.values():
        assert facts.epoch_input_s == 30
        # native_window_s is a backward-compatible alias; it must agree with epoch_input_s.
        assert facts.native_window_s == facts.epoch_input_s


def test_patch_and_sampling_rate(registry):
    assert registry["cbramod"].patch_s == 1
    assert registry["labram"].patch_s == 1
    assert registry["bendr"].patch_s is None  # manifest: patch_s: null
    assert registry["cbramod"].sampling_rate_hz == 200
    assert registry["labram"].sampling_rate_hz == 200
    assert registry["bendr"].sampling_rate_hz == 256


def test_finest_granularity_epoch(registry):
    for facts in registry.values():
        assert facts.finest_granularity == "epoch"


def test_loss_types(registry):
    assert registry["cbramod"].loss_type == "masked-reconstruction"
    assert registry["bendr"].loss_type == "contrastive"
    assert "masked" in registry["labram"].loss_type


def test_pretraining_datasets(registry):
    assert registry["cbramod"].pretraining_datasets == ("TUEG",)
    assert registry["bendr"].pretraining_datasets == ("TUEG v1.1", "TUEG v1.2")
    # LaBraM lists a large multi-dataset corpus as indented bullets.
    assert len(registry["labram"].pretraining_datasets) > 10
    assert "TUAR" in registry["labram"].pretraining_datasets
    for facts in registry.values():
        assert facts.pretraining_datasets  # non-empty for all


def test_provenance_captured(registry):
    for facts in registry.values():
        assert facts.provenance.get("source_url", "").startswith("http")
        assert facts.expected_montage  # short descriptor present
