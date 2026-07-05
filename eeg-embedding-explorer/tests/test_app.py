"""Gate 2 — drive the dashboard headlessly and assert every control functions end to end.

Uses Streamlit's AppTest to flip encoder / granularity / label / pooling / projection method +
params / 2D-3D and asserts the app runs without error each time — the honest "controls function"
check that a human clicking would otherwise have to do.
"""

from pathlib import Path

import pytest

from eeg_explorer.precompute import build_synthetic_caches
from eeg_explorer.synthetic import SyntheticConfig

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "app.py"
SMALL = SyntheticConfig(n_subjects=4, epochs_per_record=20)  # small → fast UMAP


@pytest.fixture(scope="module")
def app_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache")
    labels = root.parent / "labels.parquet"
    build_synthetic_caches(root=root, labels_out=labels, config=SMALL)
    return root, labels


@pytest.fixture
def at(app_root, monkeypatch):
    root, labels = app_root
    monkeypatch.setenv("EEG_CACHE_ROOT", str(root))
    monkeypatch.setenv("EEG_LABELS_PATH", str(labels))
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_initial_render_has_no_error(at):
    assert not at.exception
    assert not at.error  # caches exist → the "no caches" error branch is not taken
    assert any(m.label == "Points" for m in at.metric)  # point-count display
    assert at.info or at.warning  # pretraining-overlap flag rendered (info: no overlap)


def test_sweep_granularity_and_pooling(at):
    at.selectbox(key="method").set_value("PCA").run()  # fast path for the sweep
    at.selectbox(key="label").set_value("disease").run()  # valid at every grain
    for grain in ("epoch", "record", "subject"):
        at.selectbox(key="granularity").set_value(grain).run()
        assert not at.exception, f"granularity={grain}"
        if grain != "epoch":
            for pooling in ("mean", "max"):
                at.selectbox(key="pooling").set_value(pooling).run()
                assert not at.exception, f"{grain}/{pooling}"


def test_sweep_labels_at_epoch(at):
    at.selectbox(key="method").set_value("PCA").run()
    at.selectbox(key="granularity").set_value("epoch").run()
    for label in ("sleep_stage", "age", "sex", "arousal"):
        at.selectbox(key="label").set_value(label).run()
        assert not at.exception, f"label={label}"
        assert any(m.label == "Points" for m in at.metric)


def test_sweep_encoders(at):
    at.selectbox(key="method").set_value("PCA").run()
    for encoder in at.selectbox(key="encoder").options:
        at.selectbox(key="encoder").set_value(encoder).run()
        assert not at.exception, f"encoder={encoder}"


def test_3d_toggle(at):
    at.selectbox(key="method").set_value("PCA").run()
    at.radio(key="dims").set_value("3D").run()
    assert not at.exception


def test_umap_param_changes(at):
    at.selectbox(key="method").set_value("UMAP").run()
    at.slider(key="n_neighbors").set_value(30).run()
    at.slider(key="min_dist").set_value(0.5).run()
    assert not at.exception
