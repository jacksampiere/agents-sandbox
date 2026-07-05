"""Put ``src/`` on ``sys.path`` so tests can ``import eeg_explorer``.

The project uses a ``src/`` layout with ``[tool.uv] package = false`` (code is not installed).
A root conftest keeps the fixed ``pyproject.toml`` untouched while making the package importable
under pytest. Streamlit's ``streamlit run src/app.py`` gets ``src/`` on the path as the script dir.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
