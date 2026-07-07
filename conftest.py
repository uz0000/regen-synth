"""Add repo root to sys.path so engine/, contracts/, regen/ are importable.

(The GPy/paramz DeprecationWarning noise is filtered via pyproject.toml's
[tool.pytest.ini_options] filterwarnings — the canonical pytest place.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
