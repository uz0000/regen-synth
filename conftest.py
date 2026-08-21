"""Test-session setup: import path, and the demo fixture the suite depends on.

The README tells a reader to clone, install, and run ``pytest tests/ -q``. That
has to work on a fresh clone, and ``examples/transactions.csv`` — which eleven
test modules ingest — is a *generated* artifact and therefore gitignored. So the
suite generates it here if it is absent, deterministically, from the same
function ``examples/make_sample_data.py`` uses.

Without this the suite passes on a machine that has run a demo and fails on a
clean checkout, which is precisely the failure CI exists to catch, and did.

(The GPy/paramz DeprecationWarning noise is filtered via pyproject.toml's
[tool.pytest.ini_options] filterwarnings — the canonical pytest place.)
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "examples" / "transactions.csv"


@pytest.fixture(scope="session", autouse=True)
def _demo_sample_data():
    """Ensure examples/transactions.csv exists before any test ingests it.

    Seeded, so the file a fresh clone generates is byte-identical to the one a
    developer already has, and a test asserting a numeric threshold on it cannot
    depend on which machine produced it.
    """
    if not SAMPLE.exists():
        from examples.make_sample_data import make_dataset

        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        make_dataset().to_csv(SAMPLE, index=False)
    return SAMPLE
