"""
Information-protection tests for the system itself (G-F), distinct from the
generated-data privacy layer (test_privacy.py).

Covers two invariants:
  1. No real data *values* leak into logs (or, by extension, error strings) —
     manifests/logs carry statistics, schemas, and counts, never rows. A
     generation run at DEBUG level over a fixture with planted sentinel values
     must not echo any sentinel back through the logging system.
  2. No secrets in tracked files — a scan of everything git tracks finds no
     API-key / access-key patterns. (`.env` is gitignored and is not scanned;
     the engine is model-free and needs no keys.)

Both use synthetic fixtures only (G-F rule 5): no real dataset rows in tests.
"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Distinctive markers that would only appear if a real cell value were logged.
SENTINEL_NUM = 8675309.4242          # a continuous cell value
SENTINEL_CAT = "ZZ_SENTINEL_CATEGORY_ZZ"  # a categorical cell value


class TestNoValueLeakInLogs:
    def _fixture_csv(self, tmp):
        rng = np.random.default_rng(0)
        n = 400
        amt = rng.normal(100, 20, n)
        amt[:20] = SENTINEL_NUM                       # plant the numeric sentinel
        note = rng.choice(["a", "b", "c"], size=n).astype(object)
        note[:20] = SENTINEL_CAT                      # plant the categorical sentinel
        note[300] = None                              # force an imputation log path
        y = rng.choice([0, 1], size=n, p=[0.85, 0.15])
        df = pd.DataFrame({"amt": amt, "note": note, "y": y})
        p = str(Path(tmp) / "sentinel.csv")
        df.to_csv(p, index=False)
        return p

    def test_generation_at_debug_does_not_log_cell_values(self, caplog):
        from regen.api import generate
        with tempfile.TemporaryDirectory() as tmp:
            path = self._fixture_csv(tmp)
            with caplog.at_level(logging.DEBUG):   # capture engine + api loggers
                generate(path, label_col="y", n_rows=150, seed=1,
                         privacy="floored", auto=False, out_dir=tmp)
        logtext = "\n".join(r.getMessage() for r in caplog.records)
        assert str(int(SENTINEL_NUM)) not in logtext   # "8675309"
        assert "8675309" not in logtext
        assert SENTINEL_CAT not in logtext


class TestNoSecretsInTrackedFiles:
    # Patterns split so this test file itself never contains a literal match.
    SECRET_PATTERNS = [
        re.compile(r"sk-" + r"[A-Za-z0-9_\-]{20,}"),        # OpenAI-style
        re.compile(r"tabpfn_" + r"sk_[A-Za-z0-9_\-]{10,}"),  # the removed dead key
        re.compile(r"AKIA" + r"[0-9A-Z]{16}"),               # AWS access key id
        # long hex/base64 assigned to a secret-named variable
        re.compile(r"(?i)(secret|token|password|api[_-]?key)\s*[:=]\s*"
                   r"['\"][A-Za-z0-9/+_\-]{24,}['\"]"),
    ]
    SKIP_EXT = {".parquet", ".pdf", ".png", ".ico", ".jpg", ".jpeg", ".gz",
                ".pyc", ".so"}

    def _tracked_files(self):
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True,
            check=True,
        )
        return [l for l in out.stdout.splitlines() if l.strip()]

    def test_no_secret_patterns_in_tracked_files(self):
        offenders = []
        for rel in self._tracked_files():
            if Path(rel).suffix.lower() in self.SKIP_EXT:
                continue
            fp = REPO_ROOT / rel
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeError):
                continue
            for pat in self.SECRET_PATTERNS:
                if pat.search(text):
                    offenders.append((rel, pat.pattern))
        assert not offenders, f"possible secrets in tracked files: {offenders}"
