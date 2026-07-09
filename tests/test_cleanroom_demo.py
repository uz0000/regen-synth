"""
Smoke test for the certified-surrogate clean-room demo (§5.5) — guards the
showcase script from bit-rot. The underlying pieces (generate / measure_tstr /
verify_bundle) have their own tests; this exercises the end-to-end orchestration.
"""

import numpy as np
import pandas as pd


def test_cleanroom_demo_runs(tmp_path, capsys):
    from examples.certified_surrogate_demo import run
    rng = np.random.default_rng(0)
    n_norm, n_rare = 400, 120
    df = pd.concat([
        pd.DataFrame({"a": rng.normal(0, 1, n_norm), "b": rng.normal(0, 1, n_norm), "y": 0}),
        pd.DataFrame({"a": rng.normal(3, 1, n_rare), "b": rng.normal(3, 1, n_rare), "y": 1}),
    ], ignore_index=True)
    path = str(tmp_path / "f.csv")
    df.to_csv(path, index=False)

    assert run(path, "y", rare_value=1, seed=1) == 0
    out = capsys.readouterr().out
    # the three clean-room roles + the two headline artifacts appear
    assert "PRODUCER" in out and "CONSUMER" in out and "AUDITOR" in out
    assert "VERIFIED" in out and "TSTR" in out
