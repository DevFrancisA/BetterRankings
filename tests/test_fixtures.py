"""The committed fixtures must match what the current Python code produces.

The TypeScript tests check the browser port against the same files, so this
keeps both implementations tied to one source of truth.
"""

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("make_fixtures", ROOT / "scripts" / "make_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _close(a, b, path="$"):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), path
    else:
        assert a == b, path


def _same_log(path: Path, committed: bytes) -> None:
    """Ids and clicks must match exactly; floats to 1e-12 (math libraries differ by an ulp across platforms)."""
    fresh = pd.read_csv(path, float_precision="round_trip")
    old = pd.read_csv(io.BytesIO(committed), float_precision="round_trip")
    assert list(fresh.columns) == list(old.columns), path.name
    for col in fresh.columns:
        if pd.api.types.is_float_dtype(old[col]):
            np.testing.assert_allclose(fresh[col], old[col], rtol=1e-12, atol=1e-14, err_msg=f"{path.name}:{col}")
        else:
            assert fresh[col].tolist() == old[col].tolist(), f"{path.name}:{col} changed; run scripts/make_fixtures.py"


def test_fixtures_are_up_to_date():
    mod = _load_script()
    committed = json.loads((ROOT / "fixtures" / "expected.json").read_text(encoding="utf-8"))
    logs = {p: p.read_bytes() for p in (ROOT / "fixtures").glob("*.csv")}
    logs[mod.SAMPLE] = mod.SAMPLE.read_bytes()
    try:
        fresh = json.loads(json.dumps(mod.build()))
        for path, data in logs.items():
            _same_log(path, data)
    finally:  # leave the committed files exactly as they were
        for path, data in logs.items():
            path.write_bytes(data)
    _close(fresh, committed)
