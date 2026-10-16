"""The committed fixtures must match what the current Python code produces.

The TypeScript tests check the browser port against the same files, so this
keeps both implementations tied to one source of truth.
"""

import importlib.util
import json
from pathlib import Path

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


def test_fixtures_are_up_to_date(tmp_path, monkeypatch):
    mod = _load_script()
    committed = json.loads((ROOT / "fixtures" / "expected.json").read_text(encoding="utf-8"))
    logs = {p.name: p.read_bytes() for p in (ROOT / "fixtures").glob("*.csv")}
    sample = mod.SAMPLE.read_bytes()
    try:
        fresh = json.loads(json.dumps(mod.build()))
        for name, data in logs.items():
            assert (ROOT / "fixtures" / name).read_bytes() == data, f"{name} changed; run scripts/make_fixtures.py"
        assert mod.SAMPLE.read_bytes() == sample, "sample-log.csv changed; run scripts/make_fixtures.py"
    finally:  # leave the committed files exactly as they were
        for name, data in logs.items():
            (ROOT / "fixtures" / name).write_bytes(data)
        mod.SAMPLE.write_bytes(sample)
    _close(fresh, committed)
