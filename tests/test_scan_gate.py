"""Unit tests for the CI scan gate (scripts/scan_samples.py).

The CI workflow (`vuln_check.yml`) exits non-zero when a scanned file is
flagged "vulnerable". These tests lock down that behaviour without needing the
ML model or network: we inject a fake `analyze` into the module namespace.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


@pytest.fixture()
def scanner(monkeypatch):
    """Import scripts/scan_samples.py with a fake analyze() so no model is needed."""
    sys.path.insert(0, str(SCRIPTS))  # scripts/scan_samples.py imports src.* via ROOT insert itself
    mod = importlib.import_module("scan_samples")
    monkeypatch.setattr(sys, "path", [str(SCRIPTS)] + [p for p in sys.path if p != str(SCRIPTS)])
    # Re-point module globals to the real repo paths if the import used defaults.
    mod.ROOT = ROOT
    mod.DEFAULT_DIR = ROOT / "samples"

    def fake_analyze(code: str) -> SimpleNamespace:
        if "strcpy" in code and "// BAD" in code:
            return SimpleNamespace(
                verdict="vulnerable",
                confirmed_findings=[SimpleNamespace(rule_id="R002")],
                model_prediction=SimpleNamespace(label=1, probability=0.97, model_source="fake"),
            )
        return SimpleNamespace(
            verdict="likely_safe",
            confirmed_findings=[],
            model_prediction=SimpleNamespace(label=0, probability=0.001, model_source="fake"),
        )

    monkeypatch.setattr(mod, "analyze", fake_analyze)
    return mod


def test_gate_fails_on_vulnerable(tmp_path, scanner, monkeypatch):
    (tmp_path / "bad.c").write_text("// BAD\nvoid f(){ char b[4]; strcpy(b, \"x\"); }\n", encoding="utf-8")
    monkeypatch.chdir(ROOT)  # scanner resolves ROOT-relative dirs via Path(args)
    code = scanner.main(["scan", str(tmp_path)])
    assert code == 1


def test_gate_passes_on_safe(tmp_path, scanner, monkeypatch):
    (tmp_path / "good.c").write_text("void f(){ int x = 1; }\n", encoding="utf-8")
    code = scanner.main(["scan", str(tmp_path)])
    assert code == 0


def test_export_json_is_serializable(tmp_path, scanner):
    (tmp_path / "good.c").write_text("int main(){return 0;}\n", encoding="utf-8")
    import json

    out = scanner.main(["export", str(tmp_path)])
    assert out == 0
    # capture the printed json via capsys by calling scan_dir directly
    res = scanner.scan_dir(tmp_path)
    json.dumps(res)  # must not raise
    assert "good.c" in str(list(res.keys())[0])
