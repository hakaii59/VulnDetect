"""Tests for the CI scan gate (scripts/scan_samples.py).

The workflow fails a build when a scanned file gets the "vulnerable" verdict.
These tests lock that behaviour down without needing the ML model or network
by injecting a fake `analyze` into the script's module namespace.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


@pytest.fixture()
def scanner(monkeypatch):
    """scripts/scan_samples.py with Layers 1-3 stubbed out."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    mod = importlib.import_module("scan_samples")

    def fake_analyze(code: str, model_threshold: float = 0.5) -> SimpleNamespace:
        vulnerable = "strcpy" in code and "// BAD" in code
        return SimpleNamespace(
            verdict="vulnerable" if vulnerable else "likely_safe",
            reason="stubbed",
            layers_run=["regex", "ast"],
            regex_findings=[],
            confirmed_findings=[SimpleNamespace(rule_id="R002")] if vulnerable else [],
            model_prediction=None,
        )

    monkeypatch.setattr(mod, "analyze", fake_analyze)
    monkeypatch.setattr(mod, "model_available", lambda: False)
    return mod


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


BAD_C = '// BAD\nvoid f(){ char b[4]; strcpy(b, "x"); }\n'
GOOD_C = "void f(){ int x = 1; }\n"


class TestGate:
    def test_fails_on_vulnerable(self, tmp_path, scanner):
        write(tmp_path / "bad.c", BAD_C)
        assert scanner.main(["scan", str(tmp_path)]) == 1

    def test_passes_on_safe(self, tmp_path, scanner):
        write(tmp_path / "good.c", GOOD_C)
        assert scanner.main(["scan", str(tmp_path)]) == 0

    def test_passes_when_there_is_nothing_to_scan(self, tmp_path, scanner):
        """A PR touching no C/C++ file must not fail the build."""
        write(tmp_path / "notes.md", "# not code\n")
        assert scanner.main(["scan", str(tmp_path)]) == 0

    def test_accepts_explicit_file_targets(self, tmp_path, scanner):
        """CI passes the PR's changed files one by one, not a directory."""
        good = write(tmp_path / "good.c", GOOD_C)
        bad = write(tmp_path / "bad.c", BAD_C)
        assert scanner.main(["scan", str(good)]) == 0
        assert scanner.main(["scan", str(good), str(bad)]) == 1

    def test_skips_deleted_targets(self, tmp_path, scanner, capsys):
        """Deleted files appear in a PR diff; they must not crash the gate."""
        write(tmp_path / "good.c", GOOD_C)
        code = scanner.main(["scan", str(tmp_path / "good.c"), str(tmp_path / "gone.c")])
        assert code == 0
        assert "Skipping missing target" in capsys.readouterr().err

    def test_unknown_command_is_an_error(self, scanner):
        assert scanner.main(["frobnicate"]) == 2


class TestExport:
    def test_output_is_valid_json(self, tmp_path, scanner, capsys):
        write(tmp_path / "bad.c", BAD_C)
        assert scanner.main(["export", str(tmp_path)]) == 0
        payload = json.loads(capsys.readouterr().out)
        (entry,) = payload.values()
        assert entry["verdict"] == "vulnerable"
        assert entry["confirmed"] == ["R002"]

    def test_empty_target_is_still_valid_json(self, tmp_path, scanner, capsys):
        assert scanner.main(["export", str(tmp_path)]) == 0
        assert json.loads(capsys.readouterr().out) == {}


class TestSelfTest:
    def test_real_fixtures_match_expectations(self):
        """No stubbing: runs the real pipeline over samples/.

        This is the check that would have caught the gate being unfireable —
        sample_vulnerable.c must actually reach the "vulnerable" verdict.
        """
        import scan_samples  # noqa: F401  (path set up by the scanner fixture / conftest)

        importlib.reload(scan_samples)
        assert scan_samples.cmd_selftest() == 0

    def test_expectations_file_covers_every_fixture(self):
        from src.utils import iter_source_files

        expected = json.loads((ROOT / "samples" / "expected.json").read_text(encoding="utf-8"))
        on_disk = {p.name for p in iter_source_files([ROOT / "samples"])}
        assert on_disk == set(expected), "samples/expected.json is out of sync with samples/"
