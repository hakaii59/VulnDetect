"""Scan C/C++ files with the full 3-layer pipeline and gate CI.

Commands
--------
  scan     [TARGET ...]  run the pipeline over files/directories, print a
                         report, and exit 1 if any file is flagged
                         "vulnerable". This is the CI gate.
  export   [TARGET ...]  print a JSON summary (path -> verdict) for the
                         report step / PR artifact.
  list     [TARGET ...]  just list the files that would be scanned.
  selftest               run the pipeline over samples/ and assert each file
                         gets the verdict recorded in samples/expected.json.
                         Proves the gate can actually discriminate — a gate
                         that never fires is worse than no gate.

A TARGET is a file or a directory (directories are walked recursively).
With no TARGET, `scan`/`export`/`list` default to samples/.

Usage:
  python scripts/scan_samples.py scan src/ vendor/foo.c
  python scripts/scan_samples.py export samples
  python scripts/scan_samples.py selftest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python scripts/scan_samples.py` to import src.*

from src.pipeline import analyze, model_available
from src.utils import iter_source_files, read_source, relative_to_root

DEFAULT_TARGETS = [ROOT / "samples"]
EXPECTED_PATH = ROOT / "samples" / "expected.json"

#: Verdict that fails the build.
FAILING_VERDICT = "vulnerable"


def resolve_targets(args: list[str]) -> tuple[list[Path], list[str]]:
    """Turn CLI args into existing paths, reporting the ones that are missing.

    Deleted files can appear in a PR's changed-file list, so a missing target
    is skipped with a note rather than failing the gate.
    """
    targets: list[Path] = []
    missing: list[str] = []
    for arg in args:
        path = Path(arg)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            targets.append(path)
        else:
            missing.append(arg)
    return targets, missing


def scan_targets(targets: list[Path]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for path in iter_source_files(targets):
        key = relative_to_root(path)
        try:
            code = read_source(path)
        except OSError as exc:
            results[key] = {"error": str(exc)}
            continue
        res = analyze(code)
        results[key] = {
            "verdict": res.verdict,
            "reason": res.reason,
            "layers": res.layers_run,
            "confirmed": sorted({f.rule_id for f in res.confirmed_findings}),
            "regex_hits": len(res.regex_findings),
            "model_label": None if res.model_prediction is None else res.model_prediction.label,
            "model_probability": (
                None if res.model_prediction is None else round(res.model_prediction.probability, 4)
            ),
            "model_source": None if res.model_prediction is None else res.model_prediction.model_source,
        }
    return results


def print_report(results: dict[str, dict]) -> int:
    """Print the human-readable report; return the number of flagged files."""
    flagged = 0
    for path, info in sorted(results.items()):
        if "error" in info:
            print(f"[ERROR       ] {path}: {info['error']}")
            continue
        prob = info["model_probability"]
        model_part = "model=n/a" if prob is None else f"model={info['model_label']} P={prob}"
        print(f"[{info['verdict'].upper():<12}] {path}  (confirmed={info['confirmed']} {model_part})")
        print(f"{'':15}{info['reason']}")
        if info["verdict"] == FAILING_VERDICT:
            flagged += 1
    return flagged


def cmd_selftest() -> int:
    """Assert the pipeline reproduces the expected verdict for each fixture."""
    if not EXPECTED_PATH.exists():
        print(f"Missing expectations file: {EXPECTED_PATH}", file=sys.stderr)
        return 2

    expected: dict[str, str] = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    results = scan_targets([ROOT / "samples"])

    print(f"Layer 3 model available: {model_available()}")
    print(f"Checking {len(expected)} fixture(s) against samples/expected.json\n")

    failures = 0
    for name, want in sorted(expected.items()):
        key = f"samples/{name}"
        info = results.get(key)
        if info is None:
            print(f"[MISSING] {key}: fixture not found or not scanned")
            failures += 1
            continue
        got = info.get("verdict")
        if got == want:
            print(f"[OK]   {key}: {got}")
        else:
            print(f"[FAIL] {key}: expected '{want}', got '{got}' ({info.get('reason')})")
            failures += 1

    print()
    if failures:
        print(f"Self-test FAILED: {failures} fixture(s) did not match.")
        return 1
    print("Self-test PASSED: the gate discriminates vulnerable from safe fixtures.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "scan"
    rest = argv[1:]

    if cmd == "selftest":
        return cmd_selftest()

    if cmd not in {"scan", "export", "list"}:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        return 2

    if rest:
        targets, missing = resolve_targets(rest)
        for m in missing:
            print(f"Skipping missing target: {m}", file=sys.stderr)
    else:
        targets, missing = list(DEFAULT_TARGETS), []

    files = list(iter_source_files(targets))
    if cmd == "list":
        for f in files:
            print(relative_to_root(f))
        return 0

    if not files:
        if cmd == "export":
            print("{}")
        else:
            print("No C/C++ files to scan — nothing to do.")
        return 0

    results = scan_targets(targets)

    if cmd == "export":
        print(json.dumps(results, indent=2, default=str))
        return 0

    flagged = print_report(results)
    print(f"\nScanned {len(results)} file(s); {flagged} flagged '{FAILING_VERDICT}'.")
    if not model_available():
        print("Note: Layer 3 model not present — verdicts rest on regex + AST only.")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
