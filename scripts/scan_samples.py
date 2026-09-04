"""Scan a folder of C/C++ files with the full 3-layer pipeline and gate CI.

Two commands (subcommands):
  scan     [default] run every *.c/*.cc/*.cpp/*.cxx/*.h/*.hpp under <dir>,
            print a per-file report, and exit 1 if any file is flagged
            "vulnerable" (used by the CI workflow to FAIL a PR that ships a
            match of the vulnerability rules).
  export   print a compact JSON summary (file -> verdict) for the report step.

Usage:
  python scripts/scan_samples.py scan [DIR]          # gate
  python scripts/scan_samples.py export [DIR]        # summary
  python scripts/scan_samples.py list [DIR]          # just list files
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python scripts/scan_samples.py` to import src.*

from src.pipeline import analyze

DEFAULT_DIR = ROOT / "samples"
C_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"}

# Files that must never be treated as user code (they are part of the test
# harness itself or vendored build output).
EXCLUDED_DIRS = {".venv", "venv", ".git", "node_modules", "__pycache__", "build", "dist"}


def iter_source_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in C_EXTENSIONS:
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def scan_dir(directory: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for path in iter_source_files(directory):
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            results[str(path)] = {"error": str(exc)}
            continue
        res = analyze(code)
        results[str(path)] = {
            "verdict": res.verdict,
            "confirmed": [f.rule_id for f in res.confirmed_findings],
            "model_label": res.model_prediction.label,
            "model_probability": round(res.model_prediction.probability, 4),
            "model_source": res.model_prediction.model_source,
        }
    return results


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "scan"
    directory = Path(argv[1]) if len(argv) > 1 else DEFAULT_DIR
    if not directory.is_absolute():
        directory = ROOT / directory
    if not directory.exists():
        print(f"Directory not found: {directory}", file=sys.stderr)
        return 2

    if cmd == "list":
        for f in iter_source_files(directory):
            print(f)
        return 0

    results = scan_dir(directory)

    if cmd == "export":
        print(json.dumps(results, indent=2, default=str))
        return 0

    # default: human-readable scan + gate
    flagged = 0
    for path, info in results.items():
        if "error" in info:
            print(f"[ERROR] {path}: {info['error']}")
            continue
        verdict = info["verdict"]
        print(
            f"[{verdict.upper():<12}] {path}  "
            f"(confirmed={info['confirmed']} model={info['model_label']} "
            f"P={info['model_probability']})"
        )
        if verdict == "vulnerable":
            flagged += 1

    print(f"\nScanned {len(results)} file(s); {flagged} flagged 'vulnerable'.")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
