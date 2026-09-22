"""Shared helpers for the VulnDetect-CPP pipeline.

Small, dependency-free utilities used by the scan scripts, the notebooks and
the tests, kept here so the file-walking / normalization rules are defined
exactly once.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Iterator

#: Repo root (this file lives at <root>/src/utils.py).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Extensions treated as C/C++ translation units or headers.
C_EXTENSIONS = frozenset({".c", ".cc", ".cpp", ".cxx", ".c++", ".h", ".hh", ".hpp", ".hxx"})

#: Directories never scanned — virtualenvs, VCS metadata and build output are
#: either third-party code or generated, so findings in them are noise.
EXCLUDED_DIRS = frozenset(
    {".venv", "venv", ".git", ".github", "node_modules", "__pycache__", "build", "dist", ".pytest_cache"}
)

_WHITESPACE_RE = re.compile(r"\s+")


def is_source_file(path: Path) -> bool:
    """True for a C/C++ file that is not inside an excluded directory."""
    if path.suffix.lower() not in C_EXTENSIONS:
        return False
    return not any(part in EXCLUDED_DIRS for part in path.parts)


def iter_source_files(targets: Iterable[Path | str]) -> Iterator[Path]:
    """Yield unique C/C++ files from a mix of file and directory targets.

    Directories are walked recursively; explicit file targets are yielded even
    when nested in an excluded directory only if they still pass the extension
    check, so `scan path/to/one.c` always does what the caller asked.
    """
    seen: set[Path] = set()
    for target in targets:
        target = Path(target)
        candidates = sorted(target.rglob("*")) if target.is_dir() else [target]
        for path in candidates:
            if not path.is_file() or not is_source_file(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def read_source(path: Path | str) -> str:
    """Read a source file as UTF-8, replacing undecodable bytes.

    Real-world C files are not reliably UTF-8 (Latin-1 identifiers, stray bytes
    in string literals). Replacing keeps byte offsets sane instead of crashing
    a whole scan on one bad file.
    """
    return Path(path).read_text(encoding="utf-8", errors="replace")


def normalize_code(code: str) -> str:
    """Collapse all whitespace so formatting-only differences compare equal.

    Used for near-duplicate detection: Big-Vul contains the same function
    reindented across commits, which exact-string dedup misses.
    """
    return _WHITESPACE_RE.sub(" ", str(code)).strip()


def code_fingerprint(code: str) -> str:
    """Stable hash of the normalized code, for cross-split leakage checks."""
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()


def relative_to_root(path: Path | str) -> str:
    """Repo-relative POSIX path when possible, else the path as given.

    Keeps CI logs and JSON reports identical on Windows and Linux runners.
    """
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
