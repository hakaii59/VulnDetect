"""Unified record schema produced by every dataset loader.

Every loader (MegaVul, Big-Vul, CVEfixes, PrimeVul, ...) parses a different
raw format, but they all must emit ``VulnRecord`` instances. Downstream code
(merge, split, balance, train) only ever deals with this one shape, so a new
dataset can be added by writing one loader without touching anything else.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

LANGUAGES = {"c", "cpp", "java"}
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class VulnRecord:
    """One function-level (code, label) sample plus provenance metadata.

    ``project`` and ``commit_id`` are not part of the training schema itself
    (the model never sees them) but they are required to do commit/project
    level splitting without leaking near-duplicate functions across
    train/val/test — see split.py.
    """

    code: str
    label: int  # 1 = vulnerable, 0 = safe/patched
    language: str  # "c" | "cpp" | "java"
    cwe_type: str  # e.g. "CWE-119"; "" if unknown or not applicable
    source: str  # dataset name: "megavul" | "bigvul" | "cvefixes" | "primevul"
    project: str = ""  # repo/project identifier, used for group-aware split
    commit_id: str = ""  # commit hash, used for group-aware split
    func_name: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")
        if self.language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}, got {self.language!r}")
        if not self.code or not self.code.strip():
            raise ValueError("code must not be empty")
        if not self.source:
            raise ValueError("source must not be empty")

    @property
    def content_hash(self) -> str:
        """Hash of whitespace-normalized code, used for dedup and as a split fallback key."""
        normalized = _WHITESPACE_RE.sub(" ", self.code).strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @property
    def group_key(self) -> str:
        """Key that must never be split across train/val/test.

        Prefers project (functions from the same repo tend to share
        boilerplate/helpers -> near-duplicates), falls back to commit_id,
        and finally to the code hash itself so ungrouped records are at
        least self-consistent rather than silently grouped together.
        """
        return self.project or self.commit_id or self.content_hash

    def to_row(self) -> dict:
        """Dict form written to the output .jsonl files."""
        row = asdict(self)
        row["id"] = self.content_hash[:16]
        return row


def normalize_cwe(raw: str | int | None) -> str:
    """Normalize CWE identifiers to the canonical "CWE-<number>" form.

    Raw values across datasets show up as ints, "119", "CWE-119", lists
    joined with commas/semicolons, or "NVD-CWE-noinfo". Only the first
    numeric CWE id is kept since the classifier is trained on a single
    cwe_type per record; unrecognized/missing values become "".
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text or text.upper() in {"NAN", "NONE", "NVD-CWE-NOINFO", "NVD-CWE-OTHER"}:
        return ""
    match = re.search(r"(\d+)", text)
    return f"CWE-{match.group(1)}" if match else ""


def normalize_language(raw: str | None, *, filename: str = "") -> str:
    """Map free-form language/extension strings to one of LANGUAGES, or "" if unknown."""
    text = (raw or "").strip().lower()
    if text in LANGUAGES:
        return text
    if text in {"c++", "cxx"}:
        return "cpp"
    ext = filename.strip().lower().rsplit(".", 1)[-1] if "." in filename else ""
    ext_map = {"c": "c", "h": "c", "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "java": "java"}
    return ext_map.get(ext, "")
