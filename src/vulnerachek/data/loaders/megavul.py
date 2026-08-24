"""Loader for MegaVul (https://github.com/Icyrockton/MegaVul).

Expected input: ``megavul_simple.json`` / ``megavul.json`` (or any directory
containing such files, e.g. one per language release such as
``megavul_c_cpp.json`` + ``megavul_java.json``) as documented in
SPECIFICATION.md of the upstream repo. Each function entry is a flat dict
with (default column names, override via column_map if a release differs):

    func          -> source code of the function
    is_vul        -> bool/0/1 vulnerability label
    cwe_ids       -> list[str] | str of CWE ids
    commit_hash   -> commit id
    repo_name     -> "owner/repo"
    language      -> "c" | "cpp" | "java" (megavul.json only; absent in
                     megavul_simple.json, inferred from the file name instead)
    func_name     -> function name
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from vulnerachek.data.loaders.base import BaseLoader
from vulnerachek.data.schema import VulnRecord, normalize_cwe, normalize_language

logger = logging.getLogger(__name__)


class MegaVulLoader(BaseLoader):
    name = "megavul"

    def load(self) -> Iterator[VulnRecord]:
        self._require_path()
        for json_path in self._iter_json_files():
            yield from self._load_file(json_path)

    def _iter_json_files(self) -> Iterator[Path]:
        if self.path.is_dir():
            yield from sorted(self.path.glob("*.json"))
        else:
            yield self.path

    def _load_file(self, json_path: Path) -> Iterator[VulnRecord]:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[%s] could not read %s: %s", self.name, json_path, exc)
            return

        entries = data if isinstance(data, list) else list(data.values())
        fallback_language = self._guess_language_from_filename(json_path)

        for entry in entries:
            if not isinstance(entry, dict):
                self._skip(f"non-dict entry in {json_path.name}")
                continue
            record = self._parse_entry(entry, fallback_language, source_file=json_path.name)
            if record is not None:
                yield self._emit(record)

    @staticmethod
    def _guess_language_from_filename(json_path: Path) -> str:
        stem = json_path.stem.lower()
        if "java" in stem:
            return "java"
        if "cpp" in stem or "c_cpp" in stem:
            return "cpp"
        if stem.endswith("_c") or "_c_" in stem:
            return "c"
        return ""

    def _parse_entry(self, entry: dict, fallback_language: str, *, source_file: str) -> VulnRecord | None:
        code = str(self.col(entry, "code", self.col(entry, "func", "")) or "").strip()
        if not code:
            self._skip(f"empty code in {source_file}")
            return None

        is_vul = self.col(entry, "label", self.col(entry, "is_vul", None))
        if is_vul is None:
            self._skip(f"missing is_vul in {source_file}")
            return None
        label = 1 if str(is_vul).strip().lower() in {"1", "true", "yes"} else 0

        language = normalize_language(self.col(entry, "language", "") or fallback_language)
        if not language:
            self._skip(f"unrecognized language in {source_file}")
            return None

        cwe_raw = self.col(entry, "cwe_type", self.col(entry, "cwe_ids", ""))
        if isinstance(cwe_raw, list):
            cwe_raw = cwe_raw[0] if cwe_raw else ""
        cwe_type = normalize_cwe(cwe_raw)

        try:
            return VulnRecord(
                code=code,
                label=label,
                language=language,
                cwe_type=cwe_type,
                source=self.name,
                project=str(self.col(entry, "project", self.col(entry, "repo_name", "")) or ""),
                commit_id=str(self.col(entry, "commit_id", self.col(entry, "commit_hash", "")) or ""),
                func_name=str(self.col(entry, "func_name", "") or ""),
            )
        except ValueError as exc:
            self._skip(f"invalid record in {source_file}: {exc}")
            return None
