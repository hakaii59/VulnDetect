"""Loader for PrimeVul (https://github.com/DLVulDet/PrimeVul).

Expected input: one or more ``.jsonl`` files (the upstream release ships
``train.jsonl`` / ``valid.jsonl`` / ``test.jsonl``; point ``path`` at the
directory containing them, or at a single file). Each line is a JSON object;
commonly used field names (override via column_map if a release differs):

    func        -> function source code
    target      -> 1 (vulnerable) / 0 (benign)
    cwe         -> CWE id, str or list[str]
    project     -> GitHub repo name
    commit_id   -> commit hash
    file_name   -> used to disambiguate c vs cpp when no explicit language
                   field is present (PrimeVul is C/C++ only)

Note: PrimeVul's own splits already exist upstream, but we re-derive our own
commit/project-level split (see split.py) so all four datasets are combined
and split consistently -- otherwise a project present in PrimeVul-train could
still leak into our val/test via another dataset's copy of the same repo.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from vulnerachek.data.loaders.base import BaseLoader
from vulnerachek.data.schema import VulnRecord, normalize_cwe, normalize_language

logger = logging.getLogger(__name__)


class PrimeVulLoader(BaseLoader):
    name = "primevul"

    def load(self) -> Iterator[VulnRecord]:
        self._require_path()
        for jsonl_path in self._iter_files():
            yield from self._load_file(jsonl_path)

    def _iter_files(self) -> Iterator[Path]:
        if self.path.is_dir():
            yield from sorted(self.path.glob("*.jsonl"))
        else:
            yield self.path

    def _load_file(self, jsonl_path: Path) -> Iterator[VulnRecord]:
        try:
            f = open(jsonl_path, encoding="utf-8")
        except OSError as exc:
            logger.warning("[%s] could not open %s: %s", self.name, jsonl_path, exc)
            return

        with f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    self._skip(f"{jsonl_path.name}:{line_num}: invalid JSON")
                    continue
                record = self._parse_entry(entry, source_file=jsonl_path.name, line_num=line_num)
                if record is not None:
                    yield self._emit(record)

    def _parse_entry(self, entry: dict, *, source_file: str, line_num: int) -> VulnRecord | None:
        code = str(self.col(entry, "code", self.col(entry, "func", "")) or "").strip()
        if not code:
            self._skip(f"{source_file}:{line_num}: empty code")
            return None

        target = self.col(entry, "label", self.col(entry, "target", None))
        if target is None:
            self._skip(f"{source_file}:{line_num}: missing target")
            return None
        label = 1 if str(target).strip() in {"1", "True", "true"} else 0

        filename = str(self.col(entry, "file_name", "") or "")
        language = normalize_language(self.col(entry, "language", ""), filename=filename) or "cpp"

        cwe_raw = self.col(entry, "cwe_type", self.col(entry, "cwe", ""))
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
                project=str(self.col(entry, "project", "") or ""),
                commit_id=str(self.col(entry, "commit_id", "") or ""),
                func_name=str(self.col(entry, "func_name", "") or ""),
            )
        except ValueError as exc:
            self._skip(f"{source_file}:{line_num}: {exc}")
            return None
