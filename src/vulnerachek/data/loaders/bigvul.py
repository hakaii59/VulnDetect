"""Loader for Big-Vul / MSR_data_cleaned.csv
(https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset).

Expected input: the cleaned CSV (commonly named ``MSR_data_cleaned.csv``)
linked from the upstream README. Relevant columns (override names via
column_map if a release differs):

    func_before  -> function source before the fix (the vulnerable version)
    func_after   -> function source after the fix (the patched version)
    vul          -> 1 if this row is a vulnerability-fixing commit, else 0
    CWE ID       -> e.g. "CWE-119"
    project      -> GitHub repo name, used for group-aware splitting
    commit_id    -> fixing commit hash
    lang         -> "C" | "C++"

Big-Vul stores one row per (vulnerable function, its own fix), not one row
per label. So each CSV row is expanded into up to two VulnRecords:

    vul == 1: func_before -> label 1 (vulnerable), func_after -> label 0
              (the patched version of that *same* function -- a hard
              negative that teaches the model the fix, not just "any code")
    vul == 0: func_before only -> label 0 (a function untouched by any
              vulnerability-fixing commit)

We read with csv.DictReader instead of pandas: Big-Vul's cleaned CSV is
hundreds of MB with large multi-line code cells, and streaming row-by-row
avoids holding the whole file (plus a full pandas copy of it) in memory.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator

from vulnerachek.data.loaders.base import BaseLoader
from vulnerachek.data.schema import VulnRecord, normalize_cwe, normalize_language

logger = logging.getLogger(__name__)

csv.field_size_limit(10_000_000)  # code cells can be large; default limit is too small


class BigVulLoader(BaseLoader):
    name = "bigvul"

    def load(self) -> Iterator[VulnRecord]:
        self._require_path()
        try:
            f = open(self.path, encoding="utf-8", errors="replace", newline="")
        except OSError as exc:
            raise type(exc)(f"[{self.name}] could not open {self.path}: {exc}") from exc

        with f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                yield from self._parse_row(row, row_num=i)

    def _parse_row(self, row: dict, *, row_num: int) -> Iterator[VulnRecord]:
        is_vul = str(self.col(row, "label", self.col(row, "vul", "0")) or "0").strip()
        language = normalize_language(self.col(row, "language", self.col(row, "lang", "")))
        if not language:
            self._skip(f"row {row_num}: unrecognized language")
            return

        project = str(self.col(row, "project", "") or "")
        commit_id = str(self.col(row, "commit_id", "") or "")
        cwe_type = normalize_cwe(self.col(row, "cwe_type", self.col(row, "CWE ID", "")))
        func_before = str(self.col(row, "code_before", self.col(row, "func_before", "")) or "").strip()
        func_after = str(self.col(row, "code_after", self.col(row, "func_after", "")) or "").strip()

        common = dict(language=language, source=self.name, project=project, commit_id=commit_id)

        if is_vul in {"1", "true", "True"}:
            if func_before:
                rec = self._build(code=func_before, label=1, cwe_type=cwe_type, **common)
                if rec:
                    yield self._emit(rec)
            if func_after and func_after != func_before:
                rec = self._build(code=func_after, label=0, cwe_type="", **common)
                if rec:
                    yield self._emit(rec)
            if not func_before and not func_after:
                self._skip(f"row {row_num}: vul=1 but both func_before/func_after empty")
        else:
            code = func_before or func_after
            if not code:
                self._skip(f"row {row_num}: vul=0 with no code")
                return
            rec = self._build(code=code, label=0, cwe_type="", **common)
            if rec:
                yield self._emit(rec)

    def _build(self, **kwargs) -> VulnRecord | None:
        try:
            return VulnRecord(**kwargs)
        except ValueError as exc:
            self._skip(str(exc))
            return None
