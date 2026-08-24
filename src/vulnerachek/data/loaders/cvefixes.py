"""Loader for CVEfixes (https://github.com/secureIT-project/CVEfixes).

Expected input: the ``CVEfixes.db`` SQLite database (built from the
compressed SQL dump on Zenodo, DOI 10.5281/zenodo.4476563, via the
converter script in the upstream repo's INSTALL.md).

CVEfixes already stores one row per *method version* (before and after the
fix are separate rows in ``method_change``, unlike Big-Vul's before/after
columns on one row), so each SQL row maps directly to one VulnRecord:

    method_change.before_change == 1/true  -> label 1 (vulnerable)
    method_change.before_change == 0/false -> label 0 (patched)

Join path used by the default query:
    method_change -> file_change (language, commit hash)
                   -> commits (repo_url)
                   -> repository (project name)
    commits.hash -> fixes.hash -> cve.cve_id -> cwe_classification (CWE id)

CVEfixes' schema has changed across releases (see the repo's ER diagram),
so if the default query raises sqlite3.OperationalError (missing
table/column) we fall back to a minimal query using only method_change +
file_change (code/label/language, no project/CWE) and log a warning. Pass a
fully custom query via ``column_map={"query": "SELECT ..."}`` in
configs/dataset.yaml if neither works for your release -- the loader only
requires the result columns aliased as code/before_change/language/project/
commit_id/cwe_id/method_name.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator

from vulnerachek.data.loaders.base import BaseLoader
from vulnerachek.data.schema import VulnRecord, normalize_cwe, normalize_language

logger = logging.getLogger(__name__)

_FULL_QUERY = """
SELECT
    mc.code AS code,
    mc.before_change AS before_change,
    mc.name AS method_name,
    fc.programming_language AS language,
    fc.hash AS commit_id,
    r.repo_name AS project,
    cwe.cwe_id AS cwe_id
FROM method_change mc
JOIN file_change fc ON mc.file_change_id = fc.file_change_id
LEFT JOIN commits c ON fc.hash = c.hash
LEFT JOIN repository r ON c.repo_url = r.repo_url
LEFT JOIN fixes fx ON c.hash = fx.hash
LEFT JOIN cwe_classification cwe ON fx.cve_id = cwe.cve_id
"""

_FALLBACK_QUERY = """
SELECT
    mc.code AS code,
    mc.before_change AS before_change,
    mc.name AS method_name,
    fc.programming_language AS language,
    fc.hash AS commit_id,
    '' AS project,
    '' AS cwe_id
FROM method_change mc
JOIN file_change fc ON mc.file_change_id = fc.file_change_id
"""


class CVEfixesLoader(BaseLoader):
    name = "cvefixes"

    def load(self) -> Iterator[VulnRecord]:
        self._require_path()
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            query = self.column_map.get("query", _FULL_QUERY)
            try:
                cursor = conn.execute(query)
            except sqlite3.OperationalError as exc:
                if "query" in self.column_map:
                    raise
                logger.warning(
                    "[%s] default query failed (%s); retrying with a minimal "
                    "method_change+file_change query (no project/CWE). Supply "
                    "column_map.query in configs/dataset.yaml to fix this properly.",
                    self.name,
                    exc,
                )
                cursor = conn.execute(_FALLBACK_QUERY)

            for row in cursor:
                record = self._parse_row(dict(row))
                if record is not None:
                    yield self._emit(record)
        finally:
            conn.close()

    def _parse_row(self, row: dict) -> VulnRecord | None:
        code = str(row.get("code") or "").strip()
        if not code:
            self._skip("empty code")
            return None

        before_change = row.get("before_change")
        label = 1 if str(before_change).strip().lower() in {"1", "true", "t"} else 0

        language = normalize_language(row.get("language"))
        if not language:
            self._skip(f"unrecognized language: {row.get('language')!r}")
            return None

        try:
            return VulnRecord(
                code=code,
                label=label,
                language=language,
                cwe_type=normalize_cwe(row.get("cwe_id")),
                source=self.name,
                project=str(row.get("project") or ""),
                commit_id=str(row.get("commit_id") or ""),
                func_name=str(row.get("method_name") or ""),
            )
        except ValueError as exc:
            self._skip(str(exc))
            return None
