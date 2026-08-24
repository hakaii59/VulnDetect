"""Abstract loader interface every dataset-specific loader implements.

Design rationale: real dataset releases drift (column renamed between
versions, a field turns into a list instead of a string, etc.) and we cannot
verify the exact files a given user downloaded ahead of time. Instead of
hardcoding column names inside parsing logic, every loader resolves logical
field names through ``self.col(...)`` / ``self.column_map`` which the CLI
lets a user override via configs/dataset.yaml without touching Python code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vulnerachek.data.schema import VulnRecord

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """Raised when a loader's raw input path is missing or unreadable."""


class BaseLoader(ABC):
    """Yields :class:`VulnRecord` instances parsed from one dataset's raw files.

    Subclasses implement :meth:`load`. Individual malformed rows should be
    skipped with ``logger.warning`` rather than raising, so one bad row (or
    one dataset's schema surprise) never aborts the whole merge across all
    four datasets.
    """

    name: str

    def __init__(self, path: str | Path, column_map: dict[str, str] | None = None) -> None:
        self.path = Path(path)
        self.column_map = column_map or {}
        self.skipped = 0
        self.loaded = 0

    @abstractmethod
    def load(self) -> Iterator[VulnRecord]:
        """Parse ``self.path`` and yield unified records. Must not raise on a single bad row."""

    def col(self, row: dict[str, Any], field: str, default: Any = "") -> Any:
        """Resolve a logical field name (e.g. "code") to the actual raw column via column_map."""
        key = self.column_map.get(field, field)
        value = row.get(key, default)
        return default if value is None else value

    def _require_path(self) -> None:
        if not self.path.exists():
            raise LoaderError(
                f"[{self.name}] raw data not found at {self.path}. "
                f"See docs/DATASETS.md for download instructions."
            )

    def _skip(self, reason: str) -> None:
        self.skipped += 1
        logger.debug("[%s] skipped row: %s", self.name, reason)

    def _emit(self, record: VulnRecord) -> VulnRecord:
        self.loaded += 1
        return record

    def summary(self) -> str:
        return f"[{self.name}] loaded={self.loaded} skipped={self.skipped}"
