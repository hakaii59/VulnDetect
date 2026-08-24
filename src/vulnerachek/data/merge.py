"""Merge records from all four dataset loaders into one deduplicated stream.

Why dedup at all: MegaVul, Big-Vul, CVEfixes and PrimeVul were all built by
mining CVE-fixing commits from GitHub, and their crawl windows overlap
heavily (the same Linux kernel / FFmpeg / OpenSSL commits show up in more
than one dataset). Without dedup, the exact same function can land in one
dataset's train split and another dataset's test split -- an especially
sneaky form of leakage that class-balancing or a good split algorithm won't
catch, because it happens *before* splitting. We dedup on whitespace-
normalized code content (VulnRecord.content_hash), keeping the first
occurrence encountered.

This holds all records in memory (a dict keyed by content hash). Function-
level source code is small (a few hundred MB raw across all four datasets),
so this is fine for a one-off data-prep script; it is not part of the
training loop.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

from vulnerachek.data.schema import VulnRecord

logger = logging.getLogger(__name__)


def merge_records(*sources: Iterable[VulnRecord]) -> Iterator[VulnRecord]:
    """Chain records from multiple loaders, dropping exact content duplicates.

    On a duplicate, the record's label must agree across every source that
    produced it; a disagreement (dataset A says this exact function is
    vulnerable, dataset B says the identical text is safe) means at least
    one dataset mislabeled it, so the record is dropped entirely rather than
    guessing which source to trust.
    """
    first_seen: dict[str, VulnRecord] = {}
    labels_seen: dict[str, set[int]] = {}
    total = 0

    for source in sources:
        for record in source:
            total += 1
            key = record.content_hash
            if key not in first_seen:
                first_seen[key] = record
                labels_seen[key] = {record.label}
            else:
                labels_seen[key].add(record.label)

    kept = 0
    conflicts = 0
    for key, record in first_seen.items():
        if len(labels_seen[key]) > 1:
            conflicts += 1
            continue
        kept += 1
        yield record

    duplicates = total - len(first_seen)
    logger.info(
        "merge: %d input records -> %d unique -> %d kept "
        "(%d exact duplicates collapsed, %d label conflicts dropped)",
        total,
        len(first_seen),
        kept,
        duplicates,
        conflicts,
    )
