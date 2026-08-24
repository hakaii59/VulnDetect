"""Commit/project-level train/val/test split.

Why not a random per-line (per-record) split: a huge fraction of samples in
these datasets are near-duplicates of each other -- the vulnerable and
patched versions of the same function differ by one or two lines, and the
same helper/utility function is often copy-pasted across files or reused
almost unchanged across commits in the same project. A random split puts
one half of such a near-duplicate pair in train and the other in test.
GraphCodeBERT then does not need to learn what makes code vulnerable; it
only needs to recognize a function it has already memorized, which inflates
test F1 far above what the model achieves on genuinely unseen code. Grouping
every record by ``VulnRecord.group_key`` (project, falling back to commit,
falling back to the code hash) and keeping each group entirely inside one
split closes this leak.

The trade-off: because whole groups move together, exact target ratios
(e.g. 80/10/10) are only hit approximately -- a project contributing an
unusually large share of records can skew its split's size. We use a greedy
longest-processing-time bin-packing (largest groups placed first, each into
whichever split is currently furthest below its target share) which keeps
this skew small in practice as long as there are many groups, and we log the
achieved ratios plus a warning if any single group dominates.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from collections.abc import Sequence

from vulnerachek.data.schema import VulnRecord

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def group_aware_split(
    records: Sequence[VulnRecord],
    ratios: Sequence[float] = DEFAULT_RATIOS,
    split_names: Sequence[str] = DEFAULT_SPLIT_NAMES,
    seed: int = 42,
) -> dict[str, list[VulnRecord]]:
    if len(ratios) != len(split_names):
        raise ValueError("ratios and split_names must be the same length")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    groups: dict[str, list[VulnRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_key].append(record)

    total = len(records)
    if total == 0:
        return {name: [] for name in split_names}

    largest_group_frac = max(len(g) for g in groups.values()) / total
    if largest_group_frac > 0.2:
        logger.warning(
            "one group holds %.1f%% of all records; the split ratios may be "
            "noticeably skewed. Consider capping records per project upstream "
            "if this is unexpected.",
            largest_group_frac * 100,
        )

    # Shuffle first so ties in group size break randomly (reproducibly, via seed)
    # rather than in whatever order the input happened to arrive in.
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    group_items.sort(key=lambda kv: len(kv[1]), reverse=True)

    targets = {name: ratio * total for name, ratio in zip(split_names, ratios)}
    counts = {name: 0 for name in split_names}
    assigned: dict[str, list[VulnRecord]] = {name: [] for name in split_names}

    for _, group_records in group_items:
        best_split = max(split_names, key=lambda name: targets[name] - counts[name])
        assigned[best_split].extend(group_records)
        counts[best_split] += len(group_records)

    _log_split_stats(assigned, total)
    return assigned


def _log_split_stats(assigned: dict[str, list[VulnRecord]], total: int) -> None:
    for name, split_records in assigned.items():
        n = len(split_records)
        vuln_ratio = sum(r.label for r in split_records) / n if n else 0.0
        lang_counts = Counter(r.language for r in split_records)
        logger.info(
            "split %-5s: %6d records (%5.1f%% of total) | vulnerable=%.1f%% | languages=%s",
            name,
            n,
            100 * n / total if total else 0.0,
            100 * vuln_ratio,
            dict(lang_counts),
        )
