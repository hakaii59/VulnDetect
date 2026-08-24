"""Per-record sample weights for training on an imbalanced, multi-language corpus.

Two imbalances stack in this corpus: vulnerable functions are a small
minority of all functions (real codebases are mostly safe), and the four
source datasets are not evenly split across C, C++ and Java (Big-Vul and
PrimeVul are C/C++-only, MegaVul contributes the bulk of the Java examples).
Training on the raw distribution lets GraphCodeBERT reach a low loss by
mostly predicting the majority language/label combination.

Why weighted sampling instead of duplicating minority rows (oversampling) or
dropping majority rows (undersampling): duplication makes the exact same
token sequence appear many times per epoch, which nudges the model toward
memorizing that specific text rather than the underlying pattern, and
undersampling throws away real, already-scarce labeled data. A per-sample
weight consumed by torch.utils.data.WeightedRandomSampler instead reshapes
*how often each row is drawn* without touching the dataset itself -- every
distinct function is still seen, just at a rate that equalizes the
label x language combinations over the course of an epoch.

Weights are computed only for the train split. val/test must keep the
dataset's natural (imbalanced) distribution, because they exist to estimate
how the model will perform on real, imbalanced incoming code -- reweighting
them would make precision/recall/F1 numbers meaningless.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from vulnerachek.data.schema import VulnRecord


def compute_sample_weights(records: Sequence[VulnRecord]) -> list[float]:
    """Return one weight per record (same order as ``records``), mean-normalized to 1.0.

    Each weight is inversely proportional to the size of that record's
    (label, language) bucket, so every bucket ends up contributing roughly
    equal total weight regardless of how many raw examples it has.
    """
    if not records:
        return []

    bucket_counts = Counter((r.label, r.language) for r in records)
    raw_weights = [1.0 / bucket_counts[(r.label, r.language)] for r in records]

    mean_weight = sum(raw_weights) / len(raw_weights)
    return [w / mean_weight for w in raw_weights]


def bucket_summary(records: Sequence[VulnRecord]) -> dict[tuple[int, str], int]:
    """(label, language) -> record count, useful for logging/tests before/after weighting."""
    return dict(Counter((r.label, r.language) for r in records))
