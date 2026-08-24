from vulnerachek.data.balance import bucket_summary, compute_sample_weights
from vulnerachek.data.schema import VulnRecord


def rec(label: int, language: str) -> VulnRecord:
    return VulnRecord(code="int x;", label=label, language=language, cwe_type="", source="t")


def test_weights_equalize_bucket_mass():
    # 90 safe-C, 10 vulnerable-C, 5 vulnerable-java -> 3 very unequal buckets
    records = [rec(0, "c")] * 90 + [rec(1, "c")] * 10 + [rec(1, "java")] * 5
    weights = compute_sample_weights(records)

    mass_by_bucket: dict[tuple[int, str], float] = {}
    for r, w in zip(records, weights):
        mass_by_bucket[(r.label, r.language)] = mass_by_bucket.get((r.label, r.language), 0.0) + w

    masses = list(mass_by_bucket.values())
    assert max(masses) - min(masses) < 1e-6  # every bucket contributes equal total weight


def test_weights_mean_normalized_to_one():
    records = [rec(0, "c")] * 5 + [rec(1, "cpp")] * 2
    weights = compute_sample_weights(records)
    assert abs(sum(weights) / len(weights) - 1.0) < 1e-9


def test_empty_input():
    assert compute_sample_weights([]) == []


def test_bucket_summary_counts():
    records = [rec(0, "c"), rec(0, "c"), rec(1, "java")]
    assert bucket_summary(records) == {(0, "c"): 2, (1, "java"): 1}
