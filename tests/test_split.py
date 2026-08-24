import pytest

from vulnerachek.data.schema import VulnRecord
from vulnerachek.data.split import group_aware_split


def make_records(project: str, n: int, label: int = 0) -> list[VulnRecord]:
    return [
        VulnRecord(code=f"int f{i}() {{ return {i}; }}", label=label, language="c", cwe_type="", source="t", project=project)
        for i in range(n)
    ]


def test_same_project_never_split_across_sets():
    records = make_records("proj-a", 20) + make_records("proj-b", 20) + make_records("proj-c", 20)
    splits = group_aware_split(records, seed=1)

    project_to_splits: dict[str, set[str]] = {}
    for split_name, split_records in splits.items():
        for r in split_records:
            project_to_splits.setdefault(r.project, set()).add(split_name)

    assert all(len(s) == 1 for s in project_to_splits.values())


def test_all_records_are_preserved_exactly_once():
    records = make_records("proj-a", 15) + make_records("proj-b", 15)
    splits = group_aware_split(records, seed=1)
    total_out = sum(len(v) for v in splits.values())
    assert total_out == len(records)


def test_split_ratios_are_approximately_respected_with_many_small_groups():
    records = []
    for i in range(100):
        records += make_records(f"proj-{i}", 5)
    splits = group_aware_split(records, ratios=(0.8, 0.1, 0.1), seed=1)
    total = len(records)
    assert 0.7 * total < len(splits["train"]) < 0.9 * total
    assert 0.03 * total < len(splits["val"]) < 0.17 * total
    assert 0.03 * total < len(splits["test"]) < 0.17 * total


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        group_aware_split(make_records("p", 5), ratios=(0.5, 0.4, 0.4))


def test_deterministic_given_same_seed():
    records = make_records("proj-a", 10) + make_records("proj-b", 10) + make_records("proj-c", 10)
    splits_1 = group_aware_split(records, seed=7)
    splits_2 = group_aware_split(records, seed=7)
    assert {r.project for r in splits_1["train"]} == {r.project for r in splits_2["train"]}
