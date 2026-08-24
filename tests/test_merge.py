from vulnerachek.data.merge import merge_records
from vulnerachek.data.schema import VulnRecord


def rec(code, label, source, **kw):
    return VulnRecord(code=code, label=label, language="c", cwe_type="", source=source, **kw)


def test_exact_duplicates_across_sources_collapse_to_one():
    source_a = [rec("int x = 1;", 0, "a")]
    source_b = [rec("int   x =\n1;", 0, "b")]  # same content, different whitespace
    merged = list(merge_records(source_a, source_b))
    assert len(merged) == 1


def test_label_conflict_across_sources_is_dropped():
    source_a = [rec("int x = 1;", 1, "a")]
    source_b = [rec("int x = 1;", 0, "b")]
    merged = list(merge_records(source_a, source_b))
    assert merged == []


def test_distinct_records_are_all_kept():
    source_a = [rec("int x = 1;", 0, "a"), rec("int y = 2;", 1, "a")]
    merged = list(merge_records(source_a))
    assert len(merged) == 2
