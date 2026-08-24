import pytest

from vulnerachek.data.schema import VulnRecord, normalize_cwe, normalize_language


def make_record(**overrides):
    defaults = dict(code="int x = 1;", label=0, language="c", cwe_type="", source="test")
    defaults.update(overrides)
    return VulnRecord(**defaults)


def test_valid_record_roundtrips_to_row():
    record = make_record(project="proj", commit_id="abc123")
    row = record.to_row()
    assert row["code"] == "int x = 1;"
    assert row["label"] == 0
    assert row["language"] == "c"
    assert row["project"] == "proj"
    assert "id" in row and len(row["id"]) == 16


def test_rejects_invalid_label():
    with pytest.raises(ValueError):
        make_record(label=2)


def test_rejects_invalid_language():
    with pytest.raises(ValueError):
        make_record(language="python")


def test_rejects_empty_code():
    with pytest.raises(ValueError):
        make_record(code="   ")


def test_content_hash_ignores_whitespace_differences():
    a = make_record(code="int   x = 1;")
    b = make_record(code="int x =\n1;")
    assert a.content_hash == b.content_hash


def test_group_key_prefers_project_then_commit_then_hash():
    assert make_record(project="p", commit_id="c").group_key == "p"
    assert make_record(commit_id="c").group_key == "c"
    assert make_record().group_key == make_record().content_hash


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("119", "CWE-119"),
        ("CWE-119", "CWE-119"),
        (119, "CWE-119"),
        ("", ""),
        (None, ""),
        ("NVD-CWE-noinfo", ""),
        ("nan", ""),
    ],
)
def test_normalize_cwe(raw, expected):
    assert normalize_cwe(raw) == expected


@pytest.mark.parametrize(
    "raw,filename,expected",
    [
        ("c", "", "c"),
        ("C++", "", "cpp"),
        ("Java", "", "java"),
        ("", "foo.cpp", "cpp"),
        ("", "foo.h", "c"),
        ("", "foo.py", ""),
    ],
)
def test_normalize_language(raw, filename, expected):
    assert normalize_language(raw, filename=filename) == expected
