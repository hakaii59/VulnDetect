from pathlib import Path

import pytest

from vulnerachek.data.loaders.base import LoaderError
from vulnerachek.data.loaders.bigvul import BigVulLoader
from vulnerachek.data.loaders.cvefixes import CVEfixesLoader
from vulnerachek.data.loaders.megavul import MegaVulLoader
from vulnerachek.data.loaders.primevul import PrimeVulLoader


def test_bigvul_expands_vul_row_into_before_after_pair(bigvul_csv):
    records = list(BigVulLoader(bigvul_csv).load())
    # row 1: vul=1 -> 2 records (before=1, after=0); row 2: vul=0 -> 1 record
    # (before==after so after is skipped); row 3: vul=1 -> 2 records
    assert len(records) == 5
    labels = [r.label for r in records]
    assert labels.count(1) == 2
    assert labels.count(0) == 3
    assert all(r.source == "bigvul" for r in records)
    assert {r.project for r in records} == {"openssl", "curl"}
    cwe_types = {r.cwe_type for r in records if r.cwe_type}
    assert cwe_types == {"CWE-119", "CWE-131"}


def test_bigvul_missing_file_raises_loader_error(tmp_path):
    loader = BigVulLoader(tmp_path / "missing.csv")
    with pytest.raises(LoaderError):
        list(loader.load())


def test_megavul_parses_directory_of_json_files(megavul_dir):
    records = list(MegaVulLoader(megavul_dir).load())
    assert len(records) == 3
    by_commit = {r.commit_id: r for r in records}
    assert by_commit["mv001"].label == 1
    assert by_commit["mv001"].language == "c"
    assert by_commit["mv001"].cwe_type == "CWE-120"
    assert by_commit["mv002"].label == 0
    assert by_commit["mv003"].language == "java"
    assert by_commit["mv003"].cwe_type == "CWE-78"


def test_megavul_single_file_path(megavul_json):
    records = list(MegaVulLoader(megavul_json).load())
    assert len(records) == 3


def test_primevul_parses_jsonl(primevul_jsonl):
    records = list(PrimeVulLoader(primevul_jsonl).load())
    assert len(records) == 3
    labels = {r.commit_id: r.label for r in records}
    assert labels == {"pv001": 1, "pv002": 0, "pv003": 1}
    languages = {r.commit_id: r.language for r in records}
    # language is inferred per-record from file_name's extension (.c vs .cpp)
    assert languages == {"pv001": "c", "pv002": "c", "pv003": "cpp"}
    assert {r.project for r in records} == {"ffmpeg/ffmpeg", "libgd/libgd"}


def test_cvefixes_joins_project_and_cwe(cvefixes_db):
    records = list(CVEfixesLoader(cvefixes_db).load())
    assert len(records) == 2
    by_label = {r.label: r for r in records}
    assert by_label[1].project == "openssl/openssl"
    assert by_label[1].cwe_type == "CWE-120"
    assert by_label[1].commit_id == "cf001"
    assert by_label[0].label == 0
    assert "strcpy" in by_label[1].code


def test_cvefixes_falls_back_when_default_query_fails(tmp_path):
    import sqlite3

    db_path = tmp_path / "minimal.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE file_change (file_change_id TEXT PRIMARY KEY, hash TEXT, programming_language TEXT);
        CREATE TABLE method_change (
            method_change_id TEXT PRIMARY KEY, file_change_id TEXT, name TEXT, code TEXT, before_change TEXT
        );
        INSERT INTO file_change VALUES ('fc1', 'h1', 'Java');
        INSERT INTO method_change VALUES ('mc1', 'fc1', 'foo', 'void foo() {}', 'true');
        """
    )
    conn.commit()
    conn.close()

    records = list(CVEfixesLoader(db_path).load())
    assert len(records) == 1
    assert records[0].language == "java"
    assert records[0].project == ""  # repository table absent -> fallback query
