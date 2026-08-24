"""Shared pytest fixtures.

The CVEfixes fixture DB is built programmatically (instead of committing a
binary .db file) so its schema stays next to, and obviously in sync with,
the test that exercises the loader.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def bigvul_csv() -> Path:
    return FIXTURES_DIR / "bigvul_sample.csv"


@pytest.fixture
def megavul_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def megavul_json() -> Path:
    return FIXTURES_DIR / "megavul_simple.json"


@pytest.fixture
def primevul_jsonl() -> Path:
    return FIXTURES_DIR / "primevul_sample.jsonl"


@pytest.fixture
def cvefixes_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "CVEfixes_sample.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE repository (repo_url TEXT PRIMARY KEY, repo_name TEXT);
        CREATE TABLE commits (hash TEXT PRIMARY KEY, repo_url TEXT);
        CREATE TABLE file_change (
            file_change_id TEXT PRIMARY KEY,
            hash TEXT,
            programming_language TEXT
        );
        CREATE TABLE method_change (
            method_change_id TEXT PRIMARY KEY,
            file_change_id TEXT,
            name TEXT,
            code TEXT,
            before_change TEXT
        );
        CREATE TABLE fixes (hash TEXT, cve_id TEXT);
        CREATE TABLE cve (cve_id TEXT PRIMARY KEY);
        CREATE TABLE cwe_classification (cve_id TEXT, cwe_id TEXT);
        """
    )
    conn.executescript(
        """
        INSERT INTO repository VALUES ('https://github.com/openssl/openssl', 'openssl/openssl');
        INSERT INTO commits VALUES ('cf001', 'https://github.com/openssl/openssl');
        INSERT INTO file_change VALUES ('fc1', 'cf001', 'C');
        INSERT INTO method_change VALUES
            ('mc1', 'fc1', 'do_encrypt',
             'void do_encrypt(char *buf) { strcpy(buf, input); }', 'true');
        INSERT INTO method_change VALUES
            ('mc2', 'fc1', 'do_encrypt',
             'void do_encrypt(char *buf) { strncpy(buf, input, 63); }', 'false');
        INSERT INTO fixes VALUES ('cf001', 'CVE-2021-0001');
        INSERT INTO cve VALUES ('CVE-2021-0001');
        INSERT INTO cwe_classification VALUES ('CVE-2021-0001', 'CWE-120');
        """
    )
    conn.commit()
    conn.close()
    return db_path
