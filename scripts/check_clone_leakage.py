"""Look for residual clone leakage the whitespace-normalized dedup would miss.

`build_dataset.py` dedups on whitespace-normalized source, so byte-identical and
reindented twins are gone. It cannot catch a function that was copied and then
had its variables renamed or its constants changed — common in Big-Vul, which
contains many forked projects fixing the same CVE.

This reduces each function to a *structural skeleton*: comments and literals
removed, every non-keyword identifier collapsed to `V`. Two functions with the
same skeleton differ only in naming.
"""

import hashlib
import re

import pandas as pd

C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
    "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register",
    "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while", "bool", "true", "false", "NULL",
    "class", "public", "private", "protected", "virtual", "new", "delete", "namespace",
    "template", "typename", "this", "using", "try", "catch", "throw", "nullptr", "operator",
    "static_cast", "const_cast", "reinterpret_cast", "dynamic_cast", "friend", "explicit",
    "mutable",
}

IDENT = re.compile(r"[A-Za-z_]\w*")
NUM = re.compile(r"\b\d+\b")
STR = re.compile(r'"(?:[^"\\]|\\.)*"')
CH = re.compile(r"'(?:[^'\\]|\\.)*'")
CMT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
WS = re.compile(r"\s+")


def skeleton(code: str) -> str:
    s = CMT.sub(" ", str(code))
    s = STR.sub('"S"', s)
    s = CH.sub("'C'", s)
    s = NUM.sub("0", s)
    s = IDENT.sub(lambda m: m.group(0) if m.group(0) in C_KEYWORDS else "V", s)
    return WS.sub(" ", s).strip()


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


tr = pd.read_parquet("data/processed/train.parquet")
te = pd.read_parquet("data/processed/test.parquet")
print(f"train {len(tr):,}   test {len(te):,}")

for df in (tr, te):
    df["_skel"] = [skeleton(c) for c in df["func_before"]]
    df["_len"] = df["_skel"].str.len()

print()
print(f"{'threshold':<26} {'test funcs':>11} {'structural clone in train':>27}")
for min_len, label in [(0, "all functions"), (200, "skeleton >= 200 chars"), (500, "skeleton >= 500 chars")]:
    t = tr[tr["_len"] >= min_len]
    e = te[te["_len"] >= min_len]
    train_hashes = {h(x) for x in t["_skel"]}
    hit = pd.Series([h(x) in train_hashes for x in e["_skel"]], index=e.index)
    print(f"{label:<26} {len(e):>11,} {f'{hit.sum():,} ({hit.mean():.1%})':>27}")

# Does the overlap concentrate in the vulnerable class? That would matter most.
e = te[te["_len"] >= 200]
t = tr[tr["_len"] >= 200]
train_hashes = {h(x) for x in t["_skel"]}
hit = pd.Series([h(x) in train_hashes for x in e["_skel"]], index=e.index)
print()
print("Among test functions with skeleton >= 200 chars:")
for label in (0, 1):
    m = e["vul"] == label
    print(f"  vul={label}: {hit[m].sum():,} / {m.sum():,} have a structural twin in train ({hit[m].mean():.1%})")
