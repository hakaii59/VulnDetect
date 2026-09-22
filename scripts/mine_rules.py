"""Measure which dangerous-API patterns actually discriminate on Big-Vul.

Rules should be chosen from evidence, not from a textbook list: 6 of the
original 10 never fired once on 1,500 real functions.

For each candidate pattern this reports:
  hits   - how many functions contain it
  lift   - P(vulnerable | contains it) / P(vulnerable), so 1.0 means the
           pattern carries no signal at all
"""

import re

import pandas as pd

CANDIDATES = {
    # --- unbounded string ops (CWE-120/787) ---
    "gets": r"\bgets\s*\(",
    "strcpy": r"\bstrcpy\s*\(",
    "strcat": r"\bstrcat\s*\(",
    "sprintf": r"\bsprintf\s*\(",
    "vsprintf": r"\bvsprintf\s*\(",
    "wcscpy/wcscat": r"\bwcs(cpy|cat)\s*\(",
    "stpcpy": r"\bstpcpy\s*\(",
    # --- bounded but misuse-prone (CWE-119/125) ---
    "strncpy": r"\bstrncpy\s*\(",
    "strncat": r"\bstrncat\s*\(",
    "memcpy/memmove": r"\b(memcpy|memmove)\s*\(",
    "memset": r"\bmemset\s*\(",
    "alloca": r"\balloca\s*\(",
    # --- format strings (CWE-134) ---
    "printf(non-literal)": r"\b(?:v|f|s|sn)?printf\s*\(\s*[A-Za-z_]\w*\s*[,)]",
    "scanf %s no width": r"\b[fs]?scanf\s*\([^;]*%s",
    # --- integer issues (CWE-190/189) ---
    "malloc(a*b)": r"\b(?:m|c|re)alloc\s*\([^;]*[A-Za-z_0-9\]\)]\s*\*\s*[A-Za-z_0-9(]",
    "alloc(a+b)": r"\b(?:m|re)alloc\s*\([^;]*[A-Za-z_0-9\]\)]\s*\+\s*[A-Za-z_0-9(]",
    "atoi/atol": r"\bato[ilf]\s*\(",
    "int cast of size": r"\(\s*(?:unsigned\s+)?(?:short|int|char)\s*\)\s*[A-Za-z_]\w*(?:len|size|count|num)",
    # --- memory lifetime (CWE-416/476/399) ---
    "free": r"\bfree\s*\(",
    "realloc": r"\brealloc\s*\(",
    "malloc no null check": r"\bmalloc\s*\(",
    # --- command / path (CWE-78/22) ---
    "system/exec/popen": r"\b(system|popen|exec[lv][ep]?[e]?)\s*\(",
    "tmpnam/mktemp": r"\b(tmpnam|mktemp|tempnam)\s*\(",
    # --- concurrency (CWE-362) ---
    "access-then-open": r"\baccess\s*\(",
    # --- misc parsing ---
    "strtok": r"\bstrtok\s*\(",
    "sizeof on pointer arg": r"sizeof\s*\(\s*\*?\s*[A-Za-z_]\w*\s*\)",
}

df = pd.concat([pd.read_parquet(f"data/processed/{n}.parquet") for n in ("train", "val", "test")])
code = df["func_before"].astype(str)
y = df["vul"].to_numpy()
base = y.mean()
print(f"{len(df):,} functions, base vulnerable rate {base:.2%}\n")

rows = []
for name, pattern in CANDIDATES.items():
    rx = re.compile(pattern)
    hit = code.str.contains(rx, regex=True, na=False).to_numpy()
    n = int(hit.sum())
    if n == 0:
        rows.append((name, 0, 0.0, 0.0, 0))
        continue
    rate = y[hit].mean()
    rows.append((name, n, rate, rate / base, int(y[hit].sum())))

rows.sort(key=lambda r: -r[3])
print(f"{'pattern':26} {'hits':>8} {'vul hits':>9} {'P(vul|hit)':>11} {'lift':>7}")
print("-" * 66)
for name, n, rate, lift, nvul in rows:
    if n == 0:
        print(f"{name:26} {'0':>8} {'-':>9} {'-':>11} {'-':>7}")
    else:
        print(f"{name:26} {n:>8,} {nvul:>9,} {rate:>10.2%} {lift:>7.2f}")
