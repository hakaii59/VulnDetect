"""Layer 1: regex-based pattern matching for common C/C++ vulnerability patterns.

Fast and cheap, but prone to false positives (e.g. matching "strcpy" inside a
comment or a string literal). Layer 2 (ast_layer) filters those out.

Choosing the rules
------------------
The first version of this file was a textbook list of banned functions, and
6 of its 10 rules never fired once on 1,500 real Big-Vul functions. Several
were written too narrowly to match real code: `R007` required
`scanf(...%s)` with no `;` between, `R005` matched only a bare `printf(x)`,
and `R008` only `malloc(a * b)` with single identifiers.

The current set is chosen from measurement instead. For each candidate,
`P(vulnerable | pattern present)` was computed over all 163,636 functions and
divided by the 5.32% base rate to give a **lift**:

    scanf(..%s) no width     lift 6.08      strncpy               lift 2.87
    strtok                   lift 4.03      system/popen/exec     lift 2.83
    alloca                   lift 3.94      alloc(a + b)          lift 2.77
    size truncation cast     lift 3.79      strcpy                lift 2.74
    [mcre]alloc(a * b)       lift 3.44      atoi/atol             lift 2.56
    strncat                  lift 3.32      strcat                lift 2.41
    realloc                  lift 3.06      memcpy/memmove        lift 2.25
                                            sprintf               lift 2.16
                                            free                  lift 2.07

Patterns that measured near 1.0 were dropped: `sizeof(ptr)` appears in 11,917
functions at lift 2.12 but fires on 7% of the corpus, and a permissive
`printf(ident, ...)` sits at lift 1.94 because passing a format variable is
ordinary code. `gets()` is kept at zero hits — it was removed from C11 and does
not appear in Big-Vul at all, but a scanner that misses it is broken.

Lift is not precision. At a 5.32% base rate even lift 3.4 means roughly five in
six hits are not on a function Big-Vul labels vulnerable, which is why Layer 2
validates and Layer 3 corroborates.

Severity drives the gate: a `high` rule confirmed by Layer 2 fails CI on its
own (see `analyze`), so `high` is reserved for calls that are unsafe by
construction rather than merely worth a look.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Rule:
    rule_id: str
    cwe: str
    severity: str  # "high" | "medium" | "low"
    message: str
    pattern: re.Pattern
    #: Function names this rule targets. Layer 2 confirms the finding only if
    #: one of them is genuinely called on that line. Empty means the pattern is
    #: not a call, so Layer 2 falls back to a comment/string-literal check.
    func_names: tuple[str, ...] = ()


@dataclass
class RegexFinding:
    rule_id: str
    cwe: str
    severity: str
    message: str
    line: int
    col: int
    match_text: str
    func_names: tuple[str, ...] = field(default_factory=tuple)


RULES: list[Rule] = [
    # ---------------------------------------------------------------- high
    Rule("R001", "CWE-242", "high",
         "gets() cannot be used safely: it has no way to limit input length",
         re.compile(r"\bgets\s*\("), ("gets",)),
    Rule("R002", "CWE-120", "high",
         "strcpy() does not check the destination buffer size",
         re.compile(r"\bstrcpy\s*\("), ("strcpy",)),
    Rule("R003", "CWE-120", "high",
         "strcat() does not check the destination buffer size",
         re.compile(r"\bstrcat\s*\("), ("strcat",)),
    Rule("R004", "CWE-120", "high",
         "sprintf() has no size limit and can overflow the destination buffer",
         re.compile(r"\bsprintf\s*\("), ("sprintf", "vsprintf")),
    Rule("R005", "CWE-134", "high",
         "printf-family call whose format string is a variable: if that variable "
         "is attacker-controlled this is a format string vulnerability",
         re.compile(r"\b(?:v|f|s|sn|vs|vsn)?printf\s*\(\s*[A-Za-z_]\w*\s*\)"),
         ("printf", "vprintf", "fprintf", "sprintf", "snprintf", "vsprintf", "vsnprintf")),
    Rule("R006", "CWE-78", "high",
         "spawns a shell or process; unsanitized input leads to command injection",
         re.compile(r"\b(?:system|popen|exec[lv][ep]{0,2})\s*\("),
         ("system", "popen", "execl", "execlp", "execle", "execv", "execvp", "execvpe")),
    Rule("R007", "CWE-120", "high",
         "scanf() with %s and no field width will overflow the destination buffer",
         re.compile(r"\b[fsv]?scanf\s*\([^;]*%s"),
         ("scanf", "fscanf", "sscanf", "vscanf")),

    # -------------------------------------------------------------- medium
    Rule("R008", "CWE-190", "medium",
         "allocation size computed by multiplication can integer-overflow, "
         "allocating less memory than intended",
         re.compile(r"\b(?:m|c|re)alloc\s*\([^;]*[A-Za-z_0-9\]\)]\s*\*\s*[A-Za-z_0-9(]"),
         ("malloc", "calloc", "realloc")),
    Rule("R010", "CWE-119", "medium",
         "memcpy()/memmove() without an obvious size check can overflow the destination",
         re.compile(r"\b(?:memcpy|memmove)\s*\("), ("memcpy", "memmove")),
    Rule("R011", "CWE-170", "medium",
         "strncpy() does not null-terminate when the source fills the buffer, "
         "leaving the destination unterminated",
         re.compile(r"\bstrncpy\s*\("), ("strncpy",)),
    Rule("R012", "CWE-119", "medium",
         "strncat()'s size argument is the space remaining, not the buffer size — "
         "a classic off-by-one source",
         re.compile(r"\bstrncat\s*\("), ("strncat",)),
    Rule("R013", "CWE-789", "medium",
         "alloca() allocates on the stack with no failure mode; an attacker-controlled "
         "size exhausts the stack",
         re.compile(r"\balloca\s*\("), ("alloca",)),
    Rule("R014", "CWE-401", "medium",
         "realloc() returns NULL on failure without freeing the original pointer; "
         "assigning it back to the same variable leaks the old block",
         re.compile(r"\brealloc\s*\("), ("realloc",)),
    Rule("R015", "CWE-190", "medium",
         "allocation size computed by addition can integer-overflow and wrap to a "
         "small allocation",
         re.compile(r"\b(?:m|re)alloc\s*\([^;]*[A-Za-z_0-9\]\)]\s*\+\s*[A-Za-z_0-9(]"),
         ("malloc", "realloc")),
    Rule("R016", "CWE-197", "medium",
         "a length or size value cast down to a narrower type can truncate, "
         "turning a large size into a small one",
         re.compile(r"\(\s*(?:unsigned\s+)?(?:short|int|char)\s*\)\s*[A-Za-z_]\w*(?:len|size|count|num)\b",
                    re.IGNORECASE)),
    Rule("R017", "CWE-190", "medium",
         "atoi()/atol() cannot report an error or an overflow; use strtol() and "
         "check errno",
         re.compile(r"\bato[ilf]\s*\("), ("atoi", "atol", "atof")),

    # ----------------------------------------------------------------- low
    Rule("R009", "CWE-416", "low",
         "free() call — verify the pointer is not used or freed again afterwards",
         re.compile(r"\bfree\s*\("), ("free",)),
    Rule("R018", "CWE-476", "low",
         "strtok() returns NULL when no token remains and keeps state between calls; "
         "it is neither null-safe nor thread-safe",
         re.compile(r"\bstrtok\s*\("), ("strtok",)),
]


def scan(code: str) -> list[RegexFinding]:
    """Scan source code line-by-line against all rules."""
    findings: list[RegexFinding] = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        for rule in RULES:
            for m in rule.pattern.finditer(line):
                findings.append(RegexFinding(
                    rule_id=rule.rule_id,
                    cwe=rule.cwe,
                    severity=rule.severity,
                    message=rule.message,
                    line=lineno,
                    col=m.start(),
                    match_text=line.strip(),
                    func_names=rule.func_names,
                ))
    return findings
