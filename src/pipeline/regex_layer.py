"""Layer 1: regex-based pattern matching for common C/C++ vulnerability patterns.

Fast and cheap, but prone to false positives (e.g. matching "strcpy" inside a
comment or a string literal). Layer 2 (ast_layer) filters those out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Rule:
    rule_id: str
    cwe: str
    severity: str
    message: str
    pattern: re.Pattern
    func_name: str | None = None  # set when the rule targets a specific function call


@dataclass
class RegexFinding:
    rule_id: str
    cwe: str
    severity: str
    message: str
    line: int
    col: int
    match_text: str
    func_name: str | None = None


RULES: list[Rule] = [
    Rule("R001", "CWE-120", "high",
         "gets() has no bounds checking and can overflow the destination buffer",
         re.compile(r"\bgets\s*\("), func_name="gets"),
    Rule("R002", "CWE-120", "high",
         "strcpy() does not check destination buffer size",
         re.compile(r"\bstrcpy\s*\("), func_name="strcpy"),
    Rule("R003", "CWE-120", "high",
         "strcat() does not check destination buffer size",
         re.compile(r"\bstrcat\s*\("), func_name="strcat"),
    Rule("R004", "CWE-134", "high",
         "sprintf() has no size limit and can overflow the destination buffer",
         re.compile(r"\bsprintf\s*\("), func_name="sprintf"),
    Rule("R005", "CWE-134", "medium",
         "printf() called with a non-literal format string can lead to a format string vulnerability",
         re.compile(r"\bprintf\s*\(\s*[a-zA-Z_]\w*\s*\)"), func_name="printf"),
    Rule("R006", "CWE-78", "high",
         "system() executes a shell command; unsanitized input can lead to command injection",
         re.compile(r"\bsystem\s*\("), func_name="system"),
    Rule("R007", "CWE-676", "medium",
         "scanf() with %s and no width limit can overflow the destination buffer",
         re.compile(r"\bscanf\s*\([^)]*%s"), func_name="scanf"),
    Rule("R008", "CWE-190", "medium",
         "malloc() size computed via multiplication can integer-overflow, allocating less memory than intended",
         re.compile(r"\bmalloc\s*\(\s*\w+\s*\*\s*\w+"), func_name="malloc"),
    Rule("R009", "CWE-416", "low",
         "free() call — verify the pointer is not used again afterwards (use-after-free)",
         re.compile(r"\bfree\s*\("), func_name="free"),
    Rule("R010", "CWE-120", "medium",
         "memcpy()/memmove() without an obvious size check can overflow the destination buffer",
         re.compile(r"\b(memcpy|memmove)\s*\(")),
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
                    func_name=rule.func_name,
                ))
    return findings
