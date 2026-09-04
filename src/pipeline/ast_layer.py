"""Layer 2: Tree-sitter AST validation.

Layer 1 (regex) works on raw text, so it can't tell a real function call from
the same text sitting inside a comment or a string literal. This layer parses
the code into an AST and confirms (or rejects) each regex finding:

- For findings tied to a specific function (`func_name` set) — confirmed only
  if there is an actual `call_expression` with that function name on the same
  line.
- For other findings — confirmed only if the match isn't inside a comment or
  string literal.
"""

from __future__ import annotations

from dataclasses import dataclass

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

from .regex_layer import RegexFinding

CPP_LANGUAGE = Language(tree_sitter_cpp.language())
_parser = Parser(CPP_LANGUAGE)

_SKIP_NODE_TYPES = {"comment", "string_literal", "char_literal", "raw_string_literal"}


@dataclass
class ValidatedFinding:
    finding: RegexFinding
    confirmed: bool
    reason: str


def _collect_skip_ranges(root: Node) -> list[tuple[int, int]]:
    """Byte ranges covered by comments/string literals (don't descend into them)."""
    ranges: list[tuple[int, int]] = []

    def walk(node: Node) -> None:
        if node.type in _SKIP_NODE_TYPES:
            ranges.append((node.start_byte, node.end_byte))
            return
        for child in node.children:
            walk(child)

    walk(root)
    return ranges


def _collect_call_names_by_line(root: Node) -> dict[int, set[str]]:
    """1-indexed line number -> set of function names called on that line."""
    calls: dict[int, set[str]] = {}

    def walk(node: Node) -> None:
        if node.type == "call_expression":
            fn_node = node.child_by_field_name("function")
            if fn_node is not None and fn_node.type == "identifier":
                line = fn_node.start_point[0] + 1
                calls.setdefault(line, set()).add(fn_node.text.decode("utf-8"))
        for child in node.children:
            walk(child)

    walk(root)
    return calls


def _line_start_byte(code: str, line: int) -> int:
    lines = code.splitlines(keepends=True)
    return sum(len(l.encode("utf-8")) for l in lines[: line - 1])


def _in_skip_range(byte_offset: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= byte_offset < end for start, end in ranges)


def validate(code: str, findings: list[RegexFinding]) -> list[ValidatedFinding]:
    tree = _parser.parse(code.encode("utf-8"))
    root = tree.root_node

    if root.has_error:
        # Snippet doesn't parse cleanly (e.g. it's a function body without its
        # surrounding includes/types). Trust the regex layer rather than
        # silently dropping findings we can't actually verify.
        return [
            ValidatedFinding(f, True, "AST has parse errors; regex finding kept as-is")
            for f in findings
        ]

    skip_ranges = _collect_skip_ranges(root)
    calls_by_line = _collect_call_names_by_line(root)

    results: list[ValidatedFinding] = []
    for f in findings:
        if f.func_name is not None:
            called_here = calls_by_line.get(f.line, set())
            if f.func_name in called_here:
                results.append(ValidatedFinding(f, True, f"confirmed: real call to {f.func_name}() on this line"))
            else:
                results.append(ValidatedFinding(
                    f, False,
                    f"rejected: no AST call_expression for {f.func_name}() on this line "
                    f"(likely inside a comment/string, or not actually a call)",
                ))
        else:
            line_text = code.splitlines()[f.line - 1]
            offset = _line_start_byte(code, f.line) + len(line_text[: f.col].encode("utf-8"))
            if _in_skip_range(offset, skip_ranges):
                results.append(ValidatedFinding(f, False, "rejected: match is inside a comment/string literal"))
            else:
                results.append(ValidatedFinding(f, True, "confirmed: match is in real code"))

    return results
