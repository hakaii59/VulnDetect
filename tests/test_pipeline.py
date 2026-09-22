"""Tests for the 3-layer pipeline: regex rules, AST validation, verdict logic."""

from __future__ import annotations

import pytest

from src.pipeline import analyze
from src.pipeline.ast_layer import validate as ast_validate
from src.pipeline.regex_layer import RULES, scan as regex_scan

VULNERABLE_C = """
#include <string.h>
void copy_name(const char *input) {
    char name[16];
    strcpy(name, input);
}
"""

# The same dangerous token, but only ever as text: Layer 2 must reject both.
SAFE_CPP = """
#include <string>
void greet(const std::string &name) {
    // Avoid strcpy(): std::string manages its own memory.
    const char *doc = "call strcpy() here";
    std::string msg = "hello, " + name;
}
"""

FREE_ONLY_C = """
#include <stdlib.h>
void release(void *p) {
    free(p);
}
"""


class TestRegexLayer:
    def test_rule_ids_are_unique(self):
        ids = [r.rule_id for r in RULES]
        assert len(ids) == len(set(ids))

    def test_severities_are_known(self):
        assert {r.severity for r in RULES} <= {"high", "medium", "low"}

    def test_cwe_ids_are_well_formed(self):
        for rule in RULES:
            assert rule.cwe.startswith("CWE-"), rule.rule_id
            assert rule.cwe.removeprefix("CWE-").isdigit(), rule.rule_id

    def test_flags_strcpy(self):
        findings = regex_scan(VULNERABLE_C)
        assert "R002" in {f.rule_id for f in findings}

    def test_line_numbers_are_1_indexed(self):
        findings = regex_scan(VULNERABLE_C)
        strcpy_finding = next(f for f in findings if f.rule_id == "R002")
        assert VULNERABLE_C.splitlines()[strcpy_finding.line - 1].strip().startswith("strcpy(")

    def test_matches_inside_comments_too(self):
        """Layer 1 is text-only on purpose — filtering them is Layer 2's job."""
        assert any(f.rule_id == "R002" for f in regex_scan(SAFE_CPP))


class TestAstLayer:
    def test_confirms_a_real_call(self):
        validated = ast_validate(VULNERABLE_C, regex_scan(VULNERABLE_C))
        strcpy_results = [v for v in validated if v.finding.rule_id == "R002"]
        assert strcpy_results and all(v.confirmed for v in strcpy_results)

    def test_rejects_comment_and_string_mentions(self):
        validated = ast_validate(SAFE_CPP, regex_scan(SAFE_CPP))
        strcpy_results = [v for v in validated if v.finding.rule_id == "R002"]
        assert strcpy_results, "fixture should produce regex hits to reject"
        assert not any(v.confirmed for v in strcpy_results)

    def test_keeps_findings_when_the_snippet_does_not_parse(self):
        """Unparseable snippets fall back to the regex verdict rather than
        silently dropping findings we cannot verify."""
        broken = "void f( {{{ strcpy(a, b);"
        findings = regex_scan(broken)
        validated = ast_validate(broken, findings)
        assert validated and all(v.confirmed for v in validated)


class TestVerdictWithoutModel:
    """Layer 3 is optional; the deterministic layers must stand alone."""

    def test_high_severity_confirmed_is_vulnerable(self, no_model):
        result = analyze(VULNERABLE_C)
        assert result.verdict == "vulnerable"
        assert result.layers_run == ["regex", "ast"]
        assert result.model_prediction is None

    def test_rejected_findings_are_likely_safe(self, no_model):
        assert analyze(SAFE_CPP).verdict == "likely_safe"

    def test_low_severity_only_is_needs_review(self, no_model):
        result = analyze(FREE_ONLY_C)
        assert result.verdict == "needs_review"
        assert "R009" in {f.rule_id for f in result.confirmed_findings}


class TestVerdictWithModel:
    def test_high_severity_wins_even_when_model_disagrees(self, fake_model):
        """The regression that made the gate unfireable: a textbook strcpy
        overflow scores P(vuln) ~= 0.001 under a Big-Vul-trained model, so
        requiring model agreement suppressed a confirmed finding."""
        fake_model.probability = 0.001
        result = analyze(VULNERABLE_C)
        assert result.verdict == "vulnerable"
        assert result.layers_run == ["regex", "ast", "model"]

    def test_low_severity_plus_model_escalates(self, fake_model):
        fake_model.probability = 0.99
        assert analyze(FREE_ONLY_C).verdict == "vulnerable"

    def test_model_alone_is_only_needs_review(self, fake_model):
        fake_model.probability = 0.99
        result = analyze(SAFE_CPP)
        assert result.verdict == "needs_review"
        assert result.confirmed_findings == []

    def test_clean_code_stays_safe(self, fake_model):
        fake_model.probability = 0.01
        assert analyze("int add(int a, int b) { return a + b; }").verdict == "likely_safe"

    @pytest.mark.parametrize("prob", [0.49, 0.5, 0.51])
    def test_threshold_boundary_is_inclusive(self, fake_model, prob):
        fake_model.probability = prob
        result = analyze(SAFE_CPP, model_threshold=0.5)
        expected = "needs_review" if prob >= 0.5 else "likely_safe"
        assert result.verdict == expected
