"""Every rule must actually fire on the construct it claims to detect.

Three of the original ten rules never matched anything on 163,636 real
functions because their regexes were written too narrowly — a silent failure
that made Layer 1 a third smaller than it looked. Each rule now carries a
positive example, so a rule that stops matching fails the build.
"""

from __future__ import annotations

import pytest

from src.pipeline.ast_layer import validate as ast_validate
from src.pipeline.regex_layer import RULES, scan

RULES_BY_ID = {r.rule_id: r for r in RULES}

# One snippet per rule that the rule must flag.
POSITIVE = {
    "R001": 'void f(char *b) { gets(b); }',
    "R002": 'void f(char *d, char *s) { strcpy(d, s); }',
    "R003": 'void f(char *d, char *s) { strcat(d, s); }',
    "R004": 'void f(char *d, int n) { sprintf(d, "%d", n); }',
    "R005": 'void f(char *fmt) { printf(fmt); }',
    "R006": 'void f(char *cmd) { system(cmd); }',
    "R007": 'void f(char *b) { scanf("%s", b); }',
    "R008": 'void *f(int n, int w) { return malloc(n * w); }',
    "R009": 'void f(void *p) { free(p); }',
    "R010": 'void f(char *d, char *s, int n) { memcpy(d, s, n); }',
    "R011": 'void f(char *d, char *s, int n) { strncpy(d, s, n); }',
    "R012": 'void f(char *d, char *s, int n) { strncat(d, s, n); }',
    "R013": 'void f(int n) { char *p = alloca(n); }',
    "R014": 'void f(void *p, int n) { p = realloc(p, n); }',
    "R015": 'void *f(int n, int hdr) { return malloc(n + hdr); }',
    "R016": 'void f(size_t buflen) { int n = (int) buflen; }',
    "R017": 'void f(char *s) { int n = atoi(s); }',
    "R018": 'void f(char *s) { char *t = strtok(s, ","); }',
}

# Constructs each rule must NOT flag, to keep it from becoming a blanket match.
NEGATIVE = {
    "R002": 'void f(char *d, char *s) { strlcpy(d, s, 8); }',
    "R005": 'void f(int n) { printf("%d", n); }',
    "R006": 'void f(void) { int systemic = 1; }',
    "R008": 'void *f(int n) { return malloc(n); }',
    "R010": 'void f(void) { int memcpy_len = 0; }',
    "R015": 'void *f(int n) { return malloc(n); }',
    "R016": 'void f(int x) { int n = (int) x; }',
}


class TestRuleDefinitions:
    def test_rule_ids_are_unique(self):
        ids = [r.rule_id for r in RULES]
        assert len(ids) == len(set(ids))

    def test_every_rule_has_a_positive_example(self):
        """A new rule without an example would be untested."""
        assert set(RULES_BY_ID) == set(POSITIVE)

    def test_severities_are_known(self):
        assert {r.severity for r in RULES} <= {"high", "medium", "low"}

    def test_cwe_ids_are_well_formed(self):
        for rule in RULES:
            assert rule.cwe.startswith("CWE-"), rule.rule_id
            assert rule.cwe.removeprefix("CWE-").isdigit(), rule.rule_id

    def test_messages_explain_the_risk(self):
        for rule in RULES:
            assert len(rule.message) > 30, f"{rule.rule_id} message is too terse"


@pytest.mark.parametrize("rule_id", sorted(POSITIVE))
def test_rule_fires_on_its_own_example(rule_id):
    hits = {f.rule_id for f in scan(POSITIVE[rule_id])}
    assert rule_id in hits, f"{rule_id} did not match its own positive example"


@pytest.mark.parametrize("rule_id", sorted(NEGATIVE))
def test_rule_does_not_fire_on_lookalike(rule_id):
    hits = {f.rule_id for f in scan(NEGATIVE[rule_id])}
    assert rule_id not in hits, f"{rule_id} matched a construct it should ignore"


@pytest.mark.parametrize("rule_id", sorted(POSITIVE))
def test_call_rules_survive_ast_validation(rule_id):
    """A rule naming functions must name them as Tree-sitter sees them.

    Getting `func_names` wrong would make Layer 2 reject every finding from
    that rule, silently disabling it.
    """
    rule = RULES_BY_ID[rule_id]
    if not rule.func_names:
        pytest.skip(f"{rule_id} is not a call-based rule")
    code = POSITIVE[rule_id]
    validated = [v for v in ast_validate(code, scan(code)) if v.finding.rule_id == rule_id]
    assert validated, f"{rule_id} produced no finding to validate"
    assert all(v.confirmed for v in validated), [v.reason for v in validated]


class TestMultiNameRules:
    """R006 covers a family: system, popen and the exec* variants."""

    @pytest.mark.parametrize("call", ["system(cmd)", "popen(cmd, \"r\")", "execl(cmd, cmd, 0)", "execvp(cmd, args)"])
    def test_each_family_member_is_confirmed(self, call):
        code = f"void f(char *cmd, char **args) {{ {call}; }}"
        validated = [v for v in ast_validate(code, scan(code)) if v.finding.rule_id == "R006"]
        assert validated and all(v.confirmed for v in validated), call

    def test_comment_mention_is_still_rejected(self):
        code = "void f(void) { /* never call system() here */ int x = 1; }"
        validated = [v for v in ast_validate(code, scan(code)) if v.finding.rule_id == "R006"]
        assert validated, "the regex layer should still match the comment"
        assert not any(v.confirmed for v in validated)
