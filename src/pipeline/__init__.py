"""3-layer vulnerability detection pipeline: regex -> AST validation -> ML model."""

from __future__ import annotations

from dataclasses import dataclass, field

from .ast_layer import ValidatedFinding
from .ast_layer import validate as ast_validate
from .model_layer import ModelPrediction, VulnClassifier, VulnClassifierONNX, get_classifier
from .regex_layer import RegexFinding
from .regex_layer import scan as regex_scan

__all__ = [
    "PipelineResult",
    "RegexFinding",
    "ValidatedFinding",
    "ModelPrediction",
    "VulnClassifier",
    "VulnClassifierONNX",
    "analyze",
    "get_classifier",
    "model_available",
]

_classifier: "VulnClassifierONNX | VulnClassifier | None" = None
_classifier_loaded = False

# Layer 1 severities that are strong enough on their own. A confirmed call to
# gets()/strcpy()/strcat()/sprintf()/system() is a deterministic fact about the
# code, not a guess: Layer 2 already proved it is a real call and not a mention
# inside a comment or string. Layer 3 is a *corroborating* signal trained on
# Big-Vul CVE patches, so it under-fires on short, out-of-distribution code
# (a textbook strcpy overflow scores P(vuln) ~= 0.001). Gating a high-severity
# confirmed finding behind the model would silently drop real findings.
_ESCALATE_SEVERITIES = frozenset({"high"})


def _get_classifier() -> "VulnClassifierONNX | VulnClassifier | None":
    """Load Layer 3 once, caching the result — including a None (no model present)."""
    global _classifier, _classifier_loaded
    if not _classifier_loaded:
        _classifier = get_classifier()
        _classifier_loaded = True
    return _classifier


def model_available() -> bool:
    """True when a trained Layer 3 model is loadable (CI reports this)."""
    return _get_classifier() is not None


@dataclass
class PipelineResult:
    code: str
    regex_findings: list[RegexFinding]        # everything Layer 1 flagged
    validated: list[ValidatedFinding]         # Layer 2 verdict per finding
    confirmed_findings: list[RegexFinding]    # subset Layer 2 confirmed via AST
    model_prediction: ModelPrediction | None  # Layer 3 output, None if no model
    verdict: str  # "vulnerable" | "needs_review" | "likely_safe"
    reason: str = ""
    layers_run: list[str] = field(default_factory=list)


def analyze(code: str, model_threshold: float = 0.5) -> PipelineResult:
    """Run all available layers on one C/C++ snippet and combine their signals.

    Layers 1 and 2 always run. Layer 3 runs only when a trained model is
    present; without it the verdict rests on the deterministic layers alone.
    """
    findings = regex_scan(code)
    validated = ast_validate(code, findings)
    confirmed = [v.finding for v in validated if v.confirmed]
    high = [f for f in confirmed if f.severity in _ESCALATE_SEVERITIES]

    layers = ["regex", "ast"]
    classifier = _get_classifier()
    prediction = classifier.predict(code) if classifier is not None else None
    if prediction is not None:
        layers.append("model")

    model_flags = prediction is not None and prediction.probability >= model_threshold

    if high:
        verdict = "vulnerable"
        rules = ", ".join(sorted({f.rule_id for f in high}))
        reason = f"AST-confirmed high-severity finding(s): {rules}"
    elif confirmed and model_flags:
        verdict = "vulnerable"
        reason = (
            f"AST-confirmed finding(s) plus model agreement "
            f"(P={prediction.probability:.3f} >= {model_threshold})"
        )
    elif confirmed:
        verdict = "needs_review"
        rules = ", ".join(sorted({f.rule_id for f in confirmed}))
        reason = f"AST-confirmed medium/low-severity finding(s): {rules}"
    elif model_flags:
        verdict = "needs_review"
        reason = f"model-only signal (P={prediction.probability:.3f} >= {model_threshold})"
    else:
        verdict = "likely_safe"
        reason = "no AST-confirmed finding" + ("" if prediction is None else " and model below threshold")

    return PipelineResult(
        code=code,
        regex_findings=findings,
        validated=validated,
        confirmed_findings=confirmed,
        model_prediction=prediction,
        verdict=verdict,
        reason=reason,
        layers_run=layers,
    )
