"""3-layer vulnerability detection pipeline: regex -> AST validation -> ML model."""

from __future__ import annotations

from dataclasses import dataclass

from .ast_layer import validate as ast_validate
from .model_layer import ModelPrediction, VulnClassifier, VulnClassifierONNX, get_classifier
from .regex_layer import RegexFinding
from .regex_layer import scan as regex_scan

_classifier: VulnClassifierONNX | VulnClassifier | None = None


def _get_classifier() -> "VulnClassifierONNX | VulnClassifier":
    global _classifier
    if _classifier is None:
        _classifier = get_classifier()
    return _classifier


@dataclass
class PipelineResult:
    code: str
    regex_findings: list[RegexFinding]       # everything Layer 1 flagged
    confirmed_findings: list[RegexFinding]   # subset Layer 2 confirmed via AST
    model_prediction: ModelPrediction        # Layer 3 output
    verdict: str  # "vulnerable" | "needs_review" | "likely_safe"


def analyze(code: str, model_threshold: float = 0.5) -> PipelineResult:
    """Run all 3 layers on a single C/C++ function and combine their signals."""
    findings = regex_scan(code)
    validated = ast_validate(code, findings)
    confirmed = [v.finding for v in validated if v.confirmed]

    prediction = _get_classifier().predict(code)

    if confirmed and prediction.probability >= model_threshold:
        verdict = "vulnerable"
    elif confirmed or prediction.probability >= model_threshold:
        verdict = "needs_review"
    else:
        verdict = "likely_safe"

    return PipelineResult(
        code=code,
        regex_findings=findings,
        confirmed_findings=confirmed,
        model_prediction=prediction,
        verdict=verdict,
    )
