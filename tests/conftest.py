"""Shared test fixtures.

Every test here runs WITHOUT a trained model: CI installs only
requirements-ci.txt and may have no models/ directory at all. Layer 3 is
optional by design (see src/pipeline/model_layer.get_classifier), so the
deterministic layers are what the tests pin down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Both the package (src.*) and the standalone scripts must be importable
# regardless of which test runs first.
for _path in (ROOT, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture()
def no_model(monkeypatch):
    """Force the 2-layer path (regex + AST) regardless of what is on disk."""
    import src.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_classifier", None)
    monkeypatch.setattr(pipeline, "_classifier_loaded", True)
    return pipeline


@pytest.fixture()
def fake_model(monkeypatch):
    """Install a stub Layer 3 returning a caller-chosen P(vulnerable)."""
    import src.pipeline as pipeline
    from src.pipeline.model_layer import ModelPrediction

    class StubClassifier:
        probability = 0.0

        def predict(self, code: str) -> ModelPrediction:
            return ModelPrediction(
                label=int(self.probability >= 0.5),
                probability=self.probability,
                model_source="stub",
            )

    stub = StubClassifier()
    monkeypatch.setattr(pipeline, "_classifier", stub)
    monkeypatch.setattr(pipeline, "_classifier_loaded", True)
    return stub
