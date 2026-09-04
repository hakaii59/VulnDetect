"""Layer 3: fine-tuned GraphCodeBERT binary classifier (vulnerable vs. not).

Two backends, same `ModelPrediction` interface:

- `VulnClassifierONNX` — loads `models/model.onnx` (Phase 6, INT8 quantized).
  Preferred: no PyTorch/CUDA needed at inference time, ~4x smaller on disk —
  the backend `scripts/export_onnx.py` was built for, and what Phase 7's CI
  workflow uses.
- `VulnClassifier` — loads the PyTorch checkpoint directly from
  `models/graphcodebert_finetuned/`. Falls back further to the untrained
  base model if even that checkpoint is missing, so the pipeline stays
  runnable during early development (Phase 5), with a clear warning that
  predictions aren't meaningful in that case.

`get_classifier()` picks ONNX when available, otherwise PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_DEFAULT_MODEL_DIR = _MODELS_DIR / "graphcodebert_finetuned"
_DEFAULT_ONNX_PATH = _MODELS_DIR / "model.onnx"
_BASE_MODEL_NAME = "microsoft/graphcodebert-base"
_MAX_LENGTH = 512


@dataclass
class ModelPrediction:
    label: int  # 0 = not vulnerable, 1 = vulnerable
    probability: float  # P(vulnerable)
    model_source: str


class VulnClassifier:
    """PyTorch backend — loads the checkpoint from notebooks/03_finetune.ipynb.

    Imports torch lazily so importing this module (and using the ONNX
    backend instead) doesn't require a PyTorch install.
    """

    def __init__(self, model_dir: Path | str = _DEFAULT_MODEL_DIR):
        import torch
        from transformers import AutoModelForSequenceClassification

        model_dir = Path(model_dir)
        if model_dir.exists() and any(model_dir.iterdir()):
            load_from = model_dir
            self.source = str(model_dir)
        else:
            print(
                f"[model_layer] WARNING: fine-tuned checkpoint not found at '{model_dir}'. "
                f"Falling back to the base pretrained '{_BASE_MODEL_NAME}' with a randomly "
                f"initialized classification head — predictions are NOT meaningful until "
                f"Phase 4 fine-tuning finishes and the checkpoint is copied here."
            )
            load_from = _BASE_MODEL_NAME
            self.source = f"{_BASE_MODEL_NAME} (untrained fallback)"

        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_from, num_labels=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, code: str) -> ModelPrediction:
        import torch

        enc = self.tokenizer(
            code,
            truncation=True,
            max_length=_MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0].cpu().numpy()
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        label = int(np.argmax(probs))
        return ModelPrediction(label=label, probability=float(probs[1]), model_source=self.source)


class VulnClassifierONNX:
    """ONNX Runtime backend — loads models/model.onnx (Phase 6). No torch needed."""

    def __init__(self, onnx_path: Path | str = _DEFAULT_ONNX_PATH, tokenizer_dir: Path | str = _DEFAULT_MODEL_DIR):
        import onnxruntime as ort  # local import: only needed for this backend

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.source = str(onnx_path)

    def predict(self, code: str) -> ModelPrediction:
        enc = self.tokenizer(
            code,
            truncation=True,
            max_length=_MAX_LENGTH,
            padding="max_length",
            return_tensors="np",
        )
        logits = self.session.run(
            ["logits"], {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        )[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        label = int(np.argmax(probs))
        return ModelPrediction(label=label, probability=float(probs[1]), model_source=self.source)


def get_classifier() -> "VulnClassifierONNX | VulnClassifier":
    """Prefer the ONNX backend when models/model.onnx exists; else fall back to PyTorch."""
    if _DEFAULT_ONNX_PATH.exists():
        return VulnClassifierONNX()
    return VulnClassifier()
