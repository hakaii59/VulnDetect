"""Layer 3: fine-tuned GraphCodeBERT binary classifier (vulnerable vs. not).

Two backends, same `ModelPrediction` interface:

- `VulnClassifierONNX` — loads `models/model.onnx` (Phase 6, INT8 quantized).
  Preferred: no PyTorch/CUDA needed at inference time, ~4x smaller on disk —
  the backend `scripts/export_onnx.py` was built for, and what Phase 7's CI
  workflow uses when a model is available.
- `VulnClassifier` — loads the PyTorch checkpoint directly from
  `models/graphcodebert_finetuned/`.

`get_classifier()` picks ONNX when available, else the PyTorch checkpoint,
and returns **None** when neither is present. Layer 3 is an *optional*
corroborating signal: `analyze()` runs the deterministic regex + AST layers
with or without it, so a fresh clone (and CI with no model configured) still
works. Earlier versions silently fell back to the untrained base model, which
produced meaningless probabilities that looked real — that fallback is now
opt-in via `allow_untrained=True`.
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

# The tokenizer files VulnClassifierONNX needs next to model.onnx. A fast
# tokenizer only needs tokenizer.json, but AutoTokenizer reads tokenizer_config
# to pick the class, so both must be present.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


@dataclass
class ModelPrediction:
    label: int  # 0 = not vulnerable, 1 = vulnerable
    probability: float  # P(vulnerable)
    model_source: str


def _softmax_prob(logits: np.ndarray) -> tuple[int, float]:
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return int(np.argmax(probs)), float(probs[1])


def has_tokenizer(model_dir: Path | str = _DEFAULT_MODEL_DIR) -> bool:
    model_dir = Path(model_dir)
    return all((model_dir / name).is_file() for name in TOKENIZER_FILES)


def has_checkpoint(model_dir: Path | str = _DEFAULT_MODEL_DIR) -> bool:
    """True when the directory holds actual model weights (not just tokenizer files)."""
    model_dir = Path(model_dir)
    return any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))


class VulnClassifier:
    """PyTorch backend — loads the checkpoint from notebooks/03_finetune.ipynb.

    Imports torch lazily so importing this module (and using the ONNX
    backend instead) doesn't require a PyTorch install.
    """

    def __init__(
        self,
        model_dir: Path | str = _DEFAULT_MODEL_DIR,
        allow_untrained: bool = False,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification

        model_dir = Path(model_dir)
        if has_checkpoint(model_dir):
            load_from = model_dir
            self.source = str(model_dir)
        elif allow_untrained:
            print(
                f"[model_layer] WARNING: fine-tuned checkpoint not found at '{model_dir}'. "
                f"Falling back to the base pretrained '{_BASE_MODEL_NAME}' with a randomly "
                f"initialized classification head — predictions are NOT meaningful."
            )
            load_from = _BASE_MODEL_NAME
            self.source = f"{_BASE_MODEL_NAME} (untrained fallback)"
        else:
            raise FileNotFoundError(
                f"No fine-tuned checkpoint in '{model_dir}'. Run Phase 4 "
                f"(notebooks/03_finetune.ipynb), or pass allow_untrained=True "
                f"to load the untrained base model for smoke-testing only."
            )

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
        label, prob = _softmax_prob(logits)
        return ModelPrediction(label=label, probability=prob, model_source=self.source)


class VulnClassifierONNX:
    """ONNX Runtime backend — loads models/model.onnx (Phase 6). No torch needed."""

    def __init__(
        self,
        onnx_path: Path | str = _DEFAULT_ONNX_PATH,
        tokenizer_dir: Path | str = _DEFAULT_MODEL_DIR,
    ):
        import onnxruntime as ort  # local import: only needed for this backend

        tokenizer_dir = Path(tokenizer_dir)
        if not has_tokenizer(tokenizer_dir):
            raise FileNotFoundError(
                f"model.onnx needs its tokenizer files ({', '.join(TOKENIZER_FILES)}) "
                f"in '{tokenizer_dir}', but they are missing."
            )
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
        label, prob = _softmax_prob(logits)
        return ModelPrediction(label=label, probability=prob, model_source=self.source)


def get_classifier() -> "VulnClassifierONNX | VulnClassifier | None":
    """Best available Layer 3 backend, or None when no trained model is present.

    Order: ONNX (models/model.onnx + tokenizer) -> PyTorch checkpoint -> None.
    """
    if _DEFAULT_ONNX_PATH.exists() and has_tokenizer():
        return VulnClassifierONNX()
    if has_checkpoint():
        return VulnClassifier()
    return None
