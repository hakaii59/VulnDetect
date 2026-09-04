"""Fetch the ONNX model + tokenizer files a fresh CI runner needs.

Background
----------
`models/model.onnx` (~125 MB) and `models/graphcodebert_finetuned/`
(weights ~475 MB) are ignored by git (see .gitignore), so a fresh checkout
on GitHub Actions has no model. This script downloads the small files the
runtime needs from Hugging Face Hub and points the pipeline at them via an
env var.

Design choice
-------------
`model.onnx` is itself uploaded here as a workflow artifact from your machine
(that is the job of the manual "upload" workflow). To keep the CI job
self-contained AND to make the repo useful to other people cloning it, this
script actually rebuilds a runnable INT8 `model.onnx` *on the runner*:

  1. clone `benjis/bigvul-model` (or your configured repo) -> PyTorch weights
  2. tokenizer files from `microsoft/graphcodebert-base`
  3. export fp32 -> dynamic-quantize INT8 -> `models/model.onnx`
     (re-using the exact logic of scripts/export_onnx.py, but lightweight:
     no eval on the test set, no torch benchmark)

Requires the heavy deps (torch, transformers, onnx, onnxruntime, tree-sitter,
tree-sitter-cpp) -> see requirements-ci.txt.

Env overrides (all optional):
  HF_MODEL_REPO      default "benjis/bigvul-model"
  HF_TOKENIZER_REPO  default "microsoft/graphcodebert-base"
  MODEL_CACHE_DIR    default "$RUNNER_TEMP/hf-cache"
  HF_HOME            if you already have a local cache, point here to reuse it
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / "models" / "model.onnx"
CHECKPOINT_DIR = ROOT / "models" / "graphcodebert_finetuned"
MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", Path(tempfile.gettempdir()) / "hf-cache"))

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "benjis/bigvul-model")
HF_TOKENIZER_REPO = os.environ.get("HF_TOKENIZER_REPO", "microsoft/graphcodebert-base")
MAX_LENGTH = 512


def fetch_tokenizer() -> Path:
    """Download tokenizer files next to the ONNX model.

    VulnClassifierONNX loads the tokenizer from `models/graphcodebert_finetuned/`
    (see src/pipeline/model_layer.py). On a fresh runner that directory is empty
    (or, in the git-lfs partial-commit case, may already hold model.onnx), so we
    fetch the small tokenizer/config files into it.
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading tokenizer from '{HF_TOKENIZER_REPO}' ...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_TOKENIZER_REPO,
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",       # roberta repos ship these
            "merges.txt",
            "config.json",      # used by AutoTokenizer to pick the right class
        ],
        cache_dir=str(MODEL_CACHE_DIR),
        local_dir=str(CHECKPOINT_DIR),
    )
    print(f"Tokenizer files in: {CHECKPOINT_DIR}")
    return CHECKPOINT_DIR


def export_and_quantize() -> None:
    """Clone the PyTorch checkpoint and rebuild the INT8 ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint_dir = MODEL_CACHE_DIR / "checkpoint"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    print(f"Downloading fine-tuned checkpoint from '{HF_MODEL_REPO}' ...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=["model.safetensors", "config.json"],
        cache_dir=str(MODEL_CACHE_DIR),
        local_dir=str(checkpoint_dir),
        local_dir_use_symlinks=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_REPO, cache_dir=str(MODEL_CACHE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), num_labels=2, torch_dtype="auto", low_cpu_mem_usage=True
    )
    model.eval()

    dummy = tokenizer(
        "int main() { char buf[16]; strcpy(buf, argv[1]); return 0; }",
        truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt",
    )

    fp32_path = ONNX_PATH.with_suffix(".fp32.onnx")
    import torch

    print(f"Exporting fp32 ONNX -> {fp32_path.name} ...")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        dynamo=False,
    )

    print(f"Quantizing dynamic INT8 -> {ONNX_PATH.name} ...")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(ONNX_PATH),
        weight_type=QuantType.QInt8,
    )
    fp32_path.unlink(missing_ok=True)
    print(f"ONNX model ready: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    fetch_tokenizer()
    export_and_quantize()


if __name__ == "__main__":
    main()
