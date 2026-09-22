"""Fetch (or rebuild) the ONNX model + tokenizer files a fresh CI runner needs.

Background
----------
`models/model.onnx` (~125 MB) and `models/graphcodebert_finetuned/`
(weights ~475 MB) are git-ignored, so a fresh checkout on GitHub Actions has
no model. This script populates them.

Two modes
---------
  --tokenizer-only   download just the tokenizer files into
                     models/graphcodebert_finetuned/. Needed whenever
                     model.onnx comes from somewhere else (a release asset via
                     CI_MODEL_URL), because VulnClassifierONNX loads its
                     tokenizer from that directory.

  (default)          tokenizer + rebuild a runnable INT8 model.onnx on the
                     runner from a fine-tuned checkpoint on the Hub:
                       1. snapshot_download(HF_MODEL_REPO) -> PyTorch weights
                       2. export fp32 ONNX
                       3. dynamic-quantize to INT8 -> models/model.onnx
                     Same logic as scripts/export_onnx.py, minus the test-set
                     eval and the benchmark.

The default mode needs the heavy deps (torch, onnx) -> requirements-fetch.txt.
`--tokenizer-only` needs only requirements-ci.txt.

Env vars:
  HF_MODEL_REPO      REQUIRED for a rebuild — the Hub repo holding your
                     fine-tuned checkpoint (config.json + model.safetensors).
                     There is deliberately no default: a wrong default fails
                     deep inside the download with a confusing 401.
  HF_TOKENIZER_REPO  default "microsoft/graphcodebert-base"
  MODEL_CACHE_DIR    default "<tempdir>/hf-cache"
  HF_TOKEN           set for a private HF_MODEL_REPO
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / "models" / "model.onnx"
CHECKPOINT_DIR = ROOT / "models" / "graphcodebert_finetuned"
MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", Path(tempfile.gettempdir()) / "hf-cache"))

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()
HF_TOKENIZER_REPO = os.environ.get("HF_TOKENIZER_REPO", "microsoft/graphcodebert-base")
MAX_LENGTH = 512

# Files AutoTokenizer needs. tokenizer.json alone drives the fast tokenizer;
# vocab.json/merges.txt are the slow-tokenizer fallback and are small enough
# to take along. config.json lets AutoTokenizer pick the right class.
TOKENIZER_PATTERNS = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"]


def fetch_tokenizer() -> Path:
    """Download tokenizer files into models/graphcodebert_finetuned/."""
    from huggingface_hub import snapshot_download

    print(f"Downloading tokenizer from '{HF_TOKENIZER_REPO}' ...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_TOKENIZER_REPO,
        allow_patterns=TOKENIZER_PATTERNS,
        cache_dir=str(MODEL_CACHE_DIR),
        local_dir=str(CHECKPOINT_DIR),
    )
    got = sorted(p.name for p in CHECKPOINT_DIR.glob("*") if p.is_file())
    print(f"Tokenizer files in {CHECKPOINT_DIR}: {got}")
    return CHECKPOINT_DIR


def export_and_quantize() -> None:
    """Download the fine-tuned checkpoint and rebuild the INT8 ONNX model."""
    import torch
    from huggingface_hub import snapshot_download
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint_dir = MODEL_CACHE_DIR / "checkpoint"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    print(f"Downloading fine-tuned checkpoint from '{HF_MODEL_REPO}' ...")
    snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=["model.safetensors", "pytorch_model.bin", "config.json"],
        cache_dir=str(MODEL_CACHE_DIR),
        local_dir=str(checkpoint_dir),
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_REPO, cache_dir=str(MODEL_CACHE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), num_labels=2, low_cpu_mem_usage=True
    )
    model.eval()

    dummy = tokenizer(
        "int main() { char buf[16]; strcpy(buf, argv[1]); return 0; }",
        truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt",
    )

    fp32_path = ONNX_PATH.with_name("model_fp32.onnx")
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="download only the tokenizer files (model.onnx supplied elsewhere)",
    )
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    fetch_tokenizer()

    if args.tokenizer_only:
        return 0

    if not HF_MODEL_REPO:
        print(
            "ERROR: HF_MODEL_REPO is not set, so there is no checkpoint to rebuild "
            "model.onnx from.\n"
            "Either:\n"
            "  - set the CI_MODEL_URL repository variable to a direct URL of your "
            "INT8 model.onnx (e.g. a GitHub Release asset), or\n"
            "  - set the HF_MODEL_REPO repository variable to the Hub repo holding "
            "your fine-tuned checkpoint.\n"
            "Without either, CI runs the deterministic regex + AST layers only.",
            file=sys.stderr,
        )
        return 1

    export_and_quantize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
