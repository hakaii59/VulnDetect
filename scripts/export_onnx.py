"""Export the fine-tuned GraphCodeBERT checkpoint to ONNX (INT8 dynamic quantized).

ONNX Runtime lets the model run without the full PyTorch/transformers stack
at inference time — much lighter to install in CI (Phase 7). Dynamic INT8
quantization on top shrinks the model (~475MB -> ~125MB) and speeds up CPU
inference, at the cost of a small, measured accuracy trade-off.

This script exports to fp32 ONNX, quantizes to INT8, then verifies the
quantized model's *predicted labels* against the original PyTorch model on
real samples from the test set (not just raw logit closeness — quantization
shifts logits by design, so what matters is whether decisions still agree),
and benchmarks inference speed.

Usage:
    python scripts/export_onnx.py
"""

from __future__ import annotations

import time
from pathlib import Path

import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "models" / "graphcodebert_finetuned"
ONNX_FP32_PATH = ROOT / "models" / "model_fp32.onnx"
ONNX_PATH = ROOT / "models" / "model.onnx"
TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"
MAX_LENGTH = 512


def export(tokenizer: AutoTokenizer, model: torch.nn.Module) -> None:
    dummy = tokenizer(
        "int main() { char buf[16]; strcpy(buf, argv[1]); return 0; }",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )

    print(f"Exporting to {ONNX_FP32_PATH} ...")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(ONNX_FP32_PATH),
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
    print(f"Saved: {ONNX_FP32_PATH} ({ONNX_FP32_PATH.stat().st_size / 1e6:.1f} MB)")

    print(f"\nQuantizing (dynamic INT8) to {ONNX_PATH} ...")
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_PATH),
        weight_type=QuantType.QInt8,
    )
    print(f"Saved: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")


def load_eval_samples(n_per_class: int = 10) -> pd.DataFrame:
    df = pd.read_parquet(TEST_PARQUET)
    vuln = df[df["vul"] == 1].sample(n_per_class, random_state=7)
    safe = df[df["vul"] == 0].sample(n_per_class, random_state=7)
    return pd.concat([vuln, safe]).reset_index(drop=True)


def verify(tokenizer: AutoTokenizer, torch_model: torch.nn.Module, samples: pd.DataFrame) -> ort.InferenceSession:
    print(f"\nVerifying quantized ONNX predictions against PyTorch on {len(samples)} real test samples ...")
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    agree = 0
    for _, row in samples.iterrows():
        enc = tokenizer(
            row["func_before"], truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt"
        )
        with torch.no_grad():
            torch_pred = int(torch_model(**enc).logits.argmax(dim=-1).item())

        onnx_logits = session.run(
            ["logits"],
            {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()},
        )[0]
        onnx_pred = int(onnx_logits.argmax(axis=-1)[0])

        agree += int(torch_pred == onnx_pred)

    agreement_rate = agree / len(samples)
    print(f"Label agreement (PyTorch vs quantized ONNX): {agree}/{len(samples)} ({agreement_rate:.0%})")
    assert agreement_rate >= 0.9, "Quantized ONNX model diverges too much from PyTorch predictions!"
    print("PASS: quantized ONNX predictions agree with PyTorch on real samples.")
    return session


def benchmark(torch_model: torch.nn.Module, session: ort.InferenceSession, tokenizer: AutoTokenizer, n_runs: int = 30) -> None:
    print(f"\nBenchmarking inference speed ({n_runs} runs, CPU, batch_size=1, seq_len={MAX_LENGTH}) ...")
    enc = tokenizer(
        "int main() { char buf[16]; strcpy(buf, argv[1]); return 0; }",
        truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt",
    )

    with torch.no_grad():
        torch_model(**enc)  # warmup
        start = time.perf_counter()
        for _ in range(n_runs):
            torch_model(**enc)
        torch_time = (time.perf_counter() - start) / n_runs

    onnx_inputs = {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()}
    session.run(["logits"], onnx_inputs)  # warmup
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(["logits"], onnx_inputs)
    onnx_time = (time.perf_counter() - start) / n_runs

    print(f"PyTorch (fp32)     : {torch_time * 1000:.1f} ms/inference")
    print(f"ONNX (INT8 quant)  : {onnx_time * 1000:.1f} ms/inference")
    print(f"Speedup            : {torch_time / onnx_time:.2f}x")


if __name__ == "__main__":
    print(f"Loading checkpoint from {CHECKPOINT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR, num_labels=2)
    model.eval()

    export(tokenizer, model)
    samples = load_eval_samples()
    session = verify(tokenizer, model, samples)
    benchmark(model, session, tokenizer)
