"""Evaluate a fine-tuned checkpoint on the val and test splits, locally.

Evaluation is pure inference: no gradients, no optimizer state, so it needs a
fraction of the memory training does and runs fine on a small GPU — or on CPU
if you are patient. Keeping it out of the notebook means the numbers can be
reproduced without Colab, which matters when its free GPU quota runs out
mid-project.

The decision threshold is swept on **validation only** and then applied
unchanged to test; choosing it on test would be fitting to the set being
reported. Both splits are reported at 0.5 and at the tuned value.

Measured throughput on this machine (batch 16, seq 512):
    RTX 3050 Ti (fp16)  ~ minutes for both splits
    ONNX INT8 on CPU    ~ 260 ms/function  -> ~2.4 h for 32,726 functions
    PyTorch fp32 on CPU ~ 310 ms/function  -> ~2.8 h

Usage:
    python scripts/evaluate.py                        # torch, GPU if available
    python scripts/evaluate.py --backend onnx         # models/model.onnx, CPU
    python scripts/evaluate.py --limit 2000           # quick sanity check
    python scripts/evaluate.py --data-dir data/processed_project
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "models" / "graphcodebert_finetuned"
ONNX_PATH = ROOT / "models" / "model.onnx"
MAX_LENGTH = 512
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)


def metrics_at(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Score probabilities at one decision threshold.

    ROC-AUC is threshold-independent and is repeated in every row so each row
    stands alone.
    """
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")  # only one class present
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
    }


def _clean(d: dict) -> dict:
    """NaN is valid Python but not valid JSON."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def predict_torch(texts: list[str], batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    use_amp = device.type == "cuda"
    print(f"  backend: torch on {device}" + (" (fp16 autocast)" if use_amp else ""))

    out = np.empty(len(texts), dtype=np.float64)
    started = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt"
            ).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            out[i : i + len(batch)] = probs.cpu().numpy()
            _progress(i + len(batch), len(texts), started)
    print()
    return out


def predict_onnx(texts: list[str], batch_size: int) -> np.ndarray:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"{ONNX_PATH} not found — run scripts/export_onnx.py first.")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    print(f"  backend: onnxruntime on CPU ({ONNX_PATH.name})")

    out = np.empty(len(texts), dtype=np.float64)
    started = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="np"
        )
        logits = session.run(
            ["logits"], {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        )[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        out[i : i + len(batch)] = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
        _progress(i + len(batch), len(texts), started)
    print()
    return out


def _progress(done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed else 0
    eta = (total - done) / rate if rate else 0
    print(f"\r    {done:,}/{total:,}  {rate:6.1f} func/s  ETA {eta / 60:5.1f} min", end="", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only N rows per split")
    parser.add_argument("--out", default=None, help="where to write metrics.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    if not any(CHECKPOINT_DIR.glob("*.safetensors")):
        print(
            f"No checkpoint in {CHECKPOINT_DIR}.\n"
            "Download models/graphcodebert_finetuned/ from Drive first.",
            file=sys.stderr,
        )
        return 1

    predict = predict_torch if args.backend == "torch" else predict_onnx

    splits = {}
    for name in ("val", "test"):
        df = pd.read_parquet(data_dir / f"{name}.parquet")
        if args.limit:
            df = df.sample(min(args.limit, len(df)), random_state=42).reset_index(drop=True)
        splits[name] = df
        print(f"{name}: {len(df):,} functions, {int(df['vul'].sum()):,} vulnerable ({df['vul'].mean():.2%})")

    probs, labels = {}, {}
    for name, df in splits.items():
        print(f"\nScoring {name} ...")
        probs[name] = predict([str(c) for c in df["func_before"]], args.batch_size)
        labels[name] = df["vul"].to_numpy()

    # Tune on validation only, then apply that threshold to test unchanged.
    val_f1s = [metrics_at(labels["val"], probs["val"], t)["f1"] for t in THRESHOLD_GRID]
    best_threshold = float(THRESHOLD_GRID[int(np.argmax(val_f1s))])
    baseline_f1 = metrics_at(labels["val"], probs["val"], 0.5)["f1"]
    print(f"\nBest threshold on val: {best_threshold:.2f} (F1 {max(val_f1s):.4f}, vs {baseline_f1:.4f} at 0.50)")

    rows = []
    for name in ("val", "test"):
        for t in (0.5, best_threshold):
            rows.append({"split": name, **metrics_at(labels[name], probs[name], t)})
    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))

    payload = {
        "checkpoint": str(CHECKPOINT_DIR),
        "data_dir": str(data_dir),
        "backend": args.backend,
        "max_length": MAX_LENGTH,
        "limit": args.limit,
        "best_threshold": best_threshold,
        "val": _clean(metrics_at(labels["val"], probs["val"], 0.5)),
        "val_tuned": _clean(metrics_at(labels["val"], probs["val"], best_threshold)),
        "test": _clean(metrics_at(labels["test"], probs["test"], 0.5)),
        "test_tuned": _clean(metrics_at(labels["test"], probs["test"], best_threshold)),
    }
    out_path = Path(args.out) if args.out else CHECKPOINT_DIR / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
