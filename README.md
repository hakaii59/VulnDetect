# VulnDetect-CPP 🔍

**A 3-layer vulnerability detection pipeline for C/C++ source code** — regex
pattern matching → tree-sitter AST validation → a fine-tuned GraphCodeBERT
binary classifier, deployed as a CI gate that blocks vulnerable code before it
gets merged.

> **Portfolio project** — built end-to-end from raw dataset to ONNX inference
> and GitHub Actions, with every step documented and reproducible.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-ffd21e)](https://huggingface.co/microsoft/graphcodebert-base)
[![ONNX](https://img.shields.io/badge/ONNX-exported-005ced)](https://onnx.ai/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088ff)](./.github/workflows/vuln_check.yml)

---

## Table of contents

- [What it does](#what-it-does)
- [The 3-layer pipeline](#the-3-layer-pipeline)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Using the pipeline](#using-the-pipeline)
- [Dataset](#dataset)
- [Results](#results)
- [ONNX export](#onnx-export)
- [CI/CD](#cicd)
- [Tests](#tests)
- [Limitations](#limitations)

---

## What it does

Given a C or C++ function, the pipeline classifies it as **`vulnerable`**,
**`needs_review`**, or **`likely_safe`** by combining three independent
signals — so a fast-but-noisy regex layer is corrected by a structural AST
layer, and both are backed by a neural classifier trained on real-world
vulnerabilities.

```text
┌─────────────────────────────────────────────────────────────────┐
│                         C/C++ source code                        │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │  LAYER 1 — Regex            (~10 ms)          │
        │  10 curated rules for classic CWE patterns    │
        │  (strcpy, sprintf, system, scanf %s, ...)     │
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │  LAYER 2 — Tree-sitter AST      (~50 ms)      │
        │  Parses the code and rejects regex hits that  │
        │  sit inside comments / string literals        │
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │  LAYER 3 — GraphCodeBERT       (~100 ms)      │
        │  Fine-tuned binary classifier (ONNX INT8)     │
        │  outputs P(vulnerable) in [0, 1]              │
        └───────────────────────┬───────────────────────┘
                                ▼
                 ┌─────────────────────────────┐
                 │  FINAL VERDICT              │
                 │  vulnerable / needs_review  │
                 │  / likely_safe              │
                 └─────────────────────────────┘
```

### Verdict logic

| Layer 1+2 confirmed | P(vulnerable) ≥ 0.5 | Verdict        |
|---------------------|---------------------|----------------|
| ✅                  | ✅                  | `vulnerable`   |
| ✅                  | ❌                  | `needs_review` |
| ❌                  | ✅                  | `needs_review` |
| ❌                  | ❌                  | `likely_safe`  |

---

## The 3-layer pipeline

### Layer 1 — Regex (`src/pipeline/regex_layer.py`)

Ten hand-written regex rules map classic C/C++ bugs to their CWE IDs:

| Rule | Pattern | CWE | Severity |
|------|---------|-----|----------|
| R001 | `gets()` | CWE-120 (buffer overflow) | high |
| R002 | `strcpy()` | CWE-120 | high |
| R003 | `strcat()` | CWE-120 | high |
| R004 | `sprintf()` | CWE-120 | high |
| R005 | `printf(non_literal)` | CWE-134 (format string) | medium |
| R006 | `system()` | CWE-78 (command injection) | high |
| R007 | `scanf("%s")` without width | CWE-120 | medium |
| R008 | `malloc(x * y)` | CWE-190 (integer overflow) | medium |
| R009 | `free()` | CWE-416 (use-after-free) | low |
| R010 | `memcpy` / `memmove` | CWE-120 | medium |

Fast (~10 ms) but text-only — it can't tell a real call from one inside a
comment or string.

### Layer 2 — Tree-sitter AST (`src/pipeline/ast_layer.py`)

Parses the code into a syntax tree and **validates** each Layer-1 hit:

- if the rule targets a function (e.g. `strcpy`), it is confirmed only when an
  actual `call_expression` for that function exists on the same line;
- otherwise it is confirmed only when the match is *not* inside a comment,
  string, or character literal.

This removes the most common false positives (comments, log messages, docs).

### Layer 3 — GraphCodeBERT (`src/pipeline/model_layer.py`)

A `microsoft/graphcodebert-base` encoder with a 2-class head, fine-tuned on
~132K C/C++ functions. Two interchangeable backends share one interface:

- **`VulnClassifierONNX`** — loads `models/model.onnx` (INT8-quantized,
  ~125 MB) via ONNX Runtime. No PyTorch needed. ← *used by CI*
- **`VulnClassifier`** — loads the PyTorch checkpoint directly (dev /
  fallback).

`get_classifier()` picks ONNX when available, else PyTorch.

---

## Project structure

```text
vulndetect-cpp/
├── .github/
│   └── workflows/
│       └── vuln_check.yml        # CI gate (Phase 7)
├── data/
│   ├── raw/                      # downloaded Big-Vul data (git-ignored)
│   └── processed/                # train/val/test.parquet + class_weights.json
├── models/
│   ├── model.onnx                # exported INT8 model (Phase 6, git-ignored)
│   └── graphcodebert_finetuned/  # PyTorch checkpoint (Phase 4, git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb              # Phase 2 — explore the dataset
│   ├── 02_preprocessing.ipynb    # Phase 3 — clean / dedupe / split
│   └── 03_finetune.ipynb         # Phase 4 — train GraphCodeBERT (Colab)
├── samples/
│   ├── sample_vulnerable.c       # CI fixture — must be flagged
│   └── sample_safe.c             # CI fixture — must pass
├── scripts/
│   ├── export_onnx.py            # Phase 6 — export + quantize + verify
│   ├── fetch_model.py            # Phase 7 — fetch model on CI runner
│   └── scan_samples.py           # Phase 7 — scan a folder, gate on result
├── src/
│   ├── pipeline/
│   │   ├── __init__.py           # analyze() — the public entry point
│   │   ├── regex_layer.py        # Layer 1
│   │   ├── ast_layer.py          # Layer 2
│   │   └── model_layer.py        # Layer 3
│   └── utils.py
├── tests/
│   └── test_scan_gate.py         # unit tests for the CI gate logic
├── requirements.txt              # full local/development deps
├── requirements-ci.txt           # light deps — ONNX inference + tree-sitter
├── requirements-fetch.txt        # heavy deps — only for HF model rebuild on CI
└── README.md
```

---

## Quick start

```powershell
# 1. Create & activate a virtual environment (Python 3.10+)
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Run the pipeline on the bundled samples
python scripts\scan_samples.py scan samples
```

> **Model files.** `models/model.onnx` and `models/graphcodebert_finetuned/`
> are git-ignored because they are large binary artifacts. To run Layer 3 you
> need one of:
> - a `models/model.onnx` produced by [Phase 6](#onnx-export), **or**
> - a `models/graphcodebert_finetuned/` checkpoint produced by [Phase 4](#dataset).
>
> Without a model, the pipeline still runs Layers 1–2 and falls back to the
> untrained base model for Layer 3 (with a loud warning) — handy for early
> development.

---

## Using the pipeline

### As a Python library

```python
from src.pipeline import analyze

code = """
#include <string.h>
void copy(const char *input) {
    char buf[16];
    strcpy(buf, input);   // BAD: no bounds check
}
"""

result = analyze(code)
print(result.verdict)                 # 'vulnerable' | 'needs_review' | 'likely_safe'
print(result.confirmed_findings)      # regex findings that survived AST validation
print(result.model_prediction)        # ModelPrediction(label=1, probability=0.97, ...)
```

### From the command line

```powershell
# scan every C/C++ file in a folder (exit code 1 if any is 'vulnerable')
python scripts\scan_samples.py scan path\to\code

# machine-readable summary
python scripts\scan_samples.py export path\to\code
```

---

## Dataset

**Source:** [`benjis/bigvul`](https://huggingface.co/datasets/benjis/bigvul)
(Big-Vul) on Hugging Face — 217K real-world C/C++ functions mined from the
National Vulnerability Database, each labeled `vul ∈ {0, 1}`.

**Preprocessing (Phase 3, `notebooks/02_preprocessing.ipynb`):**

1. **Language normalization** — raw labels `C`, `CPP`, `C++` are normalized and
   filtered to C / C++ only.
2. **Cleaning** — strip whitespace, drop empty `func_before`.
3. **Deduplication** — exact-duplicate `func_before` rows removed globally
   (prevents the same function leaking from train into test).
4. **Stratified split** — 80/10/10 train/val/test, preserving the class ratio.
5. **Class weights** — saved to `data/processed/class_weights.json` for
   weighted sampling (the `vul=1` class is ~17× rarer than `vul=0`).

### Fine-tuning (Phase 4, `notebooks/03_finetune.ipynb`)

| Hyperparameter | Value |
|----------------|-------|
| Base model | `microsoft/graphcodebert-base` |
| Max length | 512 (with truncation) |
| Batch size | 16 (T4 GPU) |
| Epochs | 3 |
| Learning rate | 2e-5 (AdamW) |
| Scheduler | linear warmup (10% of steps) |
| Sampling | `WeightedRandomSampler` (balanced batches) |
| Checkpoint | best **validation F1** on the vulnerable class |

> The notebook auto-detects Colab vs. local and is designed to run on a free
> Google Colab T4 GPU (~1–2 h for 3 epochs).

---

## Results

*Fill in after the Phase 4 training run completes.*

| Metric | Value |
|--------|-------|
| Accuracy | — |
| Precision (vul=1) | — |
| Recall (vul=1) | — |
| F1 (vul=1) | — |
| ROC-AUC | — |

> **Why not accuracy?** With ~17:1 class imbalance a model that always predicts
> "not vulnerable" is ~94% accurate yet useless. We optimise and report
> precision / recall / F1 for the *vulnerable* class instead.

---

## ONNX export

`scripts/export_onnx.py` (Phase 6) converts the fine-tuned checkpoint into an
INT8-quantized ONNX model for fast, dependency-light inference:

```powershell
python scripts\export_onnx.py
```

The script:

1. exports the PyTorch model to fp32 ONNX (opset 14, dynamic batch/seq);
2. applies **dynamic INT8 quantization** (`model.onnx`, ~475 MB → ~125 MB);
3. **verifies** the quantized model's *predicted labels* agree with the
   original PyTorch model on real test samples (≥ 90% required);
4. **benchmarks** both backends and prints the speedup.

---

## CI/CD

[`.github/workflows/vuln_check.yml`](./.github/workflows/vuln_check.yml) runs on
every push and pull request:

```text
push / PR ──► compile-check ──► fetch-model ──► scan (the gate)
```

- **compile-check** — byte-compiles every Python file.
- **fetch-model** — makes the ONNX model + tokenizer available on the runner
  (cache → `CI_MODEL_URL` → rebuild from HuggingFace).
- **scan** — runs the full 3-layer pipeline over `samples/` and **fails the
  build if any file is flagged `vulnerable`**, uploading the report as an
  artifact on failure.

### Making the model available in CI

Because `models/*` is git-ignored, the runner has three ways to get the model,
tried in order:

| Mechanism | Setup | Notes |
|-----------|-------|-------|
| **Actions cache** | none (automatic) | built once, then restored in seconds |
| **`CI_MODEL_URL`** | set a repo *Variable* | fastest & most deterministic |
| **HuggingFace rebuild** | none | heaviest fallback path |

For a portfolio repo the recommended approach is to upload `model.onnx` to a
GitHub Release and set the repository variable `CI_MODEL_URL` to its raw URL
(*Settings → Secrets and variables → Actions → Variables*).

---

## Tests

```powershell
python -m pytest tests -q
```

`tests/test_scan_gate.py` unit-tests the CI gate logic (with a fake `analyze`
so no model/network is required): a vulnerable file must fail the scan, a safe
file must pass, and the JSON export must be serializable.

---

## Limitations

- **Truncation at 512 tokens** — GraphCodeBERT (RoBERTa-based) caps at 512
  position embeddings; longer functions are truncated.
- **Function-level granularity** — the model labels whole functions, not
  specific lines or statements.
- **Regex rules are a fixed, curated set** — they cover common CWE patterns but
  not every possible vulnerability class.
- **Labels inherit Big-Vul's noise** — commit-based labels can be imperfect.
- The bundled ONNX model is **not** a production security product; it is an
  educational pipeline and a *signal*, not a proof of safety.

---

## License
The Big-Vul dataset and the GraphCodeBERT checkpoint are subject to their own
