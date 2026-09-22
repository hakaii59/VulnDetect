# VulnDetect-CPP 🔍

**A 3-layer vulnerability detection pipeline for C/C++ source code** — regex
pattern matching → tree-sitter AST validation → a fine-tuned GraphCodeBERT
binary classifier, deployed as a CI gate that blocks vulnerable code before it
gets merged.

> **Portfolio project** — built end-to-end from raw dataset to ONNX inference
> and GitHub Actions, with every step documented and reproducible.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
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

Layers 1–2 are **deterministic**: a confirmed finding means Tree-sitter proved
there is a real call there, not a mention inside a comment or a string. Layer 3
is a *corroborating* signal and is **optional** — the pipeline runs without it.

| AST-confirmed finding | Model P(vulnerable) ≥ 0.5 | Verdict        |
|-----------------------|---------------------------|----------------|
| any `high` severity   | anything, or no model     | `vulnerable`   |
| `medium`/`low` only   | ✅                        | `vulnerable`   |
| `medium`/`low` only   | ❌, or no model           | `needs_review` |
| none                  | ✅                        | `needs_review` |
| none                  | ❌, or no model           | `likely_safe`  |

> **Why a high-severity finding is not gated behind the model.** The classifier
> is trained on Big-Vul CVE patches, so it under-fires badly on short,
> out-of-distribution code: a textbook `strcpy` overflow scores
> P(vulnerable) ≈ 0.001. Requiring model agreement therefore suppressed
> AST-confirmed findings and left the CI gate unable to fire at all.
> `python scripts/scan_samples.py selftest` guards against that regression on
> every CI run.

---

## The 3-layer pipeline

### Layer 1 — Regex (`src/pipeline/regex_layer.py`)

Eighteen regex rules map classic C/C++ bugs to their CWE IDs. **Lift** is
`P(vulnerable | rule fires)` divided by the 5.32% base rate, measured over all
163,636 functions — 1.00 would mean the rule carries no signal at all.

| Rule | Pattern | CWE | Severity | Hits | Lift |
|------|---------|-----|----------|------|------|
| R001 | `gets()` | CWE-242 | high | 0 | — |
| R002 | `strcpy()` | CWE-120 | high | 467 | 2.74 |
| R003 | `strcat()` | CWE-120 | high | 117 | 2.41 |
| R004 | `sprintf()` | CWE-120 | high | 767 | 2.16 |
| R005 | `printf(var)` | CWE-134 | high | 5 | 11.28 |
| R006 | `system/popen/exec*` | CWE-78 | high | 93 | 2.83 |
| R007 | `scanf("%s")` no width | CWE-120 | high | 34 | 6.08 |
| R008 | `[mcre]alloc(a * b)` | CWE-190 | medium | 465 | 3.44 |
| R009 | `free()` | CWE-416 | low | 3,109 | 2.07 |
| R010 | `memcpy` / `memmove` | CWE-119 | medium | 5,105 | 2.25 |
| R011 | `strncpy()` | CWE-170 | medium | 367 | 2.87 |
| R012 | `strncat()` | CWE-119 | medium | 17 | 3.32 |
| R013 | `alloca()` | CWE-789 | medium | 43 | 3.94 |
| R014 | `realloc()` | CWE-401 | medium | 264 | 3.06 |
| R015 | `[mre]alloc(a + b)` | CWE-190 | medium | 374 | 2.77 |
| R016 | `(int) len/size` cast | CWE-197 | medium | 339 | 3.49 |
| R017 | `atoi` / `atol` / `atof` | CWE-190 | medium | 404 | 2.56 |
| R018 | `strtok()` | CWE-476 | low | 28 | 4.03 |

#### The rules were chosen by measurement, not from a textbook

The first version was a list of banned functions, and **3 of its 10 rules never
matched anything** across 163,636 real functions — not because the patterns are
rare, but because the regexes were written too narrowly. `R007` required no `;`
between `scanf(` and `%s`; `R008` matched only `malloc(a * b)` with bare
identifiers. They looked like coverage and delivered none.

Each candidate was scored by lift before being kept. Rules near 1.00 were
dropped: `sizeof(ptr)` fires on 7% of the corpus at only 2.12, and a permissive
`printf(ident, ...)` sits at 1.94 because passing a format variable is ordinary
code. Measured on the test split:

| | 10 rules (before) | 18 rules (after) |
|---|---|---|
| Rules that never fire | **3 / 10** | 3 / 18 |
| Functions flagged | 763 (4.7%) | 852 (5.2%) |
| Precision | 10.88% | **11.97%** |
| Recall | 9.54% | **11.72%** |
| `high`-only precision (what gates CI) | 11.93% | **13.45%** |
| `high`-only recall | 1.49% | **1.84%** |

Precision and recall both improved and no function stopped being flagged. The
89 newly flagged functions are **21.3% vulnerable against a 5.32% base rate**,
so the added rules carry signal rather than noise.

The three rules still at zero on the test split are rare, not broken:
`R005` fires 5 times corpus-wide at the highest lift in the set (11.28), `R012`
17 times, and `R001` never — `gets()` was removed from C11 and does not appear
in Big-Vul at all. It is kept because a scanner that misses it is broken.

**Lift is not precision.** At a 5.32% base rate even lift 3.4 means roughly five
in six hits are not on a function Big-Vul labels vulnerable. That is what Layers
2 and 3 are for.

Fast (~10 ms) but text-only — it can't tell a real call from one inside a
comment or string.

### Layer 2 — Tree-sitter AST (`src/pipeline/ast_layer.py`)

Parses the code into a syntax tree and **validates** each Layer-1 hit:

- if the rule targets a function (e.g. `strcpy`), it is confirmed only when an
  actual `call_expression` for that function exists on the same line;
- otherwise it is confirmed only when the match is *not* inside a comment,
  string, or character literal.

This removes the most common false positives (comments, log messages, docs).

Severity decides the gate: a `high` rule confirmed here fails CI on its own, so
`high` is reserved for calls that are unsafe by construction (`gets`, `strcpy`,
`sprintf`, a variable format string, `scanf("%s")`, spawning a shell). `medium`
covers calls that are risky but often correct, and `low` needs context — both
produce `needs_review` rather than blocking a merge.

> **Caveat.** Big-Vul functions are extracted without their headers and type
> definitions, so ~14% of them do not parse cleanly (measured: 68/500 sampled
> test functions). When the tree has errors, this layer keeps every Layer-1
> finding rather than silently dropping hits it cannot verify — on those
> snippets the pipeline is effectively regex-only.

### Layer 3 — GraphCodeBERT (`src/pipeline/model_layer.py`)

A `microsoft/graphcodebert-base` encoder with a 2-class head, fine-tuned on
~132K C/C++ functions. Two interchangeable backends share one interface:

- **`VulnClassifierONNX`** — loads `models/model.onnx` (INT8-quantized,
  ~125 MB) via ONNX Runtime. No PyTorch needed. ← *used by CI*
- **`VulnClassifier`** — loads the PyTorch checkpoint directly (dev /
  fallback).

`get_classifier()` picks ONNX when available, else the PyTorch checkpoint, and
returns `None` when neither is present — Layer 3 is optional, so a fresh clone
and a CI run with no model configured both still work in 2-layer mode. (Loading
the *untrained* base model is now opt-in via `allow_untrained=True`: it produced
meaningless probabilities that looked real.)

---

## Project structure

```text
vulndetect-cpp/
├── .github/
│   └── workflows/
│       └── vuln_check.yml        # CI gate (Phase 7)
├── data/
│   ├── raw/                      # downloaded Big-Vul data (git-ignored)
│   └── processed/                # train/val/test.parquet, class_weights.json,
│                                 #   split_report.json (leakage checks)
├── models/
│   ├── model.onnx                # exported INT8 model (Phase 6, git-ignored)
│   └── graphcodebert_finetuned/  # PyTorch checkpoint (Phase 4, git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb              # Phase 2 — explore the dataset
│   ├── 02_preprocessing.ipynb    # Phase 3 — clean / dedupe / split
│   └── 03_finetune.ipynb         # Phase 4 — train GraphCodeBERT (Colab)
├── samples/
│   ├── sample_vulnerable.c       # CI fixture — must be flagged
│   ├── sample_safe.c             # CI fixture — must pass
│   └── expected.json             # expected verdict per fixture (self-test)
├── scripts/
│   ├── build_dataset.py          # Phase 3 — leakage-free splits (asserts it)
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
│   ├── conftest.py               # fixtures: no-model / stub-model pipelines
│   ├── test_build_dataset.py     # split integrity — the anti-leakage tests
│   ├── test_pipeline.py          # regex rules, AST validation, verdict logic
│   └── test_scan_gate.py         # unit tests for the CI gate logic
├── requirements.txt              # full local/development deps
├── requirements-ci.txt           # light deps — ONNX inference + tree-sitter
├── requirements-dev.txt          # requirements-ci.txt + pytest
├── requirements-fetch.txt        # heavy deps — only for HF model rebuild on CI
└── README.md
```

---

## Quick start

```powershell
# 1. Create & activate a virtual environment (Python 3.12+)
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
> Without a model the pipeline runs Layers 1–2 only and reports
> `model_prediction = None`. That is a supported mode, not a degraded one: the
> deterministic layers carry the gate on their own.

> **Python 3.12+ is required.** The pinned `numpy==2.5.2` declares
> `requires-python >= 3.12`, which sets the floor for the whole project. CI
> runs 3.12; development was done on 3.14.

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
print(result.reason)                  # why that verdict was reached
print(result.layers_run)              # ['regex', 'ast'] or ['regex', 'ast', 'model']
print(result.confirmed_findings)      # regex findings that survived AST validation
print(result.validated)               # every finding + its AST confirm/reject reason
print(result.model_prediction)        # ModelPrediction(...), or None if no model
```

`model_prediction` is `None` whenever Layer 3 has no trained model available,
so check it before reading `.probability`. Use
`from src.pipeline import model_available` to test for it up front.

### From the command line

```powershell
# scan files and/or folders (exit code 1 if any file is 'vulnerable')
python scripts\scan_samples.py scan path\to\code
python scripts\scan_samples.py scan src\a.c src\b.cpp

# machine-readable summary
python scripts\scan_samples.py export path\to\code

# prove the gate still discriminates (samples/ vs samples/expected.json)
python scripts\scan_samples.py selftest
```

A target may be a file or a directory; directories are walked recursively.
CI passes the PR's changed files individually, and missing targets (files
deleted in the PR) are skipped rather than failing the gate.

---

## Dataset

**Source:** [`benjis/bigvul`](https://huggingface.co/datasets/benjis/bigvul)
(Big-Vul) on Hugging Face — 217K real-world C/C++ functions mined from the
National Vulnerability Database, each labeled `vul ∈ {0, 1}`.

### The split is the hard part

Big-Vul mines rows **per CVE fix commit**: one commit contributes every
function it touched, and those functions are near-identical to one another.
217K rows come from roughly **4,000 commits** — a median of 24 functions each,
up to 1,692.

Split those rows at random and the same commit lands on both sides. The model
then scores well by recognising code it already saw. The first version of this
project did exactly that — 98.6% of test commits were also in train.

**The official split shipped on the Hub does not fix this.** Measured in
`notebooks/02_preprocessing.ipynb`:

```
official test commits also in train: 3,208 / 3,215 (99.8%)
official val  commits also in train: 3,249 / 3,256 (99.8%)
```

So the split is rebuilt from scratch, grouped on `commit_id`. The trade-off is
deliberate: results are no longer directly comparable to papers using the
official split, but they are honest.

**Preprocessing ([`scripts/build_dataset.py`](./scripts/build_dataset.py), narrated in
`notebooks/02_preprocessing.ipynb`):**

1. **Language normalization** — `C`, `CPP`, `C++` normalized, filtered to C/C++
   (the result is ~98.6% C; C++ is only 3,088 functions).
2. **Cleaning** — strip whitespace, drop empty `func_before` and rows with no
   `commit_id`.
3. **Deduplication** — on a **whitespace-normalized** hash, so functions that
   differ only in formatting collapse. This removes 53,371 rows (24.6%) that
   exact-string dedup missed.
4. **Grouped, stratified split** — `StratifiedGroupKFold` on `commit_id`,
   80/10/10. Every commit stays wholly inside one split while the vulnerable
   rate stays constant across all three.
5. **Verification** — the builder **asserts** zero commit overlap and zero code
   overlap between splits, so a regression fails loudly instead of quietly
   inflating metrics. Results are written to `data/processed/split_report.json`.
6. **Class weights** — saved to `data/processed/class_weights.json` for
   `WeightedRandomSampler` (the `vul=1` class is ~18× rarer).

Rebuild it with:

```powershell
python scripts\build_dataset.py
```

| Split | Rows | Share | Vulnerable | Commits |
|-------|------|-------|-----------|---------|
| train | 130,910 | 80.0% | 6,962 (5.32%) | 3,208 |
| val | 16,363 | 10.0% | 871 (5.32%) | 389 |
| test | 16,363 | 10.0% | 870 (5.32%) | 394 |

All six leakage checks return 0.

#### What this split does and does not guarantee

It is **commit-disjoint**, not **project-disjoint**. 94.8% of test functions
come from projects that also appear in train — Chrome alone is 43% of the test
set, Linux another 22%. So the model may still exploit project-specific idioms,
APIs and house style.

That is the standard setting for Big-Vul work and it is what the numbers below
mean, but it is easier than the cross-project setting that the most pessimistic
published figures come from.

The stricter split is built by the same script:

```powershell
python scriptsuild_dataset.py --group-by project   # -> data/processed_project/
```

No project then appears in two splits, so the test set asks whether the model
generalises to a codebase it has never seen:

| | commit-disjoint | project-disjoint |
|---|---|---|
| Output | `data/processed/` | `data/processed_project/` |
| Groups | 3,991 commits | 309 projects |
| train | Chrome, Linux, Android, ... | same giants (105 projects) |
| test | commits from the same repos | php, FFmpeg, openssl, poppler, krb5 (103 projects) |
| Asks | "a fix you have not seen" | "a codebase you have not seen" |

`StratifiedGroupKFold` cannot produce the second one: Chrome is 40.5% of all
rows, so whichever fold holds it is 40% of the data rather than 10%. Project
splits use greedy largest-first assignment instead, which lands the giants in
train and builds val/test from the tail — reaching 80.0/10.0/10.0% with zero
project overlap.

Residual near-duplicate leakage was measured separately, by reducing each
function to a structural skeleton (comments and literals removed, non-keyword
identifiers collapsed) to catch clones that survive renaming:

| Test functions | Have a structural twin in train |
|----------------|--------------------------------|
| all 16,363 | 1,605 (9.8%) |
| skeleton ≥ 200 chars (6,079) | 122 (**2.0%**) |
| skeleton ≥ 500 chars (2,011) | 30 (**1.5%**) |

The 9.8% figure is dominated by trivial one-line functions, where structural
identity is meaningless. Among functions with real bodies the overlap is ~2%,
and it does **not** favour the positive class: 1.4% of vulnerable test
functions have a twin, against 2.1% of safe ones.

### Fine-tuning (Phase 4, `notebooks/03_finetune.ipynb`)

| Hyperparameter | Value |
|----------------|-------|
| Base model | `microsoft/graphcodebert-base` |
| Max length | 512 (with truncation — see [Limitations](#limitations)) |
| Effective batch size | 16 (`BATCH_SIZE` × `GRAD_ACCUM_STEPS`) |
| Epochs | 3 |
| Learning rate | 2e-5 (AdamW), gradients clipped at 1.0 |
| Scheduler | linear warmup (10% of steps) |
| Mixed precision | fp16 AMP when CUDA is available |
| Sampling | `WeightedRandomSampler` (balanced batches) |
| Checkpoint | best **validation F1** on the vulnerable class |
| Reported on | the **test** split, once, after training |

Gradient accumulation keeps the effective batch at 16 on GPUs that cannot hold
it in one step:

| GPU | `BATCH_SIZE` | `GRAD_ACCUM_STEPS` |
|-----|--------------|--------------------|
| Colab T4 (16 GB) | 16 | 1 |
| 4 GB laptop GPU (e.g. RTX 3050 Ti) | 4 | 4 |

> The notebook auto-detects Colab vs. local. On a free Colab T4 expect roughly
> **3 hours** for 3 epochs over ~131K functions.
>
> **CPU training is not viable** — measured at ~49 s per step (batch 16, seq
> 512) on an 8-thread laptop CPU, which is ~330 hours for 3 epochs.

#### Surviving a Colab disconnect

Colab's free tier drops sessions on idle (~90 min without interaction) and when
the shared GPU quota runs out. A 3-hour run will not reliably finish in one
sitting, so the training loop checkpoints for resume.

After every epoch it writes the **full** training state to Drive — weights,
AdamW moments, scheduler, AMP scaler, epoch counter, history — as
`models/training_state.pt` (~1.4 GB, overwritten in place). Reconnect, run the
notebook from the top, and the loop reports:

```
Resumed from .../training_state.pt -> starting at epoch 2, best val F1 so far 0.41
```

Worst case a disconnect costs one epoch (~1 hour) instead of the whole run.
Delete `training_state.pt` to force a fresh start.

Practical notes: keep the browser tab open and the machine awake; Kaggle
Notebooks are an alternative with a 12-hour session cap and ~30 GPU-hours per
week if Colab keeps dropping you.

---

## Results

> ### Status: split fixed, retrain pending
>
> The data pipeline now produces a commit-disjoint split (see
> [Dataset](#dataset)), and the leakage guarantee is covered by
> `tests/test_build_dataset.py`.
>
> The checkpoint currently in `models/` was trained on the **old, leaking**
> split and cannot be evaluated honestly against anything. Its old training set
> was 80% of rows sampled at random across ~4,000 commits, so with a median of
> 24 functions per commit it saw at least part of essentially every commit in
> the dataset — there is no held-out data left for it. (Scoring it on the new
> test set returns F1 ≈ 0.97, which measures memorisation, not skill.)
>
> So the table stays empty until a model is retrained on the clean split.
> `notebooks/03_finetune.ipynb` now evaluates the **test** set at the end and
> writes everything to `models/graphcodebert_finetuned/metrics.json`.

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

## Evaluating a checkpoint

Evaluation is pure inference — no gradients, no optimizer state — so it needs a
fraction of training's memory and does not need Colab:

```powershell
python scripts\evaluate.py                       # torch, GPU if one is available
python scripts\evaluate.py --backend onnx        # models/model.onnx, CPU only
python scripts\evaluate.py --limit 2000          # quick sanity check
python scripts\evaluate.py --data-dir data/processed_project
```

It sweeps the decision threshold on **validation only**, applies it unchanged
to test, reports both splits at 0.5 and at the tuned value, and writes
`models/graphcodebert_finetuned/metrics.json`.

Measured throughput on this machine (batch 16, seq 512): ONNX INT8 on CPU runs
at ~260 ms/function, so both splits (32,726 functions) take ~2.4 h; a small
CUDA GPU does it in minutes, since inference needs roughly 1.5 GB rather than
the ~4.7 GB training wants.

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
compile-check   byte-compiles every .py file
tests           pytest over tests/  (no model, no network)
fetch-model     OPTIONAL - only when a model source is configured
      |
      v
scan            self-test the gate, then scan the changed C/C++ files
```

- **compile-check** — byte-compiles every Python file.
- **tests** — runs the full `pytest` suite.
- **fetch-model** — *skipped unless configured.* Downloads or rebuilds
  `model.onnx` **and its tokenizer**, caches it, and publishes `models/` as a
  workflow artifact.
- **scan** — runs `selftest` first to prove the gate can still tell
  `sample_vulnerable.c` from `sample_safe.c`, then scans the C/C++ files this
  push/PR actually changed and **fails the build on a `vulnerable` verdict**.

Three deliberate choices here:

1. **The gate scans changed files, not `samples/`.** `samples/` contains a
   deliberately vulnerable fixture, so gating on it would paint the badge
   permanently red. The fixtures are covered by `selftest` instead, which
   asserts the *expected* verdict for each one.
2. **`selftest` runs before the gate.** A gate that can never fire is worse
   than no gate; this is the check that catches that failure mode.
3. **Layer 3 is optional.** With no model configured, `fetch-model` is skipped
   and `scan` runs in 2-layer mode, so the workflow is green on a fresh clone
   with zero setup.

### Enabling Layer 3 in CI

`models/*` is git-ignored (the ONNX model is ~125 MB), so set **one** repository
variable under *Settings → Secrets and variables → Actions → Variables*:

| Variable | Setup | Notes |
|----------|-------|-------|
| **`CI_MODEL_URL`** | direct URL to your INT8 `model.onnx` (e.g. a GitHub Release asset) | **Recommended.** Downloads ~125 MB, no torch install. |
| **`HF_MODEL_REPO`** | Hub repo holding your fine-tuned checkpoint | CI installs torch and rebuilds the INT8 ONNX on the runner. Slowest path. |

Either result is stored in the Actions cache, so later runs skip the work.
Set neither and CI simply runs the deterministic layers.

---

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

No test needs a model, a GPU, or the network — Layer 3 is stubbed via fixtures
in `tests/conftest.py`, so the suite runs anywhere `requirements-ci.txt` installs.

| File | Covers |
|------|--------|
| `tests/test_build_dataset.py` | that the split cannot leak: no commit spans two splits, no code is shared, the label rate is preserved, and `verify()` **raises** when leakage is injected |
| `tests/test_pipeline.py` | regex rule hygiene (unique IDs, well-formed CWE IDs), AST confirm/reject behaviour including qualified calls (`std::strcpy`), parse-error fallback, and the full verdict table in both 2-layer and 3-layer mode |
| `tests/test_scan_gate.py` | the CI gate: exit 1 on `vulnerable`, exit 0 on safe, explicit file targets, deleted targets, JSON export, and a live `selftest` against `samples/` |

The regression that made the gate unfireable (a high-severity AST-confirmed
finding suppressed by a low model score) has a dedicated test, as does the
`samples/expected.json` ↔ `samples/` sync.

---

## Limitations

- **Truncation at 512 tokens, unevenly** — GraphCodeBERT (RoBERTa-based) caps
  at 512 position embeddings. Measured on 2,000 test functions: median length
  is 170 tokens, but **18.6% exceed the cap — 34.3% of vulnerable functions
  against 17.8% of safe ones.** Vulnerable functions are longer, so the model
  often never sees the part where the bug lives, and the truncation rate itself
  correlates with the label. `notebooks/03_finetune.ipynb` measures this before
  training.
- **Function-level granularity** — the model labels whole functions, not
  specific lines or statements.
- **Regex rules are a fixed, curated set** — they cover common CWE patterns but
  not every possible vulnerability class.
- **Labels inherit Big-Vul's noise** — commit-based labels can be imperfect.
- **Layer 2 cannot validate every snippet** — ~14% of Big-Vul functions do not
  parse cleanly without their headers; on those the pipeline is regex-only.
- **Layer 3 does not generalize to short, synthetic code** — trained on
  real-world CVE patches, it scores a textbook `strcpy` overflow at
  P(vulnerable) ≈ 0.001. This is why high-severity findings are not gated
  behind it.
- **The current train/test split leaks** (see [Results](#results)); the reported
  metrics are intentionally blank until it is fixed and the model is retrained.
- The bundled ONNX model is **not** a production security product; it is an
  educational pipeline and a *signal*, not a proof of safety.

---

## License
The Big-Vul dataset and the GraphCodeBERT checkpoint are subject to their own
