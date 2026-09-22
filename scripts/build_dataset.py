"""Build the train/val/test splits from Big-Vul, without data leakage.

Why this script exists
----------------------
Big-Vul mines its rows **per fix-commit**: one CVE fix contributes every
function the commit touched, and those functions are near-identical to each
other. The dataset has 217K rows but only ~4,019 distinct commits — a median
of 24 functions per commit, up to 1,692.

Splitting those rows randomly therefore puts the same commit on both sides of
the split, and a model can score very well by recognising code it already saw.
The first version of this project did exactly that: 98.6% of test commits were
also in train, and the checkpoint scored F1 ~ 0.95 where published Big-Vul
results sit around 0.3-0.6.

**The official split shipped on the Hub does not fix this** — measured here,
3,208 of its 3,215 test commits (99.8%) also appear in its train split. So it
is rebuilt from scratch instead, grouped on `commit_id`.

What it does
------------
1. concatenate the Hub's three splits (they are rebuilt anyway)
2. normalize `lang` and keep C / C++ only
3. strip whitespace, drop empty functions
4. deduplicate on a **whitespace-normalized** hash, so functions that differ
   only in formatting collapse (exact-string dedup missed ~24% of these)
5. split 80/10/10 with `StratifiedGroupKFold` grouped on `commit_id`, which
   keeps every commit wholly inside one split while balancing the label rate
6. **assert** zero commit overlap and zero code overlap between splits, so a
   regression here fails loudly instead of silently inflating metrics
7. write the parquet files, `class_weights.json`, and `split_report.json`

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --out-dir data/processed --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import code_fingerprint

HF_DATASET = "benjis/bigvul"
GROUP_COL = "commit_id"
LABEL_COL = "vul"
CODE_COL = "func_before"
LANG_MAP = {"C": "C", "CPP": "C++", "C++": "C++"}
KEEP_COLS = ["project", "commit_id", "CWE ID", "lang", "func_before", "func_after", "vul"]

#: 10 folds -> one fold is test, one is val, the remaining eight are train.
N_FOLDS = 10


def load_raw() -> pd.DataFrame:
    """Concatenate the Hub's splits back into one frame.

    The official split is discarded deliberately: it is not commit-disjoint
    (see the module docstring), so keeping it would preserve the leakage this
    script exists to remove.
    """
    from datasets import concatenate_datasets, load_dataset

    print(f"Loading '{HF_DATASET}' ...")
    dsd = load_dataset(HF_DATASET)
    print("  Hub splits:", {k: len(v) for k, v in dsd.items()})
    df = concatenate_datasets([dsd["train"], dsd["validation"], dsd["test"]]).to_pandas()
    print(f"  Combined: {len(df):,} rows")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lang"] = df["lang"].map(LANG_MAP)
    before = len(df)
    df = df[df["lang"].isin(["C", "C++"])]
    print(f"Language filter: kept {len(df):,} / {before:,}")
    print("  " + ", ".join(f"{k}={v:,}" for k, v in df["lang"].value_counts().items()))

    df[CODE_COL] = df[CODE_COL].astype(str).str.strip()
    before = len(df)
    df = df[df[CODE_COL].str.len() > 0]
    if before != len(df):
        print(f"Dropped {before - len(df):,} rows with an empty {CODE_COL}")

    before = len(df)
    df = df[df[GROUP_COL].notna() & (df[GROUP_COL].astype(str).str.strip() != "")]
    if before != len(df):
        print(f"Dropped {before - len(df):,} rows with no {GROUP_COL}")

    return df.reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop functions that are identical once whitespace is normalized."""
    df = df.copy()
    df["_fingerprint"] = [code_fingerprint(c) for c in df[CODE_COL]]
    before = len(df)
    df = df.drop_duplicates(subset=["_fingerprint"], keep="first").reset_index(drop=True)
    dropped = before - len(df)
    print(f"Deduplication: dropped {dropped:,} near-duplicates ({dropped / before:.1%}), {len(df):,} remain")
    return df


def split(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """80/10/10, grouped on commit_id and stratified on the label."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in sgkf.split(df, df[LABEL_COL], groups=df[GROUP_COL])]

    test_idx, val_idx = folds[0], folds[1]
    train_idx = np.setdiff1d(np.arange(len(df)), np.concatenate([test_idx, val_idx]))

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def verify(splits: dict[str, pd.DataFrame]) -> dict:
    """Fail loudly on any leakage. This is the check the old pipeline lacked."""
    train, val, test = splits["train"], splits["val"], splits["test"]
    checks = {
        "commit_overlap_train_test": len(set(train[GROUP_COL]) & set(test[GROUP_COL])),
        "commit_overlap_train_val": len(set(train[GROUP_COL]) & set(val[GROUP_COL])),
        "commit_overlap_val_test": len(set(val[GROUP_COL]) & set(test[GROUP_COL])),
        "code_overlap_train_test": len(set(train["_fingerprint"]) & set(test["_fingerprint"])),
        "code_overlap_train_val": len(set(train["_fingerprint"]) & set(val["_fingerprint"])),
        "code_overlap_val_test": len(set(val["_fingerprint"]) & set(test["_fingerprint"])),
    }

    print("\nLeakage checks (all must be 0):")
    for name, value in checks.items():
        print(f"  {'OK ' if value == 0 else 'FAIL'} {name}: {value}")

    failed = {k: v for k, v in checks.items() if v != 0}
    if failed:
        raise AssertionError(f"Leakage detected: {failed}")
    return checks


def class_weights(train: pd.DataFrame) -> dict[str, float]:
    """Inverse-frequency weights, for WeightedRandomSampler during training."""
    counts = train[LABEL_COL].value_counts().to_dict()
    return {str(c): len(train) / (2 * n) for c, n in counts.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/processed", help="where to write the parquet files")
    parser.add_argument("--seed", type=int, default=42, help="split seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = deduplicate(clean(load_raw()))
    print(f"\nGrouping column '{GROUP_COL}': {df[GROUP_COL].nunique():,} distinct commits "
          f"for {len(df):,} functions (median {df.groupby(GROUP_COL).size().median():.0f} per commit)")

    splits = split(df, args.seed)

    print("\nSplit sizes:")
    summary = {}
    for name, part in splits.items():
        n_vul = int(part[LABEL_COL].sum())
        summary[name] = {
            "rows": len(part),
            "share": round(len(part) / len(df), 4),
            "vulnerable": n_vul,
            "vulnerable_rate": round(float(part[LABEL_COL].mean()), 4),
            "commits": int(part[GROUP_COL].nunique()),
        }
        print(f"  {name:5s}: {len(part):7,} rows ({len(part) / len(df):5.1%})  "
              f"vul=1: {n_vul:5,} ({part[LABEL_COL].mean():.2%})  commits: {part[GROUP_COL].nunique():,}")

    checks = verify(splits)

    weights = class_weights(splits["train"])
    (out_dir / "class_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(f"\nClass weights: {weights}")

    for name, part in splits.items():
        path = out_dir / f"{name}.parquet"
        part[KEEP_COLS].to_parquet(path, index=False)
        print(f"  Saved {path.name}: {part[KEEP_COLS].shape}")

    report = {
        "dataset": HF_DATASET,
        "grouped_on": GROUP_COL,
        "seed": args.seed,
        "n_folds": N_FOLDS,
        "total_rows_after_cleaning": len(df),
        "distinct_commits": int(df[GROUP_COL].nunique()),
        "splits": summary,
        "leakage_checks": checks,
        "class_weights": weights,
    }
    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Saved split_report.json")

    print("\nDone. Splits are commit-disjoint and duplicate-free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
