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

`--group-by project` builds the stricter, cross-codebase variant into
`data/processed_project/`: no project appears in two splits, so the test set
asks whether the model generalises to a repo it has never seen. Big-Vul is
dominated by a few repos (Chrome 40.5% of rows, Linux 25.3%), so the split
proportions there are approximate by necessity — see `_split_greedy`.

What it does
------------
1. concatenate the Hub's three splits (they are rebuilt anyway)
2. normalize `lang` and keep C / C++ only
3. strip whitespace, drop empty functions
4. deduplicate on a **whitespace-normalized** hash, so functions that differ
   only in formatting collapse. Removes 53,371 rows (24.6%); exact-string
   dedup would catch 51,572 of them, normalizing catches 1,799 more. Those
   1,799 are the ones that matter: each pair spans two different commits, so
   the commit-grouped split would otherwise put them on opposite sides.
5. split 80/10/10 with `StratifiedGroupKFold` grouped on `commit_id`, which
   keeps every commit wholly inside one split while balancing the label rate
6. **assert** zero commit overlap and zero code overlap between splits, so a
   regression here fails loudly instead of silently inflating metrics
7. write the parquet files, `class_weights.json`, and `split_report.json`

Usage:
    python scripts/build_dataset.py                      # -> data/processed/
    python scripts/build_dataset.py --group-by project   # -> data/processed_project/
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


def split(df: pd.DataFrame, seed: int, group_col: str = GROUP_COL) -> dict[str, pd.DataFrame]:
    """80/10/10, grouped so no group spans two splits, stratified on the label.

    `StratifiedGroupKFold` balances both size and label rate across folds, which
    works well for commit_id (~4,000 similar-sized groups). It cannot work for
    `project`: Big-Vul is dominated by a handful of repos (Chrome 40.5% of rows,
    Linux 25.3%), so whichever fold holds Chrome is 40% of the data, not 10%.
    Project splits therefore use greedy assignment instead.
    """
    if group_col == "project":
        return _split_greedy(df, seed, group_col)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in sgkf.split(df, df[LABEL_COL], groups=df[group_col])]

    test_idx, val_idx = folds[0], folds[1]
    train_idx = np.setdiff1d(np.arange(len(df)), np.concatenate([test_idx, val_idx]))

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def _split_greedy(df: pd.DataFrame, seed: int, group_col: str) -> dict[str, pd.DataFrame]:
    """Assign whole groups largest-first to whichever split is furthest short.

    Classic bin-packing heuristic. With a few huge groups, exact 80/10/10 is
    impossible -- the giants land in train and val/test are built from the tail.
    That is the point of a project-held-out evaluation: it asks whether the
    model generalises to a codebase it has never seen.
    """
    targets = {"train": 0.8, "val": 0.1, "test": 0.1}
    sizes = df.groupby(group_col).size().sort_values(ascending=False)

    # Deterministic tie-breaking, but seed-dependent so the split can be varied.
    rng = np.random.default_rng(seed)
    order = list(sizes.index)

    assigned: dict[str, list] = {k: [] for k in targets}
    counts = {k: 0 for k in targets}
    total = len(df)

    for group in order:
        n = int(sizes[group])
        deficits = {k: targets[k] * total - counts[k] for k in targets}
        best = max(deficits.values())
        # Break ties randomly so no split is systematically favoured.
        candidates = [k for k, v in deficits.items() if v >= best - 1e-9]
        choice = candidates[0] if len(candidates) == 1 else str(rng.choice(candidates))
        assigned[choice].append(group)
        counts[choice] += n

    return {
        name: df[df[group_col].isin(groups)].reset_index(drop=True)
        for name, groups in assigned.items()
    }


def verify(splits: dict[str, pd.DataFrame], group_col: str = GROUP_COL) -> dict:
    """Fail loudly on any leakage. This is the check the old pipeline lacked."""
    train, val, test = splits["train"], splits["val"], splits["test"]
    checks = {
        f"{group_col}_overlap_train_test": len(set(train[group_col]) & set(test[group_col])),
        f"{group_col}_overlap_train_val": len(set(train[group_col]) & set(val[group_col])),
        f"{group_col}_overlap_val_test": len(set(val[group_col]) & set(test[group_col])),
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
    parser.add_argument("--out-dir", default=None, help="where to write the parquet files")
    parser.add_argument("--seed", type=int, default=42, help="split seed")
    parser.add_argument(
        "--group-by",
        choices=["commit_id", "project"],
        default=GROUP_COL,
        help="what must not span two splits. 'commit_id' is the default and "
             "removes the leakage that matters most; 'project' is the stricter, "
             "cross-codebase setting.",
    )
    args = parser.parse_args()

    group_col = args.group_by
    # Keep the two datasets side by side so both can be reported.
    default_out = "data/processed" if group_col == GROUP_COL else "data/processed_project"
    args.out_dir = args.out_dir or default_out

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = deduplicate(clean(load_raw()))
    print(f"\nGrouping on '{group_col}': {df[group_col].nunique():,} distinct groups "
          f"for {len(df):,} functions (median {df.groupby(group_col).size().median():.0f} per group)")

    splits = split(df, args.seed, group_col)

    print("\nSplit sizes:")
    summary = {}
    for name, part in splits.items():
        n_vul = int(part[LABEL_COL].sum())
        summary[name] = {
            "rows": len(part),
            "share": round(len(part) / len(df), 4),
            "vulnerable": n_vul,
            "vulnerable_rate": round(float(part[LABEL_COL].mean()), 4),
            "groups": int(part[group_col].nunique()),
        }
        print(f"  {name:5s}: {len(part):7,} rows ({len(part) / len(df):5.1%})  "
              f"vul=1: {n_vul:5,} ({part[LABEL_COL].mean():.2%})  "
              f"{group_col}s: {part[group_col].nunique():,}")

    checks = verify(splits, group_col)

    weights = class_weights(splits["train"])
    (out_dir / "class_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(f"\nClass weights: {weights}")

    for name, part in splits.items():
        path = out_dir / f"{name}.parquet"
        part[KEEP_COLS].to_parquet(path, index=False)
        print(f"  Saved {path.name}: {part[KEEP_COLS].shape}")

    report = {
        "dataset": HF_DATASET,
        "grouped_on": group_col,
        "seed": args.seed,
        "n_folds": N_FOLDS,
        "total_rows_after_cleaning": len(df),
        "distinct_groups": int(df[group_col].nunique()),
        "splits": summary,
        "leakage_checks": checks,
        "class_weights": weights,
    }
    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Saved split_report.json")

    print(f"\nDone. Splits are {group_col}-disjoint and duplicate-free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
