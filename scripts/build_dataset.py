#!/usr/bin/env python
"""Stage 1 CLI: merge MegaVul + Big-Vul + CVEfixes + PrimeVul into
train.jsonl / val.jsonl / test.jsonl under a shared schema.

Usage:
    python scripts/build_dataset.py                       # full pipeline
    python scripts/build_dataset.py --config configs/dataset.yaml
    python scripts/build_dataset.py --verify               # inspect raw
                                                             # files' columns
                                                             # without a full
                                                             # parse

See docs/DATASETS.md for where to download each dataset, and
configs/dataset.yaml for path/column_map overrides.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vulnerachek.data.balance import bucket_summary, compute_sample_weights  # noqa: E402
from vulnerachek.data.loaders import LOADERS  # noqa: E402
from vulnerachek.data.loaders.base import LoaderError  # noqa: E402
from vulnerachek.data.merge import merge_records  # noqa: E402
from vulnerachek.data.schema import VulnRecord  # noqa: E402
from vulnerachek.data.split import group_aware_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_dataset")


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_records(config: dict) -> list[VulnRecord]:
    all_records: list[VulnRecord] = []
    for name, loader_cls in LOADERS.items():
        ds_config = config["datasets"].get(name, {})
        if not ds_config.get("enabled", True):
            logger.info("[%s] disabled in config, skipping", name)
            continue

        loader = loader_cls(path=ds_config["path"], column_map=ds_config.get("column_map"))
        try:
            records = list(loader.load())
        except LoaderError as exc:
            logger.warning("%s -- skipping this dataset", exc)
            continue

        logger.info(loader.summary())
        all_records.extend(records)
    return all_records


def write_jsonl(path: Path, records: list[VulnRecord], weights: list[float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            row = record.to_row()
            if weights is not None:
                row["weight"] = weights[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote %d records -> %s", len(records), path)


def verify_columns(config: dict) -> None:
    """Print what a loader can actually see in the raw files, without a full parse."""
    import csv
    import sqlite3

    for name, ds_config in config["datasets"].items():
        path = Path(ds_config["path"])
        print(f"\n=== {name} ({path}) ===")
        if not path.exists():
            print("  NOT FOUND -- see docs/DATASETS.md")
            continue

        if name == "bigvul":
            with open(path, encoding="utf-8", errors="replace") as f:
                print("  columns:", csv.DictReader(f).fieldnames)
        elif name == "cvefixes":
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            print("  tables:", tables)
            for table in ("method_change", "file_change", "commits", "repository"):
                if table in tables:
                    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
                    print(f"  {table} columns:", cols)
            conn.close()
        elif name in ("megavul", "primevul"):
            glob_pattern = "*.json" if name == "megavul" else "*.jsonl"
            files = sorted(path.glob(glob_pattern)) if path.is_dir() else [path]
            if not files:
                print("  no matching files found")
                continue
            first_file = files[0]
            print(f"  sample file: {first_file}")
            with open(first_file, encoding="utf-8") as f:
                if name == "megavul":
                    data = json.load(f)
                    entries = data if isinstance(data, list) else list(data.values())
                    sample = entries[0] if entries else {}
                else:
                    sample = json.loads(f.readline() or "{}")
            print("  fields in first entry:", sorted(sample.keys()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument(
        "--verify", action="store_true", help="inspect raw files' detected columns/fields and exit"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.verify:
        verify_columns(config)
        return

    records_by_source = load_all_records(config)
    if not records_by_source:
        logger.error("no records loaded from any dataset -- nothing to write. Check configs/dataset.yaml.")
        sys.exit(1)

    merged = list(merge_records(records_by_source))

    split_cfg = config.get("split", {})
    splits = group_aware_split(
        merged,
        ratios=split_cfg.get("ratios", (0.8, 0.1, 0.1)),
        split_names=split_cfg.get("names", ("train", "val", "test")),
        seed=split_cfg.get("seed", 42),
    )

    output_dir = Path(config.get("output_dir", "data/processed"))
    for split_name, split_records in splits.items():
        weights = compute_sample_weights(split_records) if split_name == "train" else None
        write_jsonl(output_dir / f"{split_name}.jsonl", split_records, weights)
        if split_name == "train":
            logger.info("train (label, language) buckets before weighting: %s", bucket_summary(split_records))

    logger.info("done.")


if __name__ == "__main__":
    main()
