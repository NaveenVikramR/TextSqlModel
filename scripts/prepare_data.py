"""Download, clean, split and (optionally) push b-mc2/sql-create-context.

Usage:
    python scripts/prepare_data.py --hub-repo <hf-user>/sql-create-context-clean
    python scripts/prepare_data.py --no-push --limit 2000      # quick local dry run

Auth for pushing: run `hf auth login` once, or set HF_TOKEN.
Outputs: data/processed/{train,validation,test}.jsonl and stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.env import load_env  # noqa: E402
from text2sql.data_prep import clean_records, split_records  # noqa: E402
from text2sql.jsonl import write_jsonl  # noqa: E402

log = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-id", default="b-mc2/sql-create-context")
    p.add_argument("--hub-repo", help="Target HF dataset repo, e.g. user/sql-create-context-clean")
    p.add_argument("--no-push", action="store_true", help="Skip pushing to the Hugging Face Hub")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--test-single", type=int, default=500, help="Single-table rows in test")
    p.add_argument("--test-multi", type=int, default=100, help="Multi-table rows in test")
    p.add_argument("--val-single", type=int, default=500, help="Single-table rows in validation")
    p.add_argument("--val-multi", type=int, default=100, help="Multi-table rows in validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="Only process the first N raw rows (debugging)")
    p.add_argument("--timeout", type=float, default=2.0, help="Per-query SQLite timeout (s)")
    return p.parse_args()


def main() -> None:
    load_env()  # .env secrets (HF_TOKEN, WANDB_API_KEY); real env vars win
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.no_push and not args.hub_repo:
        sys.exit("error: pass --hub-repo <user>/<name> or --no-push")

    try:
        from datasets import Dataset, DatasetDict, load_dataset
    except ImportError:
        sys.exit("error: `datasets` is not installed. Run: pip install -r requirements.txt")

    log.info("loading %s", args.dataset_id)
    raw = load_dataset(args.dataset_id, split="train")
    if args.limit:
        raw = raw.select(range(min(args.limit, len(raw))))

    records, stats = clean_records(raw, timeout_s=args.timeout)
    log.info("cleaning stats: %s", stats.to_dict())

    splits = split_records(
        records,
        test_sizes={"single_table": args.test_single, "multi_table": args.test_multi},
        val_sizes={"single_table": args.val_single, "multi_table": args.val_multi},
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        log.info("wrote %s: %d rows", name, len(rows))

    table_counts = {
        name: {
            "single_table": sum(r["num_tables"] <= 1 for r in rows),
            "multi_table": sum(r["num_tables"] > 1 for r in rows),
        }
        for name, rows in splits.items()
    }
    log.info("table counts per split: %s", table_counts)

    meta = {
        "source": args.dataset_id,
        "seed": args.seed,
        "limit": args.limit,
        "stratified_by": "num_tables (single_table: 1, multi_table: >1)",
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "table_counts": table_counts,
        "cleaning": stats.to_dict(),
    }
    (out_dir / "stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.no_push:
        log.info("--no-push set; skipping Hub upload")
        return

    dsd = DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()})
    dsd.push_to_hub(args.hub_repo, private=True)
    log.info("pushed private dataset to https://huggingface.co/datasets/%s", args.hub_repo)


if __name__ == "__main__":
    main()
