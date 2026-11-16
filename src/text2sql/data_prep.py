"""Pure cleaning/splitting logic for b-mc2/sql-create-context.

Rows have fields: question (str), context (CREATE TABLE statements), answer (SQL).
No Hugging Face imports here; scripts/prepare_data.py handles loading and pushing.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from typing import Iterable

from .sql_utils import DEFAULT_TIMEOUT_S, SchemaError, build_sqlite_db, check_sql

log = logging.getLogger(__name__)

FIELDS = ("question", "context", "answer")


@dataclass
class CleanStats:
    total: int = 0
    dropped_empty: int = 0
    dropped_duplicate: int = 0
    dropped_bad_schema: int = 0
    dropped_invalid_sql: int = 0
    kept: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def strip_record(record: dict) -> dict:
    return {f: (record.get(f) or "").strip() for f in FIELDS}


def clean_records(
    records: Iterable[dict],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    log_every: int = 10_000,
) -> tuple[list[dict], CleanStats]:
    """Strip fields, drop empty rows, dedupe on (question, context) keeping the first
    occurrence, then drop rows whose schema or SQL fails in SQLite."""
    stats = CleanStats()
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []

    for i, raw in enumerate(records, 1):
        stats.total += 1
        rec = strip_record(raw)

        if not all(rec[f] for f in FIELDS):
            stats.dropped_empty += 1
            continue

        key = (rec["question"], rec["context"])
        if key in seen:
            stats.dropped_duplicate += 1
            continue
        seen.add(key)

        try:
            build_sqlite_db(rec["context"]).close()
        except SchemaError:
            stats.dropped_bad_schema += 1
            continue

        if not check_sql(rec["answer"], rec["context"], timeout_s=timeout_s).valid:
            stats.dropped_invalid_sql += 1
            continue

        kept.append(rec)
        if log_every and i % log_every == 0:
            log.info("processed %d rows, kept %d", i, len(kept))

    stats.kept = len(kept)
    return kept, stats


def split_records(
    records: list[dict],
    *,
    val_size: int = 500,
    test_size: int = 500,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Deterministic shuffle-then-slice split into train/validation/test."""
    if val_size < 0 or test_size < 0:
        raise ValueError("split sizes must be non-negative")
    if val_size + test_size >= len(records):
        raise ValueError(
            f"need more than {val_size + test_size} records for the split, got {len(records)}"
        )
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return {
        "test": shuffled[:test_size],
        "validation": shuffled[test_size : test_size + val_size],
        "train": shuffled[test_size + val_size :],
    }
