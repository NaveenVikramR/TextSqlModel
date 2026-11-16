"""Pure cleaning/splitting logic for b-mc2/sql-create-context.

Rows have fields: question (str), context (CREATE TABLE statements), answer (SQL).
No Hugging Face imports here; scripts/prepare_data.py handles loading and pushing.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable

from .sql_utils import DEFAULT_TIMEOUT_S, SchemaError, build_sqlite_db, check_sql, count_tables, error_kind

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
    # error category -> count, and one example row per category (for documentation)
    schema_error_kinds: Counter = field(default_factory=Counter)
    sql_error_kinds: Counter = field(default_factory=Counter)
    examples: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_error_kinds"] = dict(self.schema_error_kinds.most_common())
        d["sql_error_kinds"] = dict(self.sql_error_kinds.most_common())
        return d

    def _record(self, counter: Counter, prefix: str, message: str, rec: dict) -> None:
        kind = error_kind(message)
        counter[kind] += 1
        self.examples.setdefault(f"{prefix}: {kind}", {**rec, "error": message})


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
        except SchemaError as exc:
            stats.dropped_bad_schema += 1
            stats._record(stats.schema_error_kinds, "schema", str(exc), rec)
            continue

        result = check_sql(rec["answer"], rec["context"], timeout_s=timeout_s)
        if not result.valid:
            stats.dropped_invalid_sql += 1
            stats._record(stats.sql_error_kinds, "sql", result.error, rec)
            continue

        rec["num_tables"] = count_tables(rec["context"])
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
