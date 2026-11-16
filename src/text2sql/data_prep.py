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

from .sql_utils import (
    DEFAULT_TIMEOUT_S,
    TABLE_GROUPS,
    SchemaError,
    build_sqlite_db,
    check_sql,
    count_tables,
    error_kind,
    table_group,
)

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


DEFAULT_TEST_SIZES = {"single_table": 500, "multi_table": 100}
DEFAULT_VAL_SIZES = {"single_table": 500, "multi_table": 100}


def split_records(
    records: list[dict],
    *,
    test_sizes: dict[str, int] = DEFAULT_TEST_SIZES,
    val_sizes: dict[str, int] = DEFAULT_VAL_SIZES,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Deterministic split stratified by table count.

    Records are grouped into single_table / multi_table (via `num_tables`, counted from
    the schema if absent). Within each group, rows are shuffled and the first
    `test_sizes[group]` go to test, the next `val_sizes[group]` to validation, and the
    rest to train. Each split is then shuffled so groups are interleaved.
    Multi-table rows are rare (~2%), so reserving a fixed count gives the eval splits
    enough of them to report a meaningful multi-table metric.
    """
    unknown = (set(test_sizes) | set(val_sizes)) - set(TABLE_GROUPS)
    if unknown:
        raise ValueError(f"unknown groups {sorted(unknown)}; expected {TABLE_GROUPS}")
    if any(v < 0 for v in [*test_sizes.values(), *val_sizes.values()]):
        raise ValueError("split sizes must be non-negative")

    groups: dict[str, list[dict]] = {g: [] for g in TABLE_GROUPS}
    for rec in records:
        n = rec.get("num_tables")
        groups[table_group(count_tables(rec["context"]) if n is None else n)].append(rec)

    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"test": [], "validation": [], "train": []}
    for group in TABLE_GROUPS:  # fixed order -> deterministic
        rows = list(groups[group])
        n_test, n_val = test_sizes.get(group, 0), val_sizes.get(group, 0)
        if n_test + n_val > len(rows):
            raise ValueError(
                f"group '{group}' has {len(rows)} records, need {n_test + n_val} for test + validation"
            )
        rng.shuffle(rows)
        splits["test"] += rows[:n_test]
        splits["validation"] += rows[n_test : n_test + n_val]
        splits["train"] += rows[n_test + n_val :]

    for rows in splits.values():
        rng.shuffle(rows)
    return splits
