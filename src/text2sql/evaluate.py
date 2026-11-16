"""Evaluation metrics: valid-SQL % and normalized exact-match %.

Input is an iterable of dicts containing the schema, gold SQL and the model's raw
prediction (e.g. a predictions .jsonl produced on a GPU box or via Ollama).
Metrics are reported overall and split into single-table vs multi-table schemas.
"""

from __future__ import annotations

from typing import Iterable

from .prompt import extract_sql
from .sql_utils import DEFAULT_TIMEOUT_S, check_sql, count_tables, exact_match

GROUPS = ("single_table", "multi_table")


def table_group(num_tables: int) -> str:
    return "multi_table" if num_tables > 1 else "single_table"


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "valid_sql_pct": None, "exact_match_pct": None}
    return {
        "n": n,
        "valid_sql_pct": round(100.0 * sum(r["valid_sql"] for r in rows) / n, 2),
        "exact_match_pct": round(100.0 * sum(r["exact_match"] for r in rows) / n, 2),
    }


def evaluate_predictions(
    examples: Iterable[dict],
    *,
    pred_key: str = "prediction",
    gold_key: str = "answer",
    schema_key: str = "context",
    extract: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Score predictions. Returns aggregate percentages (overall and per table group)
    plus per-example details.

    - valid_sql: the prediction is a read-only query that executes on the schema in SQLite
    - exact_match: normalize_sql(prediction) == normalize_sql(gold)
    A missing prediction counts as invalid and not matching. `num_tables` is taken from
    the example if present, otherwise counted from the schema.
    """
    per_example = []
    for i, ex in enumerate(examples):
        raw = ex.get(pred_key)
        pred = extract_sql(raw) if extract else (raw or "").strip()
        result = check_sql(pred, ex[schema_key], timeout_s=timeout_s)
        num_tables = ex.get("num_tables")
        if num_tables is None:
            num_tables = count_tables(ex[schema_key])
        per_example.append(
            {
                "index": i,
                "num_tables": num_tables,
                "group": table_group(num_tables),
                "prediction": pred,
                "gold": ex[gold_key],
                "valid_sql": result.valid,
                "error": result.error,
                "exact_match": bool(pred) and exact_match(pred, ex[gold_key]),
            }
        )

    if not per_example:
        raise ValueError("no examples to evaluate")

    return {
        **_aggregate(per_example),
        "by_table_count": {g: _aggregate([r for r in per_example if r["group"] == g]) for g in GROUPS},
        "per_example": per_example,
    }
