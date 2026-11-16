"""Evaluation metrics: valid-SQL % and normalized exact-match %.

Input is an iterable of dicts containing the schema, gold SQL and the model's raw
prediction (e.g. a predictions .jsonl produced on a GPU box or via Ollama).
"""

from __future__ import annotations

from typing import Iterable

from .prompt import extract_sql
from .sql_utils import DEFAULT_TIMEOUT_S, check_sql, exact_match


def evaluate_predictions(
    examples: Iterable[dict],
    *,
    pred_key: str = "prediction",
    gold_key: str = "answer",
    schema_key: str = "context",
    extract: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Score predictions. Returns aggregate percentages plus per-example details.

    - valid_sql: the prediction executes as a single statement on the schema in SQLite
    - exact_match: normalize_sql(prediction) == normalize_sql(gold)
    A missing prediction counts as invalid and not matching.
    """
    per_example = []
    for i, ex in enumerate(examples):
        raw = ex.get(pred_key)
        pred = extract_sql(raw) if extract else (raw or "").strip()
        result = check_sql(pred, ex[schema_key], timeout_s=timeout_s)
        per_example.append(
            {
                "index": i,
                "prediction": pred,
                "gold": ex[gold_key],
                "valid_sql": result.valid,
                "error": result.error,
                "exact_match": bool(pred) and exact_match(pred, ex[gold_key]),
            }
        )

    n = len(per_example)
    if n == 0:
        raise ValueError("no examples to evaluate")

    n_valid = sum(r["valid_sql"] for r in per_example)
    n_em = sum(r["exact_match"] for r in per_example)
    return {
        "n": n,
        "valid_sql_pct": round(100.0 * n_valid / n, 2),
        "exact_match_pct": round(100.0 * n_em / n, 2),
        "per_example": per_example,
    }
