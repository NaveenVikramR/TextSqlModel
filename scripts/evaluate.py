"""Score a predictions file.

Input: .jsonl where each line has `context` (schema), `answer` (gold SQL) and
`prediction` (raw model output; ```sql fences are stripped automatically).

Usage:
    python scripts/evaluate.py --predictions outputs/preds_llama32_3b.jsonl \
        --report outputs/report_llama32_3b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.evaluate import evaluate_predictions  # noqa: E402
from text2sql.jsonl import read_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True)
    p.add_argument("--report", help="Optional path to write the full JSON report")
    p.add_argument("--pred-key", default="prediction")
    p.add_argument("--timeout", type=float, default=2.0)
    args = p.parse_args()

    result = evaluate_predictions(read_jsonl(args.predictions), pred_key=args.pred_key, timeout_s=args.timeout)
    print(f"n={result['n']}  valid_sql={result['valid_sql_pct']}%  exact_match={result['exact_match_pct']}%")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
