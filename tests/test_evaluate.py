import pytest

from text2sql.evaluate import evaluate_predictions

SCHEMA = "CREATE TABLE head (age INTEGER, name VARCHAR)"
GOLD = "SELECT COUNT(*) FROM head WHERE age > 56"


def ex(pred, gold=GOLD):
    return {"context": SCHEMA, "answer": gold, "prediction": pred}


def test_metrics():
    examples = [
        ex("select count(*) from head where age>56;"),  # valid + EM
        ex("SELECT COUNT(name) FROM head WHERE age > 56"),  # valid, not EM
        ex("SELECT COUNT(*) FROM heads"),  # invalid
        ex(None),  # missing
    ]
    res = evaluate_predictions(examples)
    assert res["n"] == 4
    assert res["valid_sql_pct"] == 50.0
    assert res["exact_match_pct"] == 25.0
    assert [r["valid_sql"] for r in res["per_example"]] == [True, True, False, False]
    assert res["per_example"][2]["error"]


def test_fenced_prediction_extracted():
    res = evaluate_predictions([ex(f"Here you go:\n```sql\n{GOLD}\n```")])
    assert res["valid_sql_pct"] == 100.0 and res["exact_match_pct"] == 100.0


def test_extract_disabled():
    res = evaluate_predictions([ex(f"```sql\n{GOLD}\n```")], extract=False)
    assert res["valid_sql_pct"] == 0.0


def test_custom_keys():
    res = evaluate_predictions([{"schema": SCHEMA, "gold": GOLD, "out": GOLD}],
                               pred_key="out", gold_key="gold", schema_key="schema")
    assert res["exact_match_pct"] == 100.0


def test_empty_raises():
    with pytest.raises(ValueError):
        evaluate_predictions([])
