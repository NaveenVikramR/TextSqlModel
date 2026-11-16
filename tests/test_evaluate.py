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


def test_write_query_counts_as_invalid():
    res = evaluate_predictions([ex("DELETE FROM head"), ex("WITH x AS (SELECT 1) DELETE FROM head")])
    assert res["valid_sql_pct"] == 0.0
    assert all("read-only" in r["error"] or "not authorized" in r["error"] for r in res["per_example"])


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


MULTI = "CREATE TABLE a (id INT, name VARCHAR); CREATE TABLE b (id INT)"
MULTI_GOLD = "SELECT a.name FROM a JOIN b ON a.id = b.id"


def test_split_by_table_count():
    examples = [
        ex(GOLD),                                                          # single: valid + EM
        ex("SELECT age FROM head"),                                        # single: valid only
        {"context": MULTI, "answer": MULTI_GOLD, "prediction": MULTI_GOLD},  # multi: valid + EM
        {"context": MULTI, "answer": MULTI_GOLD, "prediction": "SELECT x FROM a"},  # multi: invalid
        {"context": MULTI, "answer": MULTI_GOLD, "prediction": "SELECT name FROM a"},  # multi: valid only
    ]
    res = evaluate_predictions(examples)
    assert (res["n"], res["valid_sql_pct"], res["exact_match_pct"]) == (5, 80.0, 40.0)
    single, multi = res["by_table_count"]["single_table"], res["by_table_count"]["multi_table"]
    assert single == {"n": 2, "valid_sql_pct": 100.0, "exact_match_pct": 50.0}
    assert multi == {"n": 3, "valid_sql_pct": 66.67, "exact_match_pct": 33.33}
    assert [r["group"] for r in res["per_example"]] == ["single_table"] * 2 + ["multi_table"] * 3


def test_num_tables_field_preferred_over_schema():
    res = evaluate_predictions([{**ex(GOLD), "num_tables": 2}])
    assert res["by_table_count"]["multi_table"]["n"] == 1


def test_empty_group_reported_as_none():
    res = evaluate_predictions([ex(GOLD)])
    assert res["by_table_count"]["multi_table"] == {"n": 0, "valid_sql_pct": None, "exact_match_pct": None}


def test_empty_raises():
    with pytest.raises(ValueError):
        evaluate_predictions([])
