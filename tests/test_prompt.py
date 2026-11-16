from text2sql.prompt import SYSTEM_PROMPT, build_messages, build_prompt_completion, extract_sql


def test_build_messages_inference():
    msgs = build_messages(" How many? ", "CREATE TABLE t (a INT)\n")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["content"] == "### Schema\nCREATE TABLE t (a INT)\n\n### Question\nHow many?"


def test_build_messages_training():
    msgs = build_messages("q", "s", answer=" SELECT 1 ")
    assert msgs[-1] == {"role": "assistant", "content": "SELECT 1"}


def test_build_prompt_completion_matches_inference_prompt():
    ex = build_prompt_completion("q", "s", " SELECT 1 ")
    assert ex["prompt"] == build_messages("q", "s")
    assert ex["completion"] == [{"role": "assistant", "content": "SELECT 1"}]


def test_extract_sql():
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("Sure!\n```SQL\nSELECT a\nFROM t\n```\nDone") == "SELECT a\nFROM t"
    assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("  SELECT 1  ") == "SELECT 1"
    assert extract_sql(None) == ""
