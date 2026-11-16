import pytest

from text2sql.data_prep import clean_records, split_records

SCHEMA = "CREATE TABLE head (age INTEGER, name VARCHAR)"


def row(q="How many heads are older than 56?", ctx=SCHEMA, ans="SELECT COUNT(*) FROM head WHERE age > 56"):
    return {"question": q, "context": ctx, "answer": ans}


class TestCleanRecords:
    def test_strips_fields(self):
        kept, _ = clean_records([row(q="  Q?  ", ctx=f"\n{SCHEMA} ", ans=" SELECT age FROM head \n")])
        assert kept == [{"question": "Q?", "context": SCHEMA, "answer": "SELECT age FROM head"}]

    def test_drops_empty_and_missing(self):
        kept, stats = clean_records([row(q="   "), {"question": "q", "context": SCHEMA}, row(ans=None)])
        assert kept == [] and stats.dropped_empty == 3

    def test_dedupes_on_question_and_schema_keeping_first(self):
        records = [
            row(ans="SELECT age FROM head"),
            row(ans="SELECT name FROM head"),  # same question+schema -> dropped
            row(q=" How many heads are older than 56? "),  # duplicate after strip
            row(ctx="CREATE TABLE other (age INTEGER)", ans="SELECT age FROM other"),  # new schema
        ]
        kept, stats = clean_records(records)
        assert [r["answer"] for r in kept] == ["SELECT age FROM head", "SELECT age FROM other"]
        assert stats.dropped_duplicate == 2

    def test_drops_bad_schema(self):
        kept, stats = clean_records([row(ctx="CREATE TABLE (broken")])
        assert kept == [] and stats.dropped_bad_schema == 1

    def test_drops_invalid_sql(self):
        kept, stats = clean_records([row(ans="SELECT salary FROM head")])
        assert kept == [] and stats.dropped_invalid_sql == 1

    def test_stats_add_up(self):
        records = [row(), row(), row(q=""), row(q="x", ctx="bad"), row(q="y", ans="SELEC"), row(q="z")]
        kept, stats = clean_records(records)
        dropped = (
            stats.dropped_empty + stats.dropped_duplicate + stats.dropped_bad_schema + stats.dropped_invalid_sql
        )
        assert stats.total == 6 and stats.kept == len(kept) == 2
        assert stats.total == stats.kept + dropped


class TestSplit:
    records = [{"question": f"q{i}", "context": SCHEMA, "answer": "SELECT age FROM head"} for i in range(50)]

    def test_sizes(self):
        s = split_records(self.records, val_size=5, test_size=7, seed=42)
        assert (len(s["train"]), len(s["validation"]), len(s["test"])) == (38, 5, 7)

    def test_disjoint_and_complete(self):
        s = split_records(self.records, val_size=5, test_size=7)
        qs = [r["question"] for part in s.values() for r in part]
        assert sorted(qs) == sorted(r["question"] for r in self.records)
        assert len(set(qs)) == len(qs)

    def test_deterministic(self):
        a = split_records(self.records, val_size=5, test_size=5, seed=42)
        b = split_records(self.records, val_size=5, test_size=5, seed=42)
        c = split_records(self.records, val_size=5, test_size=5, seed=7)
        assert a == b and a != c

    def test_does_not_mutate_input(self):
        before = list(self.records)
        split_records(self.records, val_size=5, test_size=5)
        assert self.records == before

    def test_too_few_records(self):
        with pytest.raises(ValueError):
            split_records(self.records[:10], val_size=5, test_size=5)
