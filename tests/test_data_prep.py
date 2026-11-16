import pytest

from text2sql.data_prep import clean_records, split_records

SCHEMA = "CREATE TABLE head (age INTEGER, name VARCHAR)"


def row(q="How many heads are older than 56?", ctx=SCHEMA, ans="SELECT COUNT(*) FROM head WHERE age > 56"):
    return {"question": q, "context": ctx, "answer": ans}


class TestCleanRecords:
    def test_strips_fields(self):
        kept, _ = clean_records([row(q="  Q?  ", ctx=f"\n{SCHEMA} ", ans=" SELECT age FROM head \n")])
        assert kept == [{"question": "Q?", "context": SCHEMA, "answer": "SELECT age FROM head", "num_tables": 1}]

    def test_num_tables_added(self):
        multi = "CREATE TABLE a (x INT); CREATE TABLE b (x INT, y INT); CREATE TABLE c (z INT)"
        kept, _ = clean_records([row(), row(q="j", ctx=multi, ans="SELECT a.x FROM a JOIN b ON a.x = b.x")])
        assert [r["num_tables"] for r in kept] == [1, 3]

    def test_error_kinds_and_examples_recorded(self):
        dup_table = "CREATE TABLE t (a INT); CREATE TABLE t (a INT, b INT)"
        records = [
            row(q="1", ctx=dup_table, ans="SELECT a FROM t"),
            row(q="2", ctx="CREATE TABLE t (id INT, ID INT)", ans="SELECT id FROM t"),
            row(q="3", ans="SELECT salary FROM head"),
            row(q="4", ans="SELECT bonus FROM head"),
        ]
        _, stats = clean_records(records)
        assert stats.schema_error_kinds == {"table already exists": 1, "duplicate column name": 1}
        assert stats.sql_error_kinds == {"no such column": 2}
        assert stats.examples["sql: no such column"]["question"] == "3"  # first one kept
        assert set(stats.examples) == {
            "schema: table already exists", "schema: duplicate column name", "sql: no such column"
        }
        d = stats.to_dict()
        assert d["sql_error_kinds"] == {"no such column": 2}

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


MULTI_SCHEMA = "CREATE TABLE a (x INT); CREATE TABLE b (x INT)"


def make_records(n_single, n_multi, with_num_tables=True):
    recs = [{"question": f"s{i}", "context": SCHEMA, "answer": "SELECT age FROM head", "num_tables": 1}
            for i in range(n_single)]
    recs += [{"question": f"m{i}", "context": MULTI_SCHEMA, "answer": "SELECT x FROM a", "num_tables": 2}
             for i in range(n_multi)]
    if not with_num_tables:
        for r in recs:
            del r["num_tables"]
    return recs


SIZES = dict(test_sizes={"single_table": 10, "multi_table": 3}, val_sizes={"single_table": 8, "multi_table": 2})


def groups(rows):
    return (sum(r["num_tables"] == 1 for r in rows), sum(r["num_tables"] > 1 for r in rows))


class TestSplit:
    records = make_records(60, 9)

    def test_stratified_sizes(self):
        s = split_records(self.records, **SIZES)
        assert groups(s["test"]) == (10, 3)
        assert groups(s["validation"]) == (8, 2)
        assert groups(s["train"]) == (42, 4)

    def test_default_sizes(self):
        s = split_records(make_records(1300, 250))
        assert groups(s["test"]) == (500, 100)
        assert groups(s["validation"]) == (500, 100)
        assert groups(s["train"]) == (300, 50)

    def test_num_tables_counted_when_missing(self):
        recs = make_records(30, 6, with_num_tables=False)
        s = split_records(recs, **SIZES)
        assert sum(r["context"] == MULTI_SCHEMA for r in s["test"]) == 3
        assert sum(r["context"] == MULTI_SCHEMA for r in s["validation"]) == 2

    def test_disjoint_and_complete(self):
        s = split_records(self.records, **SIZES)
        qs = [r["question"] for part in s.values() for r in part]
        assert sorted(qs) == sorted(r["question"] for r in self.records)
        assert len(set(qs)) == len(qs)

    def test_groups_interleaved(self):
        s = split_records(self.records, **SIZES)
        kinds = [r["num_tables"] > 1 for r in s["test"]]
        assert kinds != sorted(kinds) and kinds != sorted(kinds, reverse=True)

    def test_deterministic(self):
        a = split_records(self.records, **SIZES, seed=42)
        b = split_records(self.records, **SIZES, seed=42)
        c = split_records(self.records, **SIZES, seed=7)
        assert a == b and a != c

    def test_does_not_mutate_input(self):
        before = [dict(r) for r in self.records]
        split_records(self.records, **SIZES)
        assert self.records == before

    def test_group_too_small(self):
        with pytest.raises(ValueError, match="multi_table"):
            split_records(make_records(60, 4), **SIZES)

    def test_exactly_enough_leaves_train_without_group(self):
        s = split_records(make_records(60, 5), **SIZES)
        assert groups(s["train"])[1] == 0

    def test_bad_args(self):
        with pytest.raises(ValueError, match="unknown"):
            split_records(self.records, test_sizes={"three_table": 1}, val_sizes={})
        with pytest.raises(ValueError, match="non-negative"):
            split_records(self.records, test_sizes={"single_table": -1}, val_sizes={})
