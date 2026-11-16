import pytest

from text2sql.sql_utils import (
    SchemaError,
    build_sqlite_db,
    check_sql,
    count_tables,
    error_kind,
    exact_match,
    is_read_only_sql,
    is_valid_sql,
    normalize_sql,
)

SCHEMA = "CREATE TABLE head (age INTEGER, name VARCHAR, born_state VARCHAR)"
MULTI_SCHEMA = (
    "CREATE TABLE department (department_id VARCHAR, name VARCHAR); "
    "CREATE TABLE management (department_id VARCHAR, head_id VARCHAR)"
)


class TestCheckSql:
    def test_valid_select(self):
        assert is_valid_sql("SELECT COUNT(*) FROM head WHERE age > 56", SCHEMA)

    def test_dataset_style_double_quoted_string_literal(self):
        # b-mc2/sql-create-context uses "..." for string values
        assert is_valid_sql('SELECT name FROM head WHERE born_state = "California"', SCHEMA)

    def test_multi_table_schema_join(self):
        sql = (
            "SELECT T1.name FROM department AS T1 JOIN management AS T2 "
            "ON T1.department_id = T2.department_id"
        )
        assert is_valid_sql(sql, MULTI_SCHEMA)

    def test_trailing_semicolon_ok(self):
        assert is_valid_sql("SELECT age FROM head;", SCHEMA)

    def test_unknown_column(self):
        res = check_sql("SELECT salary FROM head", SCHEMA)
        assert not res.valid and "salary" in res.error

    def test_unknown_table(self):
        assert not is_valid_sql("SELECT * FROM nope", SCHEMA)

    def test_syntax_error(self):
        assert not is_valid_sql("SELEC age FROM head", SCHEMA)

    @pytest.mark.parametrize("sql", ["", "   ", None])
    def test_empty(self, sql):
        res = check_sql(sql, SCHEMA)
        assert not res.valid and res.error == "empty SQL"

    def test_multiple_statements_rejected(self):
        assert not is_valid_sql("SELECT age FROM head; SELECT name FROM head", SCHEMA)

    def test_bad_schema_reported(self):
        res = check_sql("SELECT 1", "CREATE TABLE (oops")
        assert not res.valid and res.error.startswith("schema error")

    def test_timeout(self):
        sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
        res = check_sql(sql, SCHEMA, timeout_s=0.2)
        assert not res.valid and "timed out" in res.error

    def test_attach_denied_even_when_writes_allowed(self, tmp_path):
        target = tmp_path / "x.db"
        assert not is_valid_sql(f"ATTACH DATABASE '{target}' AS x", SCHEMA, read_only=False)
        assert not target.exists()

    def test_mutation_does_not_leak_between_checks(self):
        assert is_valid_sql("DROP TABLE head", SCHEMA, read_only=False)
        assert is_valid_sql("SELECT age FROM head", SCHEMA)


class TestReadOnly:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT age FROM head",
            "  select age from head",
            "WITH old AS (SELECT * FROM head WHERE age > 50) SELECT COUNT(*) FROM old",
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 5) SELECT x FROM c",
            "-- comment\nSELECT age FROM head",
            "/* block */ SELECT age FROM head",
            "SELECT MAX(age), LOWER(name) FROM head",
        ],
    )
    def test_read_queries_pass(self, sql):
        assert is_valid_sql(sql, SCHEMA)

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE head",
            "DELETE FROM head",
            "INSERT INTO head VALUES (1, 'a', 'b')",
            "UPDATE head SET age = 1",
            "CREATE TABLE x (a INT)",
            "PRAGMA table_info(head)",
            "VACUUM",
            "VALUES (1)",
            "selectx FROM head",
        ],
    )
    def test_non_select_rejected_by_prefix(self, sql):
        res = check_sql(sql, SCHEMA)
        assert not res.valid and "read-only" in res.error

    @pytest.mark.parametrize(
        "sql",
        [
            "WITH x AS (SELECT 1) DELETE FROM head",
            "WITH x AS (SELECT 1) INSERT INTO head SELECT 1, 'a', 'b' FROM x",
            "WITH x AS (SELECT 1) UPDATE head SET age = 2",
        ],
    )
    def test_cte_writes_rejected_by_authorizer(self, sql):
        assert is_valid_sql(sql, SCHEMA, read_only=False)  # sanity: valid SQL otherwise
        res = check_sql(sql, SCHEMA)
        assert not res.valid and "not authorized" in res.error

    def test_pragma_table_function_rejected(self):
        assert not is_valid_sql("SELECT * FROM pragma_table_info('head')", SCHEMA)

    def test_write_allowed_when_flag_off(self):
        assert is_valid_sql("DELETE FROM head", SCHEMA, read_only=False)

    @pytest.mark.parametrize(
        "sql,expected",
        [
            ("SELECT 1", True),
            ("with a as (select 1) select * from a", True),
            ("\n -- c\n /* d */ SELECT 1", True),
            ("DELETE FROM t", False),
            ("", False),
            (None, False),
        ],
    )
    def test_is_read_only_sql(self, sql, expected):
        assert is_read_only_sql(sql) is expected


class TestHelpers:
    @pytest.mark.parametrize(
        "schema,n",
        [
            (SCHEMA, 1),
            (MULTI_SCHEMA, 2),
            ("create table a (x int);CREATE   TABLE b (y int); CREATE TEMP TABLE c (z int)", 3),
            ("", 0),
            (None, 0),
        ],
    )
    def test_count_tables(self, schema, n):
        assert count_tables(schema) == n

    @pytest.mark.parametrize(
        "msg,kind",
        [
            ("table station already exists", "table already exists"),
            ("duplicate column name: Editor_ID", "duplicate column name"),
            ("no such column: catalog_level_number", "no such column"),
            ("ambiguous column name: document_id", "ambiguous column name"),
            ('near "SELEC": syntax error', "syntax error"),
            ("incomplete input", "syntax error"),
        ],
    )
    def test_error_kind(self, msg, kind):
        assert error_kind(msg) == kind


class TestBuildDb:
    def test_builds_tables(self):
        conn = build_sqlite_db(MULTI_SCHEMA)
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert names == {"department", "management"}

    def test_bad_schema_raises(self):
        with pytest.raises(SchemaError):
            build_sqlite_db("CREATE TABLE t (")

    def test_attach_in_schema_denied(self, tmp_path):
        target = tmp_path / "evil.db"
        with pytest.raises(SchemaError):
            build_sqlite_db(f"ATTACH DATABASE '{target}' AS evil; CREATE TABLE t (a INT)")
        assert not target.exists()


class TestNormalize:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("SELECT age FROM head", "select   age\nfrom head"),
            ("SELECT age FROM head;", "SELECT age FROM head"),
            ("SELECT COUNT(*) FROM head", "select count ( * ) from head"),
            ("SELECT a FROM t WHERE x = 1", "SELECT a FROM t WHERE x=1"),
            ("SELECT a FROM t WHERE x <> 1", "SELECT a FROM t WHERE x != 1"),
            ('SELECT a FROM t WHERE n = "Bob"', "SELECT a FROM t WHERE n = 'Bob'"),
            ("SELECT a, b FROM t", "SELECT a ,b FROM t"),
        ],
    )
    def test_equivalent(self, a, b):
        assert normalize_sql(a) == normalize_sql(b)
        assert exact_match(a, b)

    def test_literal_case_preserved(self):
        assert not exact_match("SELECT a FROM t WHERE n = 'Bob'", "SELECT a FROM t WHERE n = 'bob'")

    def test_literal_whitespace_preserved(self):
        assert normalize_sql("SELECT a FROM t WHERE n = 'New  York'").endswith("'New  York'")

    def test_quote_escaping(self):
        assert normalize_sql('SELECT a FROM t WHERE n = "O\'Neil"') == "select a from t where n='O''Neil'"
        assert exact_match('SELECT a FROM t WHERE n = "O\'Neil"', "SELECT a FROM t WHERE n = 'O''Neil'")

    def test_different_queries_differ(self):
        assert not exact_match("SELECT a FROM t", "SELECT b FROM t")

    def test_none(self):
        assert normalize_sql(None) == ""
