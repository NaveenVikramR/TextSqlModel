"""SQLite helpers: build a throwaway in-memory DB from a schema, check SQL validity,
and normalize SQL for exact-match comparison.

Everything here is stdlib-only and CPU-friendly so it can run in data prep, eval,
and serving without any ML dependencies.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass

DEFAULT_TIMEOUT_S = 2.0

# Actions denied while building the schema and while running candidate SQL.
# ATTACH/DETACH could touch files on disk; PRAGMA can change engine behaviour.
# NOTE: the serving layer receives user-supplied schemas, so it should add its own
# sandboxing (input size limits, statement allow-listing) on top of this.
_DENIED_ACTIONS = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA}


class SchemaError(ValueError):
    """Raised when a CREATE TABLE context cannot be loaded into SQLite."""


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    error: str | None = None


def _authorizer(action: int, *_args) -> int:
    return sqlite3.SQLITE_DENY if action in _DENIED_ACTIONS else sqlite3.SQLITE_OK


def build_sqlite_db(schema: str) -> sqlite3.Connection:
    """Create an in-memory SQLite DB and run the schema (one or more statements)."""
    conn = sqlite3.connect(":memory:")
    conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    conn.set_authorizer(_authorizer)
    try:
        conn.executescript(schema)
    except sqlite3.Error as exc:
        conn.close()
        raise SchemaError(str(exc)) from exc
    return conn


def check_sql(sql: str, schema: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> ValidationResult:
    """Return whether `sql` is a single statement that executes against `schema`.

    A fresh DB is built per call, so statements that mutate data can't leak between
    checks. Tables are empty, so this measures "parses and binds against the schema",
    not result correctness.
    """
    if sql is None or not sql.strip():
        return ValidationResult(False, "empty SQL")
    try:
        conn = build_sqlite_db(schema)
    except SchemaError as exc:
        return ValidationResult(False, f"schema error: {exc}")

    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    try:
        cur = conn.execute(sql)
        cur.fetchall()
        return ValidationResult(True)
    except (sqlite3.Error, sqlite3.Warning) as exc:
        # sqlite3.Warning covers "You can only execute one statement at a time."
        msg = str(exc)
        if msg == "interrupted":
            msg = f"timed out after {timeout_s}s"
        return ValidationResult(False, msg)
    finally:
        conn.close()


def is_valid_sql(sql: str, schema: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> bool:
    return check_sql(sql, schema, timeout_s=timeout_s).valid


# --- normalization -----------------------------------------------------------

# Single- or double-quoted string, with SQL-style doubled-quote escaping.
_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")
_PUNCT_SPACE_RE = re.compile(r"\s*([(),=<>!*+/%-])\s*")


def _normalize_literal(literal: str) -> str:
    quote = literal[0]
    inner = literal[1:-1].replace(quote * 2, quote)
    return "'" + inner.replace("'", "''") + "'"


def normalize_sql(sql: str) -> str:
    """Canonicalize SQL for exact-match comparison.

    - strips whitespace and trailing semicolons
    - lowercases everything outside string literals (literal case is preserved)
    - rewrites "double" and 'single' quoted strings to a single-quoted form
      (the source dataset uses double quotes for string values)
    - collapses whitespace and removes spaces around punctuation/operators
    - rewrites `<>` to `!=`
    """
    if sql is None:
        return ""
    sql = sql.strip().rstrip(";").strip()

    literals: list[str] = []

    def _stash(match: re.Match) -> str:
        literals.append(_normalize_literal(match.group()))
        return f"\x00{len(literals) - 1}\x00"

    code = _LITERAL_RE.sub(_stash, sql).lower()
    code = re.sub(r"\s+", " ", code)
    code = _PUNCT_SPACE_RE.sub(r"\1", code)
    code = code.replace("<>", "!=").strip()
    return _PLACEHOLDER_RE.sub(lambda m: literals[int(m.group(1))], code)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)
