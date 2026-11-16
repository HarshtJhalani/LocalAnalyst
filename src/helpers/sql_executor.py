
# ============================================================
# FILE: src/helpers/sql_executor.py
# PURPOSE: Run one query safely
#          (05_self_correction_guardrails.ipynb, step 5)
#
# - Read-only connection (mode=ro): SQLite itself refuses writes.
# - Authorizer: only SELECT, reads of allowed tables (the real table is
#   reported even through a CTE; reads of a CTE report no database and
#   pass), allowed functions and recursive CTEs; everything else denied.
# - Timeout: a progress handler aborts after QUERY_TIMEOUT_SECONDS.
# - Row cap: at most MAX_RESULT_ROWS rows are fetched.
#
#   execute_sql()   -> {"ok", "frame", "row_count", "truncated",
#                       "seconds", "error"}
# ============================================================

import sqlite3
import time
from pathlib import Path

import pandas as pd

from config import MAX_RESULT_ROWS, PROGRESS_CHECK_STEPS, QUERY_TIMEOUT_SECONDS
from src.helpers.sqlite_store import register_functions


# ============================================================
# 1. ALLOWED FUNCTIONS
# ============================================================


ALLOWED_SQL_FUNCTIONS = {
    "abs", "avg", "cast", "ceil", "ceiling", "coalesce", "count", "date",
    "datetime", "floor", "group_concat", "ifnull", "iif", "instr", "julianday",
    "length", "lower", "ltrim", "max", "min", "nullif", "printf", "format",
    "replace", "round", "rtrim", "substr", "substring", "sum", "strftime",
    "time", "total", "trim", "typeof", "upper", "unixepoch", "sqrt", "power",
    "pow", "ln", "log", "log10", "log2", "exp", "mod", "sign", "trunc",
    "json_group_array", "json_array", "json_extract", "row_number", "rank",
    "dense_rank", "ntile", "lag", "lead", "first_value", "last_value",
    "percent_rank", "cume_dist", "nth_value", "like", "glob", "char",
    "median_of", "stdev_of", "variance_of", "percentile_of",
}


# ============================================================
# 2. AUTHORIZER AND TIMEOUT
# ============================================================


def make_authorizer(allowed_tables):
    """SQLite authorizer allowing only reads of `allowed_tables`."""

    allowed = {table.lower() for table in allowed_tables}

    def authorizer(action, arg1, arg2, database, source):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK

        if action == sqlite3.SQLITE_READ:
            # A CTE or subquery reports database None; the real tables
            # it reads are authorized separately (database "main").
            if database is None:
                return sqlite3.SQLITE_OK

            # arg1 = table name (the real table, even through a CTE)
            return (sqlite3.SQLITE_OK if arg1 and arg1.lower() in allowed
                    else sqlite3.SQLITE_DENY)

        if action == sqlite3.SQLITE_FUNCTION:
            # arg2 = function name
            return (sqlite3.SQLITE_OK if arg2 and arg2.lower() in ALLOWED_SQL_FUNCTIONS
                    else sqlite3.SQLITE_DENY)

        if action == sqlite3.SQLITE_RECURSIVE:
            return sqlite3.SQLITE_OK

        return sqlite3.SQLITE_DENY

    return authorizer


def make_timeout_handler(timeout_seconds):
    """Progress handler that aborts the query after the deadline."""

    deadline = time.monotonic() + timeout_seconds

    def handler():
        return 1 if time.monotonic() > deadline else 0

    return handler


def friendly_sqlite_error(exc, timeout_seconds):
    """Turn SQLite errors into messages the model (and user) can act on."""

    message = str(exc)

    if "interrupted" in message:
        return (f"The query took longer than {timeout_seconds} s and was "
                "stopped. Simplify it or aggregate earlier.")

    if "not authorized" in message or "prohibited" in message:
        return ("The query tried to read a table or call a function that is "
                "not allowed. Use only the tables and columns in the schema.")

    # SQLite hides the Python error text of a failing registered function.
    if "user-defined function raised exception" in message:
        return ("A statistics function was called the wrong way. Pass a JSON "
                "array built from the column: median_of(json_group_array(col)), "
                "stdev_of(json_group_array(col)), variance_of(json_group_array(col)), "
                "percentile_of(json_group_array(col), 90) with the percent "
                "from 0 to 100.")

    if "readonly" in message or "read-only" in message:
        return "The query tried to change data; only reading is allowed."

    return message


# ============================================================
# 3. EXECUTION
# ============================================================


def execute_sql(db_path, sql, allowed_tables,
                timeout_seconds=QUERY_TIMEOUT_SECONDS,
                max_rows=MAX_RESULT_ROWS):
    """
    Run one query safely.

    Returns:
        {"ok", "frame", "row_count", "truncated", "seconds", "error"}
    """

    started = time.perf_counter()
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)

    try:
        # Step 1: statistics functions, then lock the connection down.
        register_functions(connection)
        connection.set_authorizer(make_authorizer(allowed_tables))
        connection.set_progress_handler(make_timeout_handler(timeout_seconds),
                                        PROGRESS_CHECK_STEPS)

        # Step 2: run and fetch one row more than the cap.
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [description[0] for description in cursor.description or []]

    except sqlite3.Error as exc:
        return {"ok": False, "frame": None, "row_count": 0, "truncated": False,
                "seconds": round(time.perf_counter() - started, 3),
                "error": friendly_sqlite_error(exc, timeout_seconds)}

    finally:
        connection.close()

    truncated = len(rows) > max_rows
    frame = pd.DataFrame(rows[:max_rows], columns=columns)

    return {"ok": True, "frame": frame, "row_count": len(frame),
            "truncated": truncated,
            "seconds": round(time.perf_counter() - started, 3), "error": None}
