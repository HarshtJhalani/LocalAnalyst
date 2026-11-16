
# ============================================================
# FILE: src/helpers/sqlite_store.py
# PURPOSE: Write cleaned tables to SQLite and add the statistics
#          functions SQLite lacks
#          (pipeline stage 3; 02_sqlite_profiling.ipynb, steps 3-4)
#
# SQLite has no date or boolean type: dates are stored as ISO text
# (YYYY-MM-DD), booleans as 1 / 0. Every table also gets a
# <table>_raw copy with the original text values.
#
#   write_dataset()        cleaned + raw tables of one file -> dataset.db
#   register_functions()   median_of, stdev_of, variance_of, percentile_of
# ============================================================

import json
import logging
import math
import sqlite3
import statistics
from pathlib import Path

import pandas as pd

from config import INSERT_BATCH_SIZE, RAW_TABLE_SUFFIX, SQLITE_TYPES


logger = logging.getLogger(__name__)


# ============================================================
# 1. WRITING TABLES
# ============================================================

def quote_identifier(name):
    """Quote a table or column name for SQL."""

    return '"' + str(name).replace('"', '""') + '"'


def to_sqlite_column(series, kind):
    """Convert one column to plain Python values SQLite accepts."""

    if kind == "date":
        values = series.dt.strftime("%Y-%m-%d")
    elif kind == "datetime":
        values = series.dt.strftime("%Y-%m-%d %H:%M:%S")
    elif kind in {"boolean", "integer"}:
        values = series.astype("Int64")
    elif kind == "float":
        values = series.astype("Float64")
    else:
        values = series.astype("string")

    # Missing values of every kind (NA, NaT, NaN) become None.
    values = values.astype(object)

    return values.where(values.notna(), None)


def create_table(connection, table_name, column_types):
    """Drop and recreate one table with explicit column types."""

    columns_sql = ",\n    ".join(
        f"{quote_identifier(column)} {SQLITE_TYPES[kind]}"
        for column, kind in column_types.items()
    )

    connection.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")
    connection.execute(
        f"CREATE TABLE {quote_identifier(table_name)} (\n    {columns_sql}\n)"
    )


def insert_rows(connection, table_name, frame, column_types):
    """Insert a DataFrame in batches. Returns the number of rows."""

    columns = list(column_types)
    prepared = pd.DataFrame({
        column: to_sqlite_column(frame[column], column_types[column])
        for column in columns
    })

    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(quote_identifier(column) for column in columns)
    sql = (f"INSERT INTO {quote_identifier(table_name)} ({column_sql}) "
           f"VALUES ({placeholders})")

    rows = prepared.values.tolist()

    for start in range(0, len(rows), INSERT_BATCH_SIZE):
        connection.executemany(sql, rows[start:start + INSERT_BATCH_SIZE])

    return len(rows)


def write_dataset(database_path, tables):
    """
    Write every table of one dataset (cleaned + raw) into SQLite.

    Args:
        tables: [{"table_name", "cleaned", "raw", "report"}]

    Returns:
        dict: table name -> {"rows", "raw_table", "raw_rows"}
    """

    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    written = {}

    connection = sqlite3.connect(database_path)

    try:
        with connection:
            for table in tables:
                name = table["table_name"]
                raw_name = f"{name}{RAW_TABLE_SUFFIX}"
                column_types = table["report"]["column_types"]

                # Step 1: cleaned table with real types.
                create_table(connection, name, column_types)
                rows = insert_rows(connection, name, table["cleaned"], column_types)

                # Step 2: raw table, every column text.
                raw_types = {column: "text" for column in table["raw"].columns}
                create_table(connection, raw_name, raw_types)
                raw_rows = insert_rows(connection, raw_name, table["raw"], raw_types)

                written[name] = {"rows": rows, "raw_table": raw_name,
                                 "raw_rows": raw_rows}

                logger.info("Wrote %s (%d rows) and %s (%d rows) to %s",
                            name, rows, raw_name, raw_rows, database_path.name)

    except sqlite3.Error as exc:
        raise OSError(f"Could not write {database_path}: {exc}") from exc

    finally:
        connection.close()

    return written


# ============================================================
# 2. STATISTICS SQLITE DOES NOT HAVE
# ============================================================
# Proper SQLite aggregates need a class, so these are plain functions
# that take a JSON array built by json_group_array():
#   SELECT city, median_of(json_group_array(total_sales)) FROM t GROUP BY city
#
# Called the wrong way (median_of(col) without json_group_array, or a
# percent outside 0-100) they raise, so SQLite fails the query and the
# retry loop can tell the model what to fix. Returning NULL instead
# would pass as a real answer.

def json_numbers(json_array):
    """
    '[1, null, 3]' -> [1.0, 3.0]; values that are not numbers are
    skipped. ValueError when the argument is not a JSON array.
    """

    if json_array is None:
        return []

    try:
        values = json.loads(json_array)

    except (TypeError, ValueError):
        values = None

    if not isinstance(values, list):
        raise ValueError("expected a JSON array from json_group_array(column)")

    numbers = []

    for value in values:
        if isinstance(value, bool) or value is None:
            continue

        try:
            number = float(value)

        except (TypeError, ValueError):
            continue

        if not math.isnan(number):
            numbers.append(number)

    return numbers


def median_of(json_array):
    numbers = json_numbers(json_array)
    return statistics.median(numbers) if numbers else None


def stdev_of(json_array):
    """Sample standard deviation (needs at least two values)."""

    numbers = json_numbers(json_array)
    return statistics.stdev(numbers) if len(numbers) > 1 else None


def variance_of(json_array):
    numbers = json_numbers(json_array)
    return statistics.variance(numbers) if len(numbers) > 1 else None


def percentile_of(json_array, percent):
    """
    Linear-interpolated percentile, percent from 0 to 100.

    A percent between 0 and 1 is refused: it is almost always a
    fraction (0.9 written for the 90th percentile), which would
    silently return a value near the minimum.
    """

    numbers = sorted(json_numbers(json_array))

    if percent is None:
        return None

    if isinstance(percent, bool) or not isinstance(percent, (int, float)) \
            or not 0 <= percent <= 100 or 0 < percent < 1:
        raise ValueError("percent must be a number from 0 to 100, e.g. 90")

    if not numbers:
        return None

    position = (len(numbers) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    return numbers[lower] + (numbers[upper] - numbers[lower]) * (position - lower)


SQL_FUNCTIONS = {
    "median_of": (median_of, 1),
    "stdev_of": (stdev_of, 1),
    "variance_of": (variance_of, 1),
    "percentile_of": (percentile_of, 2),
}


def register_functions(connection):
    """Add the statistics functions to one SQLite connection."""

    for name, (function, arg_count) in SQL_FUNCTIONS.items():
        connection.create_function(name, arg_count, function, deterministic=True)

    return connection
