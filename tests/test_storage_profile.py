
# ============================================================
# FILE: tests/test_storage_profile.py
# PURPOSE: sqlite_store + data_profiler + ingestion
#          (checks from 02_sqlite_profiling.ipynb)
# ============================================================

import sqlite3

import pandas as pd

from src.helpers.pipeline import load_dataset, read_status
from src.helpers.sqlite_store import median_of, percentile_of, register_functions, stdev_of


def query(paths, sql):
    with sqlite3.connect(paths["database"]) as connection:
        register_functions(connection)
        return connection.execute(sql).fetchall()


def test_ingestion_status_and_files(sales):
    status = read_status(sales)

    assert status["status"] == "completed" and status["stage"] == "ready"
    assert status["tables"][0]["rows"] == 32
    for key in ("database", "profile", "schema", "cleaning_report"):
        assert sales[key].is_file()


def test_cleaned_and_raw_tables(sales):
    assert query(sales, "SELECT COUNT(*) FROM messy_sales")[0][0] == 32
    assert query(sales, "SELECT COUNT(*) FROM messy_sales_raw")[0][0] == 35

    types = {row[1]: row[2] for row in query(sales, "PRAGMA table_info(messy_sales)")}
    assert types["order_date"] == "TEXT" and types["paid"] == "INTEGER"
    assert {row[2] for row in query(sales, "PRAGMA table_info(messy_sales_raw)")} == {"TEXT"}


def test_stored_values(sales):
    assert query(sales, "SELECT order_id FROM messy_sales LIMIT 1")[0][0] == "00101"
    assert {row[0] for row in query(sales, "SELECT DISTINCT paid FROM messy_sales")} == {0, 1}
    months = query(sales, "SELECT strftime('%Y-%m', order_date) m, COUNT(*) "
                          "FROM messy_sales GROUP BY m ORDER BY m")
    assert [month for month, _ in months] == ["2024-01", "2024-02", "2024-03"]


def test_statistics_functions(sales):
    assert median_of("[]") is None and stdev_of("[5]") is None
    assert percentile_of("[1, 2, 3, 4, 5]", 50) == 3
    assert median_of("[1, null, 3, \"x\"]") == 2

    prices = [row[0] for row in query(sales, "SELECT unit_price FROM messy_sales")
              if row[0] is not None]
    assert query(sales, "SELECT median_of(json_group_array(unit_price)) "
                        "FROM messy_sales")[0][0] == pd.Series(prices).median()


def test_schema_text(sales):
    schema = load_dataset(sales)["schema_text"]

    assert "TABLE messy_sales (32 rows)" in schema
    assert "'Mumbay'" in schema and "messy_sales_raw" not in schema
    assert "range 1600 to 9999999" in schema                 # no thousands commas
    assert "order_id TEXT | unique per row" in schema
    assert "percent (10 means 10%)" in schema


def test_excel_dataset_has_both_tables(workspace):
    dataset = load_dataset(workspace["messy_employees"])

    assert dataset["allowed_tables"] == {"messy_employees_employees", "messy_employees_targets"}
    assert "unknown day/month order" in dataset["schema_text"]
