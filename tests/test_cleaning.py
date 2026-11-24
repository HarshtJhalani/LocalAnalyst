
# ============================================================
# FILE: tests/test_cleaning.py
# PURPOSE: file_loader + data_cleaner on the sample and held-out files
#          (checks from 01_loading_cleaning.ipynb and the cleaning part
#          of 08_evaluation.ipynb)
# ============================================================

import json

import pandas as pd
import pytest

from config import EVAL_DIR, INPUT_DIR
from src.evaluation.run_evaluation import grade_table
from src.helpers.data_cleaner import (
    convert_column,
    load_and_clean,
    merge_category_variants,
    new_report,
    parse_number,
    type_matches,
)
from src.helpers.file_loader import detect_delimiter, read_file, validate_file


def flags_of(report, issue):
    return [flag["column"] for flag in report["flags"] if flag["issue"] == issue]


@pytest.fixture(scope="module")
def sales_result():
    return load_and_clean(INPUT_DIR / "messy_sales.csv")[0]


def test_sales_header_and_names(sales_result):
    report = sales_result["report"]

    assert report["header_row"] == 3
    assert list(sales_result["cleaned"].columns) == [
        "order_id", "order_date", "customer_name", "city", "region", "product",
        "units_sold", "unit_price", "discount_pct", "total_sales", "paid",
        "sales_rep", "sales_rep_2", "notes",
    ]


def test_sales_rows_and_types(sales_result):
    frame, report = sales_result["cleaned"], sales_result["report"]
    types = report["column_types"]

    assert report["rows_before"] == 35 and len(frame) == 32
    assert types["order_id"] == "text"
    assert types["order_date"] == "date"
    assert types["units_sold"] == "integer" and frame["units_sold"].isna().sum() == 2         # "twelve" is read as 12
    assert types["paid"] == "boolean"
    assert report["column_units"] == {"unit_price": "currency", "discount_pct": "percent",
                                      "total_sales": "currency"}


def test_sales_flags(sales_result):
    report = sales_result["report"]

    assert "order_id" in flags_of(report, "identifier_kept_as_text")
    assert flags_of(report, "outliers") == ["total_sales"]
    assert "city" in flags_of(report, "similar_categories")
    assert set(sales_result["cleaned"]["city"]) == {
        "Delhi", "Mumbai", "Mumbay", "Bengaluru", "Pune", "New Delhi"}


def test_excel_sheets_and_flags():
    results = {r["table_name"]: r for r in load_and_clean(INPUT_DIR / "messy_employees.xlsx")}
    employees = results["messy_employees_employees"]["report"]

    assert set(results) == {"messy_employees_employees", "messy_employees_targets"}
    assert employees["header_row"] == 2
    assert "salary" in flags_of(employees, "partly_numeric")       # 2 of 20 not numbers
    assert "last_review" in flags_of(employees, "ambiguous_dates")


def test_json_and_parquet():
    customers = load_and_clean(INPUT_DIR / "customers.json")[0]
    products = load_and_clean(INPUT_DIR / "products.parquet")[0]

    assert {"address_city", "address_pin"} <= set(customers["cleaned"].columns)
    assert customers["report"]["column_types"]["signup"] == "date"
    assert products["report"]["column_types"]["list_price"] == "integer"


def test_table_base_name_keeps_the_uploaded_name(tmp_path):
    stored = tmp_path / "upload_123.csv"
    stored.write_bytes((INPUT_DIR / "messy_sales.csv").read_bytes())

    assert load_and_clean(stored, table_base_name="Sales Report.csv")[0]["table_name"] \
        == "sales_report"


@pytest.mark.parametrize("text, expected", [
    ("$1,200.50", (1200.5, "currency")),
    ("Rs 1,250.50", (1250.5, "currency")),
    ("12.5%", (12.5, "percent")),
    ("(1,200)", (-1200.0, None)),
    ("98765 43210", None),
    ("abc", None),
])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_single_failure_rule():
    assert type_matches(19, 20) and type_matches(17, 18)
    assert not type_matches(16, 18) and not type_matches(2, 3)

    frame = pd.DataFrame({"hours": [str(i) for i in range(17)] + ["unknown"]})
    report = new_report("t", "t.csv")
    frame = convert_column(frame, "hours", report)

    assert report["column_types"]["hours"] == "integer"
    assert frame["hours"].isna().sum() == 1


def test_category_merge_prefers_normal_casing():
    frame = pd.DataFrame({"c": ["BEVERAGES"] * 3 + ["Beverages", "beverages",
                                                    "hr", "hr", "HR"] + ["x"] * 4})
    report = new_report("t", "t.csv")
    report["column_types"]["c"] = "text"

    assert set(merge_category_variants(frame, report)["c"]) == {"Beverages", "HR", "x"}


def test_validation_errors(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    wrong = tmp_path / "data.exe"
    wrong.write_text("x")

    with pytest.raises(ValueError):
        validate_file(empty)
    with pytest.raises(ValueError):
        validate_file(wrong)
    with pytest.raises(FileNotFoundError):
        read_file(tmp_path / "missing.csv")


def test_delimiter_ignores_title_lines(tmp_path):
    text = "Report title\nA;B;C\n1;2;3\n4;5;6\n"

    assert detect_delimiter(text, tmp_path / "x.csv") == ";"


@pytest.mark.parametrize("file_name", ["eval_inventory.csv", "eval_tickets.xlsx"])
def test_held_out_files_match_answer_key(file_name):
    truth = json.loads((EVAL_DIR / "eval_cleaning_truth.json").read_text(encoding="utf-8"))
    results = {r["table_name"]: r for r in load_and_clean(EVAL_DIR / file_name)}

    for table_name, expected in truth[file_name]["tables"].items():
        failed = [(check, detail) for check, passed, detail
                  in grade_table(results[table_name], expected) if not passed]
        assert failed == []
