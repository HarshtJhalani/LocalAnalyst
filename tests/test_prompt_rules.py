
# ============================================================
# FILE: tests/test_prompt_rules.py
# PURPOSE: SQL prompt rules for "<number> by <category>", missing values,
#          "which rows" and kept groupings; the follow-up rewrite guard;
#          the dev questions added for them
# ============================================================

import json
import sqlite3

import pytest

import src.helpers.chat_memory as chat_memory
import src.helpers.pipeline as pipeline
from config import DEV_QUESTIONS_PATH
from src.helpers.chat_memory import is_self_contained, rewrite_question
from src.helpers.sql_generator import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, build_messages

from tests.test_pipeline_memory import SETTINGS, scripted_model

SALES_COLUMNS = ["order_id", "order_date", "region", "units_sold", "total_sales", "paid"]


# ---- the SQL prompt -----------------------------------------------

def test_prompt_has_the_new_rules():
    assert "SUM(<number>)" in SYSTEM_PROMPT
    assert "IS NULL (never = 0)" in SYSTEM_PROMPT
    assert "COUNT(*) ... WHERE <column> IS NULL" in SYSTEM_PROMPT
    assert '"Which rows/orders/records ..."' in SYSTEM_PROMPT
    assert '"by <X>" or "per <X>" needs GROUP BY <X>' in SYSTEM_PROMPT


def test_new_examples_use_the_made_up_orders_table():
    examples = {question: answer["sql"] for question, answer in FEW_SHOT_EXAMPLES}

    assert "SUM(quantity)" in examples["Plot quantity by store."]
    assert "WHERE quantity IS NULL" in examples["Which orders have no quantity recorded?"]
    assert "GROUP BY" not in examples["Which orders have no quantity recorded?"]
    by_store = examples["How many orders are missing a quantity, by store?"]
    assert "COUNT(*)" in by_store and "IS NULL" in by_store and "GROUP BY store" in by_store

    # Examples never mention the real datasets.
    assert not any("messy_sales" in sql or "units_sold" in sql for sql in examples.values())


def test_prompt_fits_the_small_context(sales):
    dataset = pipeline.load_dataset(sales)
    messages, detail = pipeline.fit_messages({"num_ctx": 4096}, dataset,
                                             "How many orders are missing units sold, by region")

    assert messages is not None and detail == "full"
    assert len(build_messages(dataset["schema_text"], "q")) == 2 * len(FEW_SHOT_EXAMPLES) + 2


# ---- the follow-up rewrite guard ----------------------------------

@pytest.mark.parametrize("question", [
    "bar plot of units_sold by region",
    "Which rows have no units sold?",
    "How many orders are missing units sold, by region",
])
def test_self_contained_questions_skip_the_rewrite(monkeypatch, question):
    def no_model(*args, **kwargs):
        raise AssertionError("the rewrite model must not be called")

    monkeypatch.setattr(chat_memory, "chat_json", no_model)
    history = [{"question": "Total sales by region", "standalone_question": None,
                "sql": "SELECT 1", "preview": None, "row_count": 0}]

    assert is_self_contained(question, SALES_COLUMNS)
    assert rewrite_question(SETTINGS, history, question, SALES_COLUMNS) == {
        "uses_history": False, "standalone_question": question, "seconds": 0}


@pytest.mark.parametrize("question", [
    "only paid orders", "now by region", "what about the West region?",
    "and how much did it make in total",          # no column named
    "show me x and y values",                      # 2-letter names ignored
])
def test_follow_ups_still_rewritten(question):
    assert not is_self_contained(question, SALES_COLUMNS + ["x", "y"])


def test_ask_in_chat_passes_the_column_names(monkeypatch, sales):
    seen = {}

    def fake_rewrite(settings, history, question, columns=None):
        seen["columns"] = columns
        return {"uses_history": False, "standalone_question": question, "seconds": 0}

    monkeypatch.setattr(pipeline, "rewrite_question", fake_rewrite)
    scripted_model(monkeypatch, [{"sql": "SELECT COUNT(*) AS n FROM messy_sales"}] * 2)
    first = pipeline.ask_in_chat(SETTINGS, sales, None, "How many orders?")
    pipeline.ask_in_chat(SETTINGS, sales, first["chat_id"], "Which rows have no units sold?")

    assert "units_sold" in seen["columns"] and "region" in seen["columns"]


# ---- dev questions (tuning set; eval/ stays held out) -------------

NEW_DEV_QUESTIONS = {
    "bar plot of units_sold by region": [("North", 119), ("South", 21), ("West", 36)],
    "Which rows have no units sold?": [("00104",), ("00109",)],                  # 00113 had "twelve" units,
    "How many orders are missing units sold, by region": [("North", 2)],
}


def test_dev_questions_added_with_expected_results(sales):
    questions = json.loads(DEV_QUESTIONS_PATH.read_text(encoding="utf-8"))
    by_text = {item["question"]: item for item in questions}

    with sqlite3.connect(sales["database"]) as connection:
        for question, expected in NEW_DEV_QUESTIONS.items():
            item = by_text[question]
            assert item["dataset"] == "messy_sales"
            rows = connection.execute(item["expected_sql"]).fetchall()
            got = sorted(tuple(row[:len(expected[0])]) for row in rows)
            assert got == sorted(expected)
    connection.close()
