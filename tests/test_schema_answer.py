
# ============================================================
# FILE: tests/test_schema_answer.py
# PURPOSE: Structure questions answered from the profile (no SQL), the
#          pragma safety net, schema turns in chat memory, and the loop
#          stopping on any repeated query
# ============================================================

import pytest

import src.helpers.pipeline as pipeline
from src.helpers.chat_memory import load_turns, recent_history
from src.helpers.pipeline import load_dataset
from src.helpers.schema_answer import is_schema_question, mentions_schema_sql, schema_answer
from src.helpers.sql_guardrails import check_sql

from tests.test_pipeline_memory import SETTINGS, scripted_model


# ---- recognising structure questions ------------------------------

@pytest.mark.parametrize("question", [
    "what columns are there? name them", "What are the columns?", "column names please",
    "list the columns", "List all fields", "what fields does it have",
    "what tables are there", "describe the data", "Describe this dataset",
    "what data do I have", "show me the schema", "how many columns are there",
    "which columns are available", "what's in the data",
])
def test_schema_questions_detected(question):
    assert is_schema_question(question)


@pytest.mark.parametrize("question", [
    "which column has the highest sales", "total sales by region",
    "What is the average price by cut?", "Show the monthly trend.",
    "How many rows are there?", "List all orders above 5000",
    "which product sold the most", "list customers by total sales",
    "describe the sales trend by month", "what tables were sold most",
])
def test_data_questions_not_detected(question):
    assert not is_schema_question(question)


def test_catalog_sql_detected_and_still_blocked(sales):
    allowed = load_dataset(sales)["allowed_tables"]

    for sql in ("SELECT name FROM pragma_table_info('messy_sales')",
                "SELECT sql FROM sqlite_master", "select * from SQLITE_SCHEMA"):
        assert mentions_schema_sql(sql)
        assert not check_sql(sql, allowed)["ok"]          # guardrail unchanged

    assert not mentions_schema_sql("SELECT region FROM messy_sales")


# ---- the answer ---------------------------------------------------

def test_schema_answer_lists_cleaned_columns_only(sales):
    dataset = load_dataset(sales)
    answer = schema_answer(dataset["profile"], dataset["allowed_tables"])
    frame = answer["frame"]

    assert list(frame.columns) == ["table", "column", "type", "example", "missing"]
    assert set(frame["table"]) == {"messy_sales"}
    assert not frame["table"].str.endswith("_raw").any()
    assert {"region", "total_sales", "order_date"} <= set(frame["column"])
    assert answer["summary"].startswith(f"messy_sales has {len(frame)} columns: ")
    assert 3 <= len(answer["suggestions"]) <= 5
    assert "Total sales by region" in answer["suggestions"]
    assert "Total sales per month" in answer["suggestions"]


def test_schema_question_skips_the_model(monkeypatch, sales):
    calls = scripted_model(monkeypatch, [])
    outcome = pipeline.answer_question(SETTINGS, sales, "what columns are there? name them")

    assert calls == []                                     # no SQL model call
    assert outcome["status"] == "schema" and outcome["sql"] is None
    assert outcome["attempts"] == [] and outcome["row_count"] == len(outcome["frame"])
    assert outcome["suggestions"]

    presented = pipeline.present_answer(SETTINGS, outcome, load_dataset(sales)["profile"])
    assert presented["chart"]["type"] == "table"
    assert presented["summary"]["source"] == "schema"


def test_pragma_sql_is_answered_from_the_profile(monkeypatch, sales):
    calls = scripted_model(monkeypatch, [
        {"sql": "SELECT name FROM pragma_table_info('messy_sales')"},
    ])
    outcome = pipeline.answer_question(SETTINGS, sales, "Give me every header used")

    assert len(calls) == 1                                 # no retries
    assert outcome["status"] == "schema"
    assert outcome["attempts"][0]["outcome"] == "schema_redirect"


# ---- the loop stops on any repeated query -------------------------

def test_repeated_non_consecutive_sql_stops(monkeypatch, sales):
    bad_a = "SELECT nope FROM messy_sales"
    bad_b = "SELECT nada FROM messy_sales"
    scripted_model(monkeypatch, [{"sql": bad_a}, {"sql": bad_b},
                                 {"sql": "  select NOPE from messy_sales ; "},
                                 {"sql": "SELECT COUNT(*) AS n FROM messy_sales"}])
    outcome = pipeline.answer_question(SETTINGS, sales, "Total nope?")

    assert outcome["status"] == "failed" and len(outcome["attempts"]) == 3


def test_repeated_empty_query_is_accepted(monkeypatch, sales):
    empty = "SELECT * FROM messy_sales WHERE city = 'Atlantis'"
    scripted_model(monkeypatch, [{"sql": empty}, {"sql": empty.lower() + ";"}])
    outcome = pipeline.answer_question(SETTINGS, sales, "Orders in Atlantis?")

    assert outcome["status"] == "empty" and len(outcome["attempts"]) == 2


# ---- chat memory --------------------------------------------------

def test_schema_turn_stored_but_not_used_as_context(monkeypatch, sales):
    scripted_model(monkeypatch, [])
    result = pipeline.run_question(SETTINGS, sales, None, "describe the data")
    turn = load_turns(sales["chats"], result["chat_id"])[0]

    assert turn["status"] == "schema" and turn["sql"] is None
    assert turn["summary"].startswith("messy_sales has")
    assert turn["suggestions"] == result["suggestions"]
    assert turn["preview"] is not None and not turn["preview"].empty
    assert recent_history(sales["chats"], result["chat_id"]) == []
