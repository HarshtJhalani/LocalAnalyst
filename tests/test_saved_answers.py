
# ============================================================
# FILE: tests/test_saved_answers.py
# PURPOSE: Summaries see pre-formatted numbers; each turn keeps its
#          summary, caveats and chart; older chats.db files upgrade
# ============================================================

import sqlite3

import pandas as pd

from src.helpers import answer_summary
from src.helpers.answer_summary import drop_invented_currency, result_as_text, summarize
from src.helpers.chart_builder import choose_chart
from src.helpers.chat_memory import add_turn, create_chat, load_turns, save_presentation
from src.helpers.pipeline import error_feedback


def test_result_text_has_formatted_numbers():
    frame = pd.DataFrame({"region": ["North", "South"], "sales": [11641854, 585250],
                          "share": [0.71875, 0.28125], "paid": [True, False]})
    text = result_as_text(frame)

    assert "North | 11,641,854 | 0.7188 | True" in text
    assert "South | 585,250 | 0.2812 | False" in text


def test_model_summary_with_copied_numbers_is_kept(monkeypatch):
    monkeypatch.setattr(answer_summary, "chat_json", lambda settings, messages, schema: (
        {"summary": "South had $585,250 in sales."}, {}))
    frame = pd.DataFrame({"region": ["North", "South"], "sales": [11641854, 585250]})
    result = summarize({}, "Sales by region?", "SELECT ...", frame, choose_chart(frame))

    assert result["source"] == "model"
    assert result["summary"] == "South had 585,250 in sales."


def test_missing_label_stays_visible():
    frame = pd.DataFrame({"town": [None, "Cherbourg"], "survivors": [2, 93]})

    assert "(blank) | 2" in result_as_text(frame)


def test_copied_table_is_not_a_summary(monkeypatch):
    replies = iter(["2.06 | cut | color\n2.14 | cut | color", "Diamonds of 2.06 and 2.14 carats matched."])
    monkeypatch.setattr(answer_summary, "chat_json", lambda settings, messages, schema: (
        {"summary": next(replies)}, {}))
    frame = pd.DataFrame({"carat": [2.06, 2.14], "cut": ["Fair", "Good"]})
    result = summarize({}, "List big diamonds", "SELECT ...", frame, choose_chart(frame))

    assert result["source"] == "model" and result["attempts"] == 2
    assert result["summary"] == "Diamonds of 2.06 and 2.14 carats matched."


def test_no_data_summary_rejected_when_rows_exist(monkeypatch):
    replies = iter(["No data matched the question.", "Cherbourg had 93 survivors."])
    monkeypatch.setattr(answer_summary, "chat_json", lambda settings, messages, schema: (
        {"summary": next(replies)}, {}))
    frame = pd.DataFrame({"town": [None, "Cherbourg"], "survivors": [2, 93]})
    result = summarize({}, "Survivors per town?", "SELECT ...", frame, choose_chart(frame))

    assert result["summary"] == "Cherbourg had 93 survivors."


def test_summary_without_numbers_rejected(monkeypatch):
    replies = iter(["No recorded units sold", "North has 3 orders with no units sold."])
    sent = []

    def fake(settings, messages, schema):
        sent.append(messages)
        return {"summary": next(replies)}, {}

    monkeypatch.setattr(answer_summary, "chat_json", fake)
    frame = pd.DataFrame({"region": ["North"], "missing_orders": [3]})
    result = summarize({}, "Missing units by region?", "SELECT ...", frame, choose_chart(frame))

    assert result["summary"] == "North has 3 orders with no units sold."
    assert "(blank)" not in sent[0][0]["content"]          # rule only when needed


def test_blank_rule_added_only_for_blank_cells(monkeypatch):
    sent = []
    monkeypatch.setattr(answer_summary, "chat_json", lambda settings, messages, schema: (
        sent.append(messages) or {"summary": "Cherbourg had 93 survivors."}, {}))
    frame = pd.DataFrame({"town": [None, "Cherbourg"], "survivors": [2, 93]})
    summarize({}, "Survivors per town?", "SELECT ...", frame, choose_chart(frame))

    assert "'(blank)' cell" in sent[0][0]["content"]


def test_unknown_column_feedback_lists_real_columns():
    profile = {"tables": {"titanic": {"columns": {"survived": {}, "class": {}}}}}

    feedback = error_feedback("no such column: order_id", profile)

    assert "titanic: survived, class" in feedback and "order_id does not exist" in feedback
    assert "only columns" not in error_feedback("syntax error", profile)


def test_currency_kept_when_the_data_has_it():
    frame = pd.DataFrame({"price": ["$5"]})

    assert drop_invented_currency("It costs $5.", frame) == "It costs $5."


def test_turn_keeps_summary_caveats_and_chart(tmp_path):
    db = tmp_path / "chats.db"
    chat_id = create_chat(db, "Sales by region")
    frame = pd.DataFrame({"region": ["North"], "sales": [10]})
    position = add_turn(db, chat_id, {"question": "Sales by region", "status": "answered",
                                      "sql": "SELECT 1", "frame": frame, "row_count": 1})

    assert load_turns(db, chat_id)[0]["summary"] is None

    save_presentation(db, chat_id, position, {
        "summary": {"summary": "North sold 10."}, "caveats": ["Watch out."],
        "chart": {"type": "bar", "x": "sales", "y": "region"},
        "figures": {"light": '{"data": []}', "dark": None}})
    turn = load_turns(db, chat_id)[0]

    assert turn["summary"] == "North sold 10."
    assert turn["caveats"] == ["Watch out."]
    assert turn["chart"]["type"] == "bar"
    assert turn["figures"] == {"light": '{"data": []}', "dark": None}


def test_old_chats_db_gets_new_columns(tmp_path):
    db = tmp_path / "chats.db"

    # A chats.db from before summaries were stored.
    with sqlite3.connect(db) as connection:
        connection.executescript("""
            CREATE TABLE chats (chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE turns (turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL, position INTEGER NOT NULL, question TEXT NOT NULL,
                standalone_question TEXT, uses_history INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL, sql TEXT, plan TEXT, message TEXT,
                row_count INTEGER NOT NULL DEFAULT 0, columns_json TEXT, preview_json TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, seconds REAL, created_at TEXT NOT NULL);
            INSERT INTO chats VALUES (1, 'old', '2026-01-01', '2026-01-01');
            INSERT INTO turns (chat_id, position, question, status, created_at)
                VALUES (1, 1, 'old question', 'answered', '2026-01-01');
        """)
    connection.close()

    turn = load_turns(db, 1)[0]

    assert turn["question"] == "old question"
    assert turn["summary"] is None and turn["caveats"] == [] and turn["figures"] is None
