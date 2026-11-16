
# ============================================================
# FILE: src/helpers/chat_memory.py
# PURPOSE: Store chats and turn follow-ups into standalone questions
#          (07_memory.ipynb)
#
# chats.db lives in the dataset folder next to the read-only dataset.db.
# Each turn keeps what the UI needs to redraw it without rerunning
# anything (question, standalone question, SQL, status, first rows).
#
# Follow-ups: with history, the model rewrites the new question into a
# standalone one using the last MEMORY_TURNS answered turns. The question
# counts as building on history when the rewrite changed it (the model's
# own flag is unreliable on small models); then the last SQL is offered
# to the SQL step as context.
#
#   create_chat / add_turn / list_chats / get_chat / load_turns /
#   rename_chat / delete_chat        storage (db_path = paths["chats"])
#   recent_history()                 last answered turns
#   rewrite_question()               follow-up -> standalone question
#   sql_context()                    last SQL, for the SQL prompt
# ============================================================

import json
import logging
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    CHAT_TITLE_LENGTH,
    HISTORY_PREVIEW_ROWS,
    MEMORY_TURNS,
    STORED_PREVIEW_ROWS,
)
from src.helpers.model_client import MODEL_ERRORS, chat_json


logger = logging.getLogger(__name__)


# ============================================================
# 1. STORAGE
# ============================================================

CHATS_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    turn_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id              INTEGER NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    position             INTEGER NOT NULL,
    question             TEXT NOT NULL,
    standalone_question  TEXT,
    uses_history         INTEGER NOT NULL DEFAULT 0,
    status               TEXT NOT NULL,
    sql                  TEXT,
    plan                 TEXT,
    message              TEXT,
    row_count            INTEGER NOT NULL DEFAULT 0,
    columns_json         TEXT,
    preview_json         TEXT,
    attempts             INTEGER NOT NULL DEFAULT 0,
    seconds              REAL,
    created_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_by_chat ON turns(chat_id, position);
"""


def open_chats_db(db_path):
    """
    Connection to a dataset's chat store (created on first use).

    Use it as `with closing(open_chats_db(path)) as connection, connection:`
    The connection's own `with` only commits; closing() is what releases
    the file, which Windows needs before chats.db can be moved or deleted.
    """

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(CHATS_SCHEMA)

    return connection


def now():
    return datetime.now().isoformat(timespec="seconds")


def create_chat(db_path, title):
    """Start a chat; returns its chat_id."""

    title = re.sub(r"\s+", " ", title).strip()[:CHAT_TITLE_LENGTH] or "New chat"

    with closing(open_chats_db(db_path)) as connection, connection:
        cursor = connection.execute(
            "INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now(), now()))
        chat_id = cursor.lastrowid

    return chat_id


def add_turn(db_path, chat_id, turn):
    """Append one turn to a chat; returns its position (1, 2, ...)."""

    frame = turn.get("frame")
    preview = (frame.head(STORED_PREVIEW_ROWS).to_json(orient="values",
                                                       date_format="iso")
               if frame is not None else None)
    columns = json.dumps(list(frame.columns)) if frame is not None else None

    with closing(open_chats_db(db_path)) as connection, connection:
        position = connection.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM turns WHERE chat_id = ?",
            (chat_id,)).fetchone()[0]

        connection.execute(
            """INSERT INTO turns (chat_id, position, question, standalone_question,
                   uses_history, status, sql, plan, message, row_count,
                   columns_json, preview_json, attempts, seconds, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chat_id, position, turn["question"], turn.get("standalone_question"),
             int(turn.get("uses_history", False)), turn["status"], turn.get("sql"),
             turn.get("plan"), turn.get("message"), turn.get("row_count", 0),
             columns, preview, len(turn.get("attempts", [])), turn.get("seconds"),
             now()))
        connection.execute("UPDATE chats SET updated_at = ? WHERE chat_id = ?",
                           (now(), chat_id))

    return position


def list_chats(db_path):
    """All chats of a dataset, most recently used first."""

    with closing(open_chats_db(db_path)) as connection, connection:
        rows = connection.execute(
            """SELECT c.chat_id, c.title, c.created_at, c.updated_at,
                      COUNT(t.turn_id) AS turns
               FROM chats c LEFT JOIN turns t ON t.chat_id = c.chat_id
               GROUP BY c.chat_id ORDER BY c.updated_at DESC, c.chat_id DESC""")
        return [dict(row) for row in rows]


def get_chat(db_path, chat_id):
    """One chat's row, or None when it does not exist."""

    with closing(open_chats_db(db_path)) as connection, connection:
        row = connection.execute(
            "SELECT chat_id, title, created_at, updated_at FROM chats "
            "WHERE chat_id = ?", (chat_id,)).fetchone()

    return dict(row) if row else None


def load_turns(db_path, chat_id):
    """Every turn of a chat in order, with the preview rows decoded."""

    with closing(open_chats_db(db_path)) as connection, connection:
        rows = connection.execute(
            "SELECT * FROM turns WHERE chat_id = ? ORDER BY position",
            (chat_id,)).fetchall()

    turns = []

    for row in rows:
        turn = dict(row)
        turn["uses_history"] = bool(turn["uses_history"])
        columns = json.loads(turn.pop("columns_json") or "null")
        values = json.loads(turn.pop("preview_json") or "null")
        turn["preview"] = (pd.DataFrame(values, columns=columns)
                           if columns is not None else None)
        turns.append(turn)

    return turns


def rename_chat(db_path, chat_id, title):
    title = re.sub(r"\s+", " ", title).strip()[:CHAT_TITLE_LENGTH] or "New chat"

    with closing(open_chats_db(db_path)) as connection, connection:
        connection.execute("UPDATE chats SET title = ?, updated_at = ? WHERE chat_id = ?",
                           (title, now(), chat_id))


def delete_chat(db_path, chat_id):
    """Permanently delete a chat and its turns (the API asks first)."""

    with closing(open_chats_db(db_path)) as connection, connection:
        connection.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))


# ============================================================
# 2. HISTORY AND THE REWRITE STEP
# ============================================================

REWRITE_PROMPT = """You turn a follow-up question into a standalone question about the same data.

Use the earlier conversation only to fill in what the new question leaves out:
- a filter, grouping or measure it continues ("only the <X> ones", "now by <Y>", "what about <Z>?");
- things it points at ("it", "they", "those"): replace them with the exact values from the earlier results.
Add nothing that neither the new question nor the earlier turns contain.

If the new question makes sense on its own, return it unchanged and set uses_history to false.
Never answer the question and never write SQL."""


REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "uses_history": {"type": "boolean"},
        "standalone_question": {"type": "string"},
    },
    "required": ["uses_history", "standalone_question"],
}


def recent_history(db_path, chat_id, limit=MEMORY_TURNS):
    """Last answered turns, oldest first."""

    answered = [turn for turn in load_turns(db_path, chat_id)
                if turn["status"] in {"answered", "empty"} and turn["sql"]]

    return answered[-limit:]


def history_text(history):
    """Earlier turns as plain text for the prompts."""

    blocks = []

    for number, turn in enumerate(history, start=1):
        preview = turn["preview"]
        rows = (preview.head(HISTORY_PREVIEW_ROWS).to_csv(index=False).strip()
                if preview is not None and not preview.empty else "(no rows)")
        more = (f"\n... {turn['row_count']} rows in total"
                if turn["row_count"] > HISTORY_PREVIEW_ROWS else "")
        blocks.append(f"Turn {number}\nQuestion: "
                      f"{turn['standalone_question'] or turn['question']}\n"
                      f"SQL: {turn['sql']}\nResult:\n{rows}{more}")

    return "\n\n".join(blocks)


def rewrite_question(settings, history, question):
    """
    Returns:
        {"uses_history", "standalone_question", "seconds"}
    """

    if not history:
        return {"uses_history": False, "standalone_question": question,
                "seconds": 0}

    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"Earlier conversation:\n\n{history_text(history)}"
                                    f"\n\nNew question: {question}"},
    ]

    try:
        answer, stats = chat_json(settings, messages, REWRITE_SCHEMA)

    except (ValueError, *MODEL_ERRORS):
        # Rewrite failed: fall back to the question as typed.
        return {"uses_history": False, "standalone_question": question,
                "seconds": 0}

    standalone = re.sub(r"\s+", " ", answer["standalone_question"]).strip() or question

    # The model's own uses_history flag is unreliable on small models, so
    # the question counts as building on history when the rewrite changed it.
    changed = comparable(standalone) != comparable(question)

    return {"uses_history": changed, "standalone_question": standalone,
            "seconds": stats["seconds"]}


def comparable(text):
    """Lowercase words only, to compare two phrasings."""

    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def sql_context(history):
    """The last query, offered to the SQL step for reuse."""

    return ("Earlier query in this conversation (reuse its filters, joins and "
            f"grouping where the question builds on it):\n{history[-1]['sql']}")
