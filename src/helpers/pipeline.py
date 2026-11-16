
# ============================================================
# FILE: src/helpers/pipeline.py
# PURPOSE: Orchestrate the stages for one dataset
#
#   A. ingest_dataset()   upload -> clean -> SQLite -> profile -> schema
#   B. answer_question()  question -> guardrails -> SQL -> check ->
#                         safe execution, with self-correction retries
#   C. ask_in_chat()      guardrail on the typed text -> follow-up
#                         rewrite -> B -> stored turn
#   D. run_question()     C + chart, checked summary and caveats
#
# All data for a dataset lives in datasets/<dataset_id>/ (see
# config.get_dataset_paths). Every question is traced to
# logs/traces.jsonl.
# ============================================================

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from config import MAX_OUTPUT_TOKENS, RETRY_TOKEN_RESERVE
from src.helpers.answer_summary import build_caveats, summarize
from src.helpers.chart_builder import build_figure, choose_chart
from src.helpers.chat_memory import (
    add_turn,
    create_chat,
    get_chat,
    recent_history,
    rewrite_question,
    sql_context,
)
from src.helpers.data_cleaner import load_and_clean
from src.helpers.data_profiler import allowed_tables, build_schema_text, profile_dataset
from src.helpers.input_guardrails import check_question, check_table_limits
from src.helpers.logging_config import write_trace
from src.helpers.model_client import MODEL_ERRORS
from src.helpers.sql_executor import execute_sql
from src.helpers.sql_generator import (
    build_messages,
    clean_sql,
    estimate_tokens,
    generate_sql_from_messages,
)
from src.helpers.sql_guardrails import check_sql
from src.helpers.sqlite_store import write_dataset


logger = logging.getLogger(__name__)


# ============================================================
# 1. STATUS FILES
# ============================================================

def _now():
    return datetime.now().isoformat(timespec="seconds")


def read_json(path):
    """Read a JSON file; ValueError when it is corrupt."""

    try:
        with Path(path).open(encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt JSON file {path}: {exc}") from exc


def write_json(path, data):
    """Write JSON atomically (temp file, then replace)."""

    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=str)
        temp_path.replace(path)

    except OSError as exc:
        raise OSError(f"Could not write {path}: {exc}") from exc


def read_status(paths):
    """status.json of a dataset, or None when ingestion never started."""

    if not paths["status"].is_file():
        return None

    return read_json(paths["status"])


def write_status(paths, status, stage, **fields):
    """Record the ingestion stage; keeps created_at from the first write."""

    previous = read_status(paths) or {}
    data = {**previous, **fields, "dataset_id": paths["dataset_id"],
            "status": status, "stage": stage,
            "created_at": previous.get("created_at", _now()),
            "updated_at": _now()}

    write_json(paths["status"], data)

    return data


# ============================================================
# 2. A. INGESTION
# ============================================================

def ingest_dataset(paths, upload_path, original_filename):
    """
    Clean an uploaded file and make it queryable.

    upload -> clean (every sheet) -> size limits -> dataset.db (cleaned +
    raw tables) -> profile.json -> schema.txt -> status.json

    Returns:
        {"dataset_id", "file_name", "tables": [...], "schema_text"}
    """

    write_status(paths, "running", "cleaning", file_name=original_filename,
                 error=None)

    try:
        # Step 1: read and clean every table.
        tables = load_and_clean(upload_path, table_base_name=original_filename)

        # Step 2: size limits per table.
        for table in tables:
            limits = check_table_limits(table["table_name"], table["cleaned"])

            if not limits["ok"]:
                raise ValueError(limits["reason"])

        # Step 3: SQLite, profile and schema text.
        write_status(paths, "running", "storing")
        written = write_dataset(paths["database"], tables)

        # JSON round trip so the stored profile and the one used now match.
        profile = json.loads(json.dumps(profile_dataset(tables, written), default=str))
        schema_text = build_schema_text(profile)

        write_json(paths["cleaning_report"], [table["report"] for table in tables])
        write_json(paths["profile"], profile)
        paths["schema"].write_text(schema_text, encoding="utf-8")

    except Exception as exc:
        write_status(paths, "failed", "ingestion", error=f"{type(exc).__name__}: {exc}")
        logger.error("Ingestion failed for %s: %s", paths["dataset_id"], exc)
        raise

    table_summaries = [{
        "table_name": table["table_name"],
        "rows_before": table["report"]["rows_before"],
        "rows": table["report"]["rows_after"],
        "columns": table["report"]["column_types"],
        "actions": len(table["report"]["actions"]),
        "flags": len(table["report"]["flags"]),
    } for table in tables]

    write_status(paths, "completed", "ready", tables=table_summaries)

    logger.info("Dataset %s ready: %d table(s) from %s", paths["dataset_id"],
                len(tables), original_filename)

    return {"dataset_id": paths["dataset_id"], "file_name": original_filename,
            "tables": table_summaries, "schema_text": schema_text}


def load_dataset(paths):
    """
    Schema text, allowed tables and profile of an ingested dataset.

    Raises FileNotFoundError when the dataset was never ingested.
    """

    if not paths["schema"].is_file() or not paths["profile"].is_file():
        raise FileNotFoundError(f"Dataset {paths['dataset_id']} is not ready.")

    profile = read_json(paths["profile"])

    return {"schema_text": paths["schema"].read_text(encoding="utf-8"),
            "allowed_tables": allowed_tables(profile),
            "profile": profile}


# ============================================================
# 3. B. SELF-CORRECTING ANSWER LOOP
# ============================================================
# A failed check or SQLite error goes back to the model with the error
# text, up to max_retries times; an empty result gets one second look
# and is then accepted; a repeated failing query stops early; a refusal
# is final; a model timeout ends as a clean failure.

def error_feedback(error):
    return (f"That query failed: {error}\n"
            "Write a corrected query. Use only the tables and columns in the "
            "schema, and copy category values exactly.")


EMPTY_RESULT_FEEDBACK = (
    "That query ran but returned no rows. Check that every filter value is "
    "spelled exactly as in the schema (case-sensitive) and that dates use "
    "'YYYY-MM-DD'. If no rows is truly the correct answer, return the same "
    "query."
)

# At temperature 0 the same messages give the same broken answer, so a
# retry needs something new in the conversation.
BAD_JSON_FEEDBACK = (
    "Your last answer was not complete, valid JSON. Answer again with a "
    "short plan (one sentence) and one query, as JSON with the keys plan, "
    "can_answer and sql."
)

# Schema text versions tried in order until the prompt fits the context.
SCHEMA_DETAILS = ("full", "compact", "names")


def fit_messages(settings, dataset, question, context=None):
    """
    Build the SQL prompt with the most detailed schema text that fits
    the model's context, leaving room for the answer and the retries.

    Returns:
        (messages, detail), or (None, None) when even column names and
        types alone do not fit.
    """

    budget = settings["num_ctx"] - MAX_OUTPUT_TOKENS - RETRY_TOKEN_RESERVE

    for detail in SCHEMA_DETAILS:
        schema_text = (dataset["schema_text"] if detail == "full"
                       else build_schema_text(dataset["profile"], detail))
        messages = build_messages(schema_text, question, context)

        if estimate_tokens(messages) <= budget:
            return messages, detail

    return None, None


def answer_question(settings, paths, question, first_answer=None, context=None):
    """
    Question -> checked, executed, self-corrected SQL result.

    first_answer replaces the first model answer (tests inject a
    broken query to check recovery); context is the conversation
    context from chat_memory.sql_context().

    Returns:
        {"status", "question", "sql", "plan", "frame", "row_count",
         "truncated", "attempts", "message", "seconds"}
        status: answered | empty | refused | blocked | failed
    """

    started = time.perf_counter()
    outcome = {"status": "failed", "question": question, "sql": None,
               "plan": None, "frame": None, "row_count": 0,
               "truncated": False, "attempts": [], "message": None}

    # Step 1: question guardrails.
    check = check_question(question)

    if not check["ok"]:
        outcome.update(status="blocked", message=check["reason"])
        return finish(outcome, started, settings, paths)

    question = outcome["question"] = check["question"]
    dataset = load_dataset(paths)
    allowed = dataset["allowed_tables"]
    db_path = paths["database"]

    # Step 1b: the prompt must fit the model's context.
    messages, schema_detail = fit_messages(settings, dataset, question, context)

    if messages is None:
        outcome["message"] = ("This dataset has too many columns for the model "
                              "to read at once. Upload a file with fewer columns.")
        return finish(outcome, started, settings, paths)

    if schema_detail != "full":
        logger.warning("Schema text shortened to %r to fit num_ctx=%d (%s)",
                       schema_detail, settings["num_ctx"], paths["dataset_id"])

    previous_sql = None
    empty_retry_used = False

    for attempt_number in range(1, settings["max_retries"] + 2):
        attempt = {"attempt": attempt_number}
        outcome["attempts"].append(attempt)

        # Step 2: generate (or use the injected test answer).
        try:
            if attempt_number == 1 and first_answer is not None:
                answer = {**first_answer, "sql": clean_sql(first_answer["sql"]),
                          "seconds": 0, "prompt_tokens": None}
            else:
                answer = generate_sql_from_messages(settings, messages)

        except ValueError as exc:
            attempt.update(outcome="bad_json", error=str(exc))
            messages = messages + [{"role": "user", "content": BAD_JSON_FEEDBACK}]
            continue

        except MODEL_ERRORS as exc:
            # Timeout or Ollama unreachable: stop cleanly, never crash the caller.
            attempt.update(outcome="model_error", error=f"{type(exc).__name__}: {exc}")
            outcome["message"] = ("The model did not answer in time or could not be "
                                  "reached. Please try again.")
            return finish(outcome, started, settings, paths)

        sql = answer["sql"]
        attempt.update(plan=answer["plan"], sql=sql,
                       seconds=answer["seconds"],
                       prompt_tokens=answer["prompt_tokens"])
        outcome.update(plan=answer["plan"], sql=sql)

        # Step 3: a refusal is final.
        if not answer["can_answer"]:
            attempt["outcome"] = "refused"
            outcome.update(status="refused", message=answer["plan"])
            return finish(outcome, started, settings, paths)

        # Step 4: static check, then execution.
        sql_check = check_sql(sql, allowed)

        if sql_check["ok"]:
            result = execute_sql(db_path, sql, allowed)
            error = result["error"]
        else:
            result, error = None, sql_check["reason"]

        # Step 5a: success with rows.
        if error is None and result["row_count"] > 0:
            attempt["outcome"] = "ok"
            outcome.update(status="answered", frame=result["frame"],
                           row_count=result["row_count"],
                           truncated=result["truncated"])
            return finish(outcome, started, settings, paths)

        # Step 5b: empty result: one second look, then accept.
        if error is None:
            if empty_retry_used or sql == previous_sql:
                attempt["outcome"] = "empty_accepted"
                outcome.update(status="empty", frame=result["frame"],
                               message="The query ran and no rows match.")
                return finish(outcome, started, settings, paths)

            attempt["outcome"] = "empty"
            empty_retry_used = True
            feedback = EMPTY_RESULT_FEEDBACK

        # Step 5c: error: stop if the model repeats the same query.
        else:
            attempt.update(outcome="sql_check_failed" if not sql_check["ok"]
                           else "sql_error", error=error)

            if sql == previous_sql:
                break

            feedback = error_feedback(error)

        # Step 6: send the attempt and the feedback back to the model.
        messages = messages + [
            {"role": "assistant", "content": json.dumps(
                {"plan": answer["plan"], "can_answer": True, "sql": sql})},
            {"role": "user", "content": feedback},
        ]
        previous_sql = sql

    outcome["message"] = ("Could not produce a working query: "
                          + (outcome["attempts"][-1].get("error") or "unknown error"))

    return finish(outcome, started, settings, paths)


def finish(outcome, started, settings, paths):
    """Add timing, write the trace and return the outcome."""

    outcome["seconds"] = round(time.perf_counter() - started, 2)

    write_trace(paths["trace_log"], {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": settings["model"],
        "dataset": paths["dataset_id"],
        "question": outcome["question"],
        "status": outcome["status"],
        "row_count": outcome["row_count"],
        "seconds": outcome["seconds"],
        "message": outcome["message"],
        "attempts": outcome["attempts"],
    })

    logger.info("%-8s %d attempt(s) %5.1fs  %s", outcome["status"],
                len(outcome["attempts"]), outcome["seconds"],
                outcome["question"][:60])

    return outcome


# ============================================================
# 4. C. ASKING INSIDE A CHAT
# ============================================================

def ask_in_chat(settings, paths, chat_id, question, use_memory=True):
    """
    Answer a question in a chat (creates the chat when chat_id is None).

    A blocked question is stored as a turn of an existing chat, but it
    does not start a new chat: chat_id and position are then None.

    Returns:
        outcome dict from answer_question() plus chat_id, position,
        standalone_question, uses_history

    Raises LookupError when chat_id is given and no such chat exists
    (checked first, not after the model has already answered).
    """

    started = time.perf_counter()

    if chat_id is not None and get_chat(paths["chats"], chat_id) is None:
        raise LookupError(f"Chat {chat_id} does not exist.")

    # Step 1: check what the user actually typed, before any rewrite.
    check = check_question(question)

    if not check["ok"]:
        outcome = {"status": "blocked", "question": question,
                   "message": check["reason"], "attempts": [], "row_count": 0,
                   "sql": None, "frame": None, "standalone_question": None,
                   "uses_history": False}

        if chat_id is None:
            outcome.update(seconds=round(time.perf_counter() - started, 2),
                           chat_id=None, position=None)
            return outcome

    else:
        if chat_id is None:
            chat_id = create_chat(paths["chats"], check["question"])

        # Step 2: rewrite with history.
        history = recent_history(paths["chats"], chat_id) if use_memory else []
        rewrite = rewrite_question(settings, history, check["question"])
        context = sql_context(history) if rewrite["uses_history"] else None

        # Step 3: the retry loop on the standalone question.
        outcome = answer_question(settings, paths, rewrite["standalone_question"],
                                  context=context)
        outcome.update(question=check["question"],
                       standalone_question=rewrite["standalone_question"],
                       uses_history=rewrite["uses_history"])

    outcome["seconds"] = round(time.perf_counter() - started, 2)
    outcome["chat_id"] = chat_id
    outcome["position"] = add_turn(paths["chats"], chat_id, outcome)

    return outcome


# ============================================================
# 5. D. CHART, SUMMARY AND CAVEATS
# ============================================================

def present_answer(settings, outcome, profile):
    """
    Chart, summary and caveats for an answered question.

    Returns:
        {"chart", "figures": {"light", "dark"} (Plotly JSON or None),
         "summary", "caveats"}
    """

    frame = outcome.get("frame")

    # Nothing to show: the message explains why (blocked / refused / failed).
    if outcome["status"] not in {"answered", "empty"} or frame is None:
        return {"chart": {"type": "none"}, "figures": {"light": None, "dark": None},
                "summary": {"summary": outcome.get("message"), "verified": True,
                            "source": "message", "attempts": 0,
                            "rejected_numbers": []},
                "caveats": []}

    question = outcome.get("standalone_question") or outcome["question"]
    spec = choose_chart(frame, question)
    figures = {}

    # The answer is already stored; a chart that cannot be drawn must
    # not lose it, so the result falls back to the table.
    try:
        for mode in ("light", "dark"):
            figure = build_figure(frame, spec, mode)
            figures[mode] = figure.to_json() if figure is not None else None

    except Exception as exc:
        logger.warning("Chart %r failed, showing the table instead: %s: %s",
                       spec["type"], type(exc).__name__, exc)
        spec = {**spec, "type": "table", "x": None, "y": None, "color": None,
                "note": "The chart could not be drawn; the table shows the result."}
        figures = {"light": None, "dark": None}

    return {
        "chart": spec,
        "figures": figures,
        "summary": summarize(settings, question, outcome["sql"], frame, spec),
        "caveats": build_caveats(outcome["sql"], profile, outcome.get("truncated", False)),
    }


def run_question(settings, paths, chat_id, question, use_memory=True):
    """
    Everything the API needs for one question in a chat: guardrails,
    follow-up rewrite, SQL with retries, stored turn, chart, summary.
    """

    outcome = ask_in_chat(settings, paths, chat_id, question, use_memory)
    presented = present_answer(settings, outcome, load_dataset(paths)["profile"])

    return {**outcome, **presented}
