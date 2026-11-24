
# ============================================================
# FILE: src/helpers/pipeline.py
# PURPOSE: Orchestrate the stages for one dataset
#
#   A. ingest_dataset()   upload -> clean -> SQLite -> profile -> schema
#   B. answer_question()  question -> guardrails -> SQL -> check ->
#                         safe execution, with self-correction retries
#                         (structure questions such as "what columns are
#                         there?" are answered from the profile, no SQL)
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
import re
import sqlite3
import time

import pandas as pd
from pathlib import Path

from config import ASK_TIME_BUDGET_SECONDS, MAX_OUTPUT_TOKENS, RETRY_TOKEN_RESERVE
from src.helpers.answer_summary import build_caveats, summarize
from src.helpers.chart_builder import CHART_REQUEST, build_figure, choose_chart
from src.helpers.chat_memory import (
    add_turn,
    save_presentation,
    create_chat,
    get_chat,
    now,
    recent_history,
    rewrite_question,
    sql_context,
)
from src.helpers.cleaning_review import build_review
from src.helpers.data_cleaner import load_and_clean
from src.helpers.dataset_preview import overview_suggestions, overview_text
from src.helpers.data_profiler import allowed_tables, build_schema_text, profile_dataset
from src.helpers.input_guardrails import check_question, check_table_limits
from src.helpers.logging_config import write_trace
from src.helpers.message_intent import (
    NO_EARLIER_ANSWER,
    WRITE_REPLY,
    classify_message,
    is_write_request,
)
from src.helpers.schema_answer import is_schema_question, mentions_schema_sql, schema_answer
from src.helpers.model_client import MODEL_ERRORS, past_deadline
from src.helpers.sql_executor import execute_sql
from src.helpers.sql_generator import (
    build_messages,
    clean_sql,
    estimate_tokens,
    generate_sql_from_messages,
    top_per_group,
    top_per_group_hint,
)
from src.helpers.sql_guardrails import check_sql
from src.helpers.sqlite_store import write_dataset


logger = logging.getLogger(__name__)


# ============================================================
# 1. STATUS FILES
# ============================================================

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
            "created_at": previous.get("created_at", now()),
            "updated_at": now()}

    write_json(paths["status"], data)

    return data


# ============================================================
# 2. A. INGESTION
# ============================================================

def ingest_dataset(paths, upload_path, original_filename, choices=None):
    """
    Clean an uploaded file and make it queryable.

    choices: the user's review choices (cleaning_review.py). With them the
    file is cleaned again from the original upload and the database is
    rebuilt; without them only the safe automatic fixes run.

    upload -> clean (every sheet) -> size limits -> dataset.db (cleaned +
    raw tables) -> profile.json -> schema.txt -> status.json

    Returns:
        {"dataset_id", "file_name", "tables": [...], "schema_text"}
    """

    write_status(paths, "running", "cleaning", file_name=original_filename,
                 error=None)

    try:
        # Step 1: read and clean every table.
        tables = load_and_clean(upload_path, table_base_name=original_filename,
                                choices=choices)

        # Step 2: size limits per table.
        for table in tables:
            limits = check_table_limits(table["table_name"], table["cleaned"])

            if not limits["ok"]:
                raise ValueError(limits["reason"])

        # Step 3: SQLite, profile and schema text.
        write_status(paths, "running", "storing")
        # Rebuilt from scratch, so tables a choice removed do not linger.
        paths["database"].unlink(missing_ok=True)
        written = write_dataset(paths["database"], tables)

        # JSON round trip so the stored profile and the one used now match.
        profile = json.loads(json.dumps(profile_dataset(tables, written), default=str))
        schema_text = build_schema_text(profile)

        reports = [table["report"] for table in tables]
        write_json(paths["cleaning_report"], reports)
        write_json(paths["cleaning_choices"], choices or {})
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

    # "pending" until the user applies or skips the review (choices given
    # means it was just applied); "none" when there is nothing to decide.
    review = ("done" if choices is not None
              else "pending" if build_review(reports) else "none")
    write_status(paths, "completed", "ready", tables=table_summaries, review=review)

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

def error_feedback(error, profile=None):
    """
    The error for the model. For an unknown column the real column names
    are listed: small models otherwise keep reusing a name (for example
    one from the prompt's examples) on every retry.
    """

    feedback = (f"That query failed: {error}\n"
                "Write a corrected query. Use only the tables and columns in the "
                "schema, and copy category values exactly.")

    missing = re.search(r"no such column: ([\w.\"]+)", str(error))

    if profile and missing:
        tables = "; ".join(f"{name}: {', '.join(table['columns'])}"
                           for name, table in profile["tables"].items())
        feedback += (f"\n{missing.group(1)} does not exist, so remove it from the query. "
                     f"The only columns that exist are -> {tables}. "
                     "To count rows use COUNT(*), or SUM(CASE WHEN ... THEN 1 ELSE 0 END) "
                     "to count rows that meet a condition.")

    return feedback


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


GROUP_BY = re.compile(r"\bgroup\s+by\s+(.+?)(?:\bhaving\b|\border\b|\blimit\b|\)|$)",
                      re.IGNORECASE | re.DOTALL)


def tables_used(sql, profile):
    """
    The profile's tables the query reads. A column is judged only in
    those: "department" is unique in a small targets sheet but not in the
    employees sheet the query uses.
    """

    return {name: table for name, table in profile["tables"].items()
            if re.search(rf'(?<![\w"]){re.escape(name)}(?![\w"])|"{re.escape(name)}"',
                         sql, re.IGNORECASE)}


def count_per_unique_column(sql, profile):
    """
    "Most watched movie" -> COUNT(*) ... GROUP BY original_title: every
    title is on one row, so every count is 1 and the "top" one is random.
    Returns (unique column, number columns to rank by instead), or None.
    """

    if not re.search(r"\b(count|sum|avg|min|max)\s*\(", sql, re.IGNORECASE):
        return None

    for match in GROUP_BY.finditer(sql):
        grouped = {part.strip().split(".")[-1].strip('"` ').lower()
                   for part in match.group(1).split(",")}

        for table in tables_used(sql, profile).values():
            for name, info in table["columns"].items():
                if name.lower() in grouped and info.get("unique") == table["rows"] \
                        and table["rows"] > 1:
                    numbers = [other for other, column in table["columns"].items()
                               if column["kind"] in ("integer", "float")
                               and column.get("unique") != table["rows"]]
                    return name, numbers

    return None


def repeated_listing(sql, frame):
    """
    "Which countries are in Other?" -> SELECT country FROM ... WHERE ...:
    one row per movie, so names repeat and nothing is counted. True for a
    one- or two-column listing (no grouping, DISTINCT or aggregate) whose
    first column repeats.
    """

    if re.search(r"\b(group\s+by|distinct|count|sum|avg|min|max)\b", sql, re.IGNORECASE):
        return False

    if frame is None or frame.empty or len(frame.columns) > 2:
        return False

    first = frame.iloc[:, 0]

    is_text = not (pd.api.types.is_numeric_dtype(first) or pd.api.types.is_bool_dtype(first))

    return is_text and first.duplicated().any()


OTHER_SLICE = re.compile(r"\bothers?\b", re.IGNORECASE)
OTHER_PHRASE = re.compile(
    r"\b(?:(?:in|of|from|inside|under)\s+)?(?:the\s+)?[\"']?others?[\"']?"
    r"(?:\s+(?:category|categories|slice|slices|group|groups|part|parts|section|ones?))?",
    re.IGNORECASE)


def spell_out_other(question, history):
    """
    "countries in the other category" after a pie that grouped slices as
    "Other": the phrase is replaced by the grouped names themselves, so
    the SQL filters on them. Left in, the word "other" reads as "every
    one except" and a small model writes NOT IN.
    """

    if not history or not OTHER_SLICE.search(question):
        return question

    chart = history[-1].get("chart") or {}
    grouped = chart.get("other_labels")

    if not grouped:
        return question

    names = ", ".join("'" + name.replace("'", "''") + "'" for name in grouped)
    replaced = OTHER_PHRASE.sub(f"where {chart['x']} IN ({names})", question, count=1)

    return (f"{replaced}. List each {chart['x']} once with its {chart['y']}, "
            f"only for {chart['x']} IN ({names}).")


PERIOD_WORDS = re.compile(r"\b(year|years|yearly|annual|annually|month|months|"
                          r"monthly|decade|decades)\b", re.IGNORECASE)


GROUPING_CLAUSE = re.compile(
    r"\b(?:group|partition)\s+by\s+(.*?)(?=\border\s+by\b|\bhaving\b|\blimit\b|\)|$)",
    re.IGNORECASE | re.DOTALL)
STRFTIME_CALL = re.compile(r"\bstrftime\s*\([^()]*\)", re.IGNORECASE)


def raw_date_grouping(question, sql, profile):
    """
    "For every year ..." whose GROUP BY or PARTITION BY uses a full date
    column (one group per day), even when the SELECT shows strftime():
    the date column, else None.
    """

    if not PERIOD_WORDS.search(question):
        return None

    clauses = GROUPING_CLAUSE.findall(STRFTIME_CALL.sub(" period ", sql))

    for table in profile["tables"].values():
        for name, info in table["columns"].items():
            if info["kind"] in ("date", "datetime") and any(
                    re.search(rf"\b{re.escape(name)}\b", clause) for clause in clauses):
                return name

    return None


def keep_top_per_group(frame, shape):
    """
    (group, item, number) rows -> the top `count` items of each group by
    the number (lowest first for "least"). A frame already one row per
    group, or not in that shape, is returned as it is.
    """

    numeric = [column for column in frame.columns
               if pd.api.types.is_numeric_dtype(frame[column])]

    if len(frame.columns) < 3 or not numeric or frame.iloc[:, 0].is_unique:
        return frame

    group, measure = frame.columns[0], numeric[-1]
    if measure == group:
        return frame

    ranked = frame.sort_values([group, measure], ascending=[True, shape["order"] == "ASC"],
                               na_position="last", kind="stable")

    return ranked.groupby(group, dropna=False, sort=False).head(shape["count"]) \
                 .reset_index(drop=True)


SOLD_QUANTITY = re.compile(
    r"\bhow\s+many\b(?!\s+(?:orders|rows|records|transactions|sales\s+were\s+made|times))"
    r".*\b(sold|sell|sells|selling|bought|buy|buys|purchased|purchase|ordered|shipped|"
    r"ship|delivered)\b", re.IGNORECASE)


def counts_rows_for_a_quantity(question, sql, profile):
    """
    "How many monitors were sold" answered with COUNT(*): the number of
    orders, not of monitors. Only when the table has a number column to
    add up. Returns those number columns, else None.
    """

    if not SOLD_QUANTITY.search(question) or re.search(r"\bsum\s*\(", sql, re.IGNORECASE) \
            or not re.search(r"\bcount\s*\(\s*\*\s*\)", sql, re.IGNORECASE):
        return None

    numbers = [name for table in profile["tables"].values()
               for name, info in table["columns"].items()
               if info["kind"] in ("integer", "float") and info.get("unique") != table["rows"]]

    return numbers or None


def count_of_number_column(sql, profile):
    """
    COUNT(units_sold): counts the rows that have a value, never the total.
    Returns the number column, else None.
    """

    for name in re.findall(r"\bcount\s*\(\s*(?:distinct\s+)?[\"`]?(\w+)[\"`]?\s*\)",
                           sql, re.IGNORECASE):
        for table in tables_used(sql, profile).values():
            info = table["columns"].get(name)

            if info and info["kind"] in ("integer", "float") \
                    and info.get("unique") != table["rows"]:
                return name

    return None


JOIN_CONDITION = re.compile(r"\bon\s+([\w.\"]+)\s*=\s*([\w.\"]+)", re.IGNORECASE)
TEXT_EQUALS = re.compile(r"([\w.\"]+)\s*=\s*'((?:[^']|'')*)'")


def wrong_join_key(sql, profile):
    """
    A split list table joined on anything but its key ("ON imbd_title_id =
    genre.row_id" -> no matches). Returns the key, else None.
    """

    for table_name, table in profile["tables"].items():
        link = table.get("list_of")

        if not link or not re.search(rf"\b{re.escape(table_name)}\b", sql):
            continue

        for left, right in JOIN_CONDITION.findall(sql):
            names = {side.split(".")[-1].strip('"') for side in (left, right)}
            if link["key"] in names and names != {link["key"]}:
                return link["key"]

    return None


TEXT_IN_LIST = re.compile(r"([\w.\"]+)\s+in\s*\(((?:\s*'(?:[^']|'')*'\s*,?)+)\)",
                          re.IGNORECASE)


def compared_values(sql):
    """(column, value) for every column = 'value' and column IN ('a', 'b')."""

    pairs = [(column, value) for column, value in TEXT_EQUALS.findall(sql)]

    for column, values in TEXT_IN_LIST.findall(sql):
        pairs += [(column, value) for value in re.findall(r"'((?:[^']|'')*)'", values)]

    return [(column.split(".")[-1].strip('"'), value.replace("''", "'"))
            for column, value in pairs]


def missing_text_value(sql, profile, db_path):
    """
    column = 'value' (or IN (...)) where no row has that exact value:
    the column that does hold it ('Mouse' is a product, not a category),
    the spelling that exists ('laptop' -> 'Laptop'), or a LIKE match
    ('Spielberg' -> 'Steven Spielberg').
    Returns (column, value, hint) or None.
    """

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    def has(table, column, value, op="="):
        quoted = '"' + table.replace('"', '""') + '"'
        column_sql = '"' + column.replace('"', '""') + '"'
        rows = connection.execute(f"SELECT DISTINCT {column_sql} FROM {quoted} "
                                  f"WHERE {column_sql} {op} ? LIMIT 3", (value,)).fetchall()
        return [row[0] for row in rows]

    try:
        for name, value in compared_values(sql):
            for table_name, table in tables_used(sql, profile).items():
                info = table["columns"].get(name)

                if not info or info["kind"] != "text" or has(table_name, name, value):
                    continue

                # The value is in another column of the same table.
                for other, other_info in table["columns"].items():
                    if other != name and other_info["kind"] == "text" \
                            and has(table_name, other, value):
                        return name, value, (f"'{value}' is a {other} value, not a {name}: "
                                             f"filter on {other} instead.")

                close = has(table_name, name, f"%{value}%", "LIKE")

                if close:
                    return name, value, (f"Values that contain it: {close}. Use the exact "
                                         f"value, or {name} LIKE '%{value}%' to match all.")
    finally:
        connection.close()

    return None


def ungrouped_question(question, sql, profile):
    """
    GROUP BY a plain column the question never mentions, with no "by",
    "per", "each" or "every" ("Average score of Comedy movies" grouped by
    genre): the column, else None.
    """

    if re.search(r"\b(by|per|each|every|for all|breakdown|split)\b", question, re.IGNORECASE):
        return None

    # Only questions asking for one overall figure ("Average score of Comedy
    # movies"); "which item sold most" groups by product without naming it.
    if not re.search(r"\b(average|avg|mean|total|sum|how many|how much|count|number of)\b",
                     question, re.IGNORECASE) or re.search(
            r"\b(which|who|most|least|top|bottom|highest|lowest|best|worst|rank|compare|"
            r"list|show)\b", question, re.IGNORECASE):
        return None

    words = set()

    for word in re.findall(r"[a-z0-9]+", question.lower()):
        # cities -> city, products -> product, sales -> sale / sales
        words |= {word, word.rstrip("s"), word[:-3] + "y" if word.endswith("ies") else word}

    for match in GROUP_BY.finditer(sql):
        for part in match.group(1).split(","):
            name = part.strip().split(".")[-1].strip('"` ').lower()

            for table in tables_used(sql, profile).values():
                if name in table["columns"] and not (
                        {piece for piece in name.split("_") if len(piece) > 2} & words
                        or name in words):
                    return name

    return None


GROUP_COMPARISON = re.compile(
    r"\bwhich\b.*\b(highest|lowest|most|least|largest|smallest|biggest|best|worst)\b",
    re.IGNORECASE)
SINGLE_ROW_WORDS = re.compile(r"\b(single|one|largest order|biggest order|individual)\b",
                              re.IGNORECASE)


def single_row_comparison(question, sql):
    """
    "Which region has the lowest total sales" answered with MIN(total_sales)
    per region: that compares each region's smallest single order, not the
    regions. True when the groups should be compared by SUM.
    """

    return bool(GROUP_COMPARISON.search(question) and not SINGLE_ROW_WORDS.search(question)
                and re.search(r"\b(min|max)\s*\(", sql, re.IGNORECASE)
                and re.search(r"\bgroup\s+by\b", sql, re.IGNORECASE)
                and not re.search(r"\b(sum|avg|count)\s*\(", sql, re.IGNORECASE))


AGGREGATE_CALL = re.compile(r"\b(count|sum|avg|min|max|total|group_concat|median_of|"
                            r"stdev_of|percentile_of)\s*\(", re.IGNORECASE)


def date_compared_to_year(sql, profile):
    """
    release_year BETWEEN '2000' AND '2010' on a full 'YYYY-MM-DD' date:
    text comparison drops every 2010 date ('2010-05-01' > '2010').
    Returns the date column, else None.
    """

    for table in tables_used(sql, profile).values():
        for name, info in table["columns"].items():
            if info["kind"] not in ("date", "datetime"):
                continue

            column = rf'(?<!\()(?:\w+\.)?"?{re.escape(name)}"?'
            year = r"'\d{4}'"

            if re.search(rf"{column}\s*(?:=|<>|!=|<=|>=|<|>)\s*{year}"
                         rf"|{column}\s+between\s+{year}\s+and\s+{year}"
                         rf"|{column}\s+in\s*\(\s*{year}", sql, re.IGNORECASE):
                return name

    return None


def unaggregated_numbers(sql, profile):
    """
    SELECT product, total_sales / units_sold ... GROUP BY product: no
    aggregate at all, so each group shows one arbitrary row's numbers.
    Returns a number column shown that way, else None.
    """

    if not re.search(r"\bgroup\s+by\b", sql, re.IGNORECASE) or AGGREGATE_CALL.search(sql) \
            or re.search(r"\bover\s*\(", sql, re.IGNORECASE):
        return None

    select = re.search(r"\bselect\b(.*?)\bfrom\b", sql, re.IGNORECASE | re.DOTALL)
    grouped = " ".join(match.group(1) for match in GROUP_BY.finditer(sql)).lower()

    if not select:
        return None

    for table in tables_used(sql, profile).values():
        for name, info in table["columns"].items():
            if (info["kind"] in ("integer", "float")
                    and re.search(rf"\b{re.escape(name)}\b", select.group(1), re.IGNORECASE)
                    and not re.search(rf"\b{re.escape(name)}\b", grouped)):
                return name

    return None


DECADE_WORD = re.compile(r"\bdecades?\b", re.IGNORECASE)
HISTOGRAM_WORD = re.compile(r"\b(histogram|distribution)\b", re.IGNORECASE)


def decade_without_decades(question, sql):
    """ "per decade" answered per year: no division by 10 in the SQL."""

    return bool(DECADE_WORD.search(question) and re.search(r"\bstrftime\s*\(", sql, re.IGNORECASE)
                and not re.search(r"/\s*10\b", sql))


def bucketed_histogram(question, sql):
    """ "Score histogram" answered with home-made buckets (CASE / GROUP BY)."""

    return bool(HISTOGRAM_WORD.search(question)
                and re.search(r"\b(group\s+by|case\s+when)\b", sql, re.IGNORECASE))


def column_under_longer_name(question, columns):
    """
    "price" asked, the column is list_price: (word, column). Only a word
    that is part of a longer column name counts; a question naming a whole
    column ("profit for each region") is no sign the refusal was wrong.
    """

    words = set(re.findall(r"[a-z]+", question.lower()))

    for column in columns:
        parts = [part for part in column.lower().split("_") if len(part) > 3]

        if len(parts) < 2 and "_" not in column:
            continue

        for part in parts:
            if part in words and part != column.lower():
                return part, column

    return None


HAS_WORDS = re.compile(r"\b(have|has|had|with)\b", re.IGNORECASE)
LACKS_WORDS = re.compile(r"\b(no|not|missing|without|blank|empty|lack|lacking|unknown|null)\b",
                         re.IGNORECASE)


def has_read_as_missing(question, sql):
    """ "How many orders have notes?" answered with notes IS NULL (the opposite)."""

    return bool(HAS_WORDS.search(question) and not LACKS_WORDS.search(question)
                and re.search(r"\bis\s+null\b", sql, re.IGNORECASE)
                and not re.search(r"\bis\s+not\s+null\b", sql, re.IGNORECASE))


AFTER_YEAR = re.compile(r"\bafter\s+(\d{4})\b", re.IGNORECASE)


def after_year_includes_it(question, sql):
    """
    "after 2020" written as >= '2020-01-01' (or >= '2020'): 2020 is not
    after 2020. Returns the year, else None.
    """

    match = AFTER_YEAR.search(question)

    if not match:
        return None

    year = match.group(1)

    if re.search(rf">=\s*'{year}(-01(-01)?)?'", sql):
        return year

    return None


def child_matched_on_other_column(sql, profile):
    """
    ... WHERE id IN (SELECT id FROM genre_table ...) on anything but the
    join key: matches nothing, or everything. Returns the key, else None.
    """

    for table_name, table in profile["tables"].items():
        link = table.get("list_of")

        if not link:
            continue

        for left, inner in re.findall(
                rf"([\w.\"]+)\s+in\s*\(\s*select\s+(?:distinct\s+)?([\w.\"]+)\s+from\s+"
                rf"\"?{re.escape(table_name)}\b", sql, re.IGNORECASE):
            names = {left.split(".")[-1].strip('"'), inner.split(".")[-1].strip('"')}
            if names != {link["key"]}:
                return link["key"]

    return None


BREAKDOWN_WORDS = re.compile(r"\b(by|per|for each|in each|across|split by|for every)\b",
                             re.IGNORECASE)


def grouped_but_not_shown(sql):
    """
    GROUP BY department, performance_score but SELECT only department:
    the rows repeat a department with counts nobody can tell apart.
    Returns the hidden column, else None.
    """

    select = re.search(r"^\s*select\b(.*?)\bfrom\b", sql, re.IGNORECASE | re.DOTALL)
    match = GROUP_BY.search(sql)

    if not select or not match or re.search(r"\bover\s*\(", sql, re.IGNORECASE):
        return None

    shown = select.group(1).lower()

    for part in match.group(1).split(","):
        name = part.strip().strip('"`').split(".")[-1].lower()
        # WHERE city = 'Delhi' ... GROUP BY city: one value, nothing to tell apart.
        fixed = re.search(rf"\b{re.escape(name)}\b\s*=\s*'", sql, re.IGNORECASE)
        if (re.fullmatch(r"[a-z_][a-z0-9_]*", name) and not name.isdigit() and not fixed
                and not re.search(rf"\b{re.escape(name)}\b", shown)):
            return name

    return None


def breakdown_column_dropped(question, sql, profile):
    """
    "performance by department": a text category column named in the
    question, a "by / per / each" grouping, and SQL that leaves the text
    column out (counting rows per department). Returns the column, else None.
    """

    if not BREAKDOWN_WORDS.search(question):
        return None

    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    words |= {word.rstrip("s") for word in words}

    for table in tables_used(sql, profile).values():
        # A word names a column only when no other column shares it
        # ("performance" -> performance_score; "sales" is in total_sales too).
        shared = {}
        for other in table["columns"]:
            for part in set(other.lower().split("_")):
                shared[part] = shared.get(part, 0) + 1

        for name, info in table["columns"].items():
            parts = {part for part in name.lower().split("_")
                     if len(part) > 3 and shared.get(part) == 1} | {name.lower()}
            if info["kind"] != "text" or not info.get("is_category"):
                continue

            # Left out of the SQL, or listed into one cell (GROUP_CONCAT(grade)).
            listed = re.search(rf"\bgroup_concat\s*\(\s*(?:distinct\s+)?[\w.\"]*\b"
                               rf"{re.escape(name)}\b", sql, re.IGNORECASE)
            if listed or (parts & words
                          and not re.search(rf"\b{re.escape(name)}\b", sql, re.IGNORECASE)):
                return name

    return None


PERCENT_WORDS = re.compile(r"\b(percent|percentage|share|proportion|ratio|rate|%)",
                           re.IGNORECASE)


def whole_number_division(question, sql):
    """
    "share in percent" as COUNT(*) * 100 / COUNT(*): SQLite divides whole
    numbers into a whole number (65.625 -> 65).
    """

    if not PERCENT_WORDS.search(question) or re.search(r"\b\d+\.\d*|\bcast\s*\(|\breal\b",
                                                        sql, re.IGNORECASE):
        return False

    return bool(re.search(r"(count|sum)\s*\([^)]*\)\s*(\*\s*100\s*)?/", sql, re.IGNORECASE)
                or re.search(r"\*\s*100\s*/", sql))


def subqueries(sql):
    """The text inside each "( SELECT ... )", brackets matched properly."""

    bodies = []

    for match in re.finditer(r"\(\s*select\b", sql, re.IGNORECASE):
        depth, position = 0, match.start()

        for position in range(match.start(), len(sql)):
            depth += {"(": 1, ")": -1}.get(sql[position], 0)
            if depth == 0:
                break

        bodies.append(sql[match.start() + 1:position])

    return bodies


def aggregate_without_from(sql):
    """
    WITH t AS (SELECT COUNT(*) AS total) ...: no FROM, so SQLite counts
    one empty row and the total is 1. True when a (sub)query aggregates
    without a FROM of its own.
    """

    for body in subqueries(sql):
        # Its own text, without the subqueries nested inside it.
        own = body
        for inner in subqueries(body[1:]):
            own = own.replace(inner, "")

        if (re.search(r"\b(count|sum|avg|min|max)\s*\(", own, re.IGNORECASE)
                and not re.search(r"\bfrom\b", own, re.IGNORECASE)):
            return True

    return False


def dataset_columns(profile):
    """Every column name of the cleaned tables."""

    return [name for table in profile["tables"].values() for name in table["columns"]]


def normalize_sql(sql):
    """Compare queries ignoring spacing, case and a trailing ';'."""

    return re.sub(r"\s+", " ", str(sql or "")).strip().rstrip(";").strip().lower()


def answer_from_schema(outcome, dataset, attempt=None):
    """
    Answer a structure question ("what columns are there?") from the
    profile. The guardrails block pragma_* and sqlite_master, so SQL
    could only fail here.
    """

    answer = schema_answer(dataset["profile"], dataset["allowed_tables"])

    if attempt is not None:
        attempt["outcome"] = "schema_redirect"

    outcome.update(status="schema", sql=None, frame=answer["frame"],
                   row_count=len(answer["frame"]), truncated=False,
                   schema_summary=answer["summary"],
                   suggestions=answer["suggestions"], message=None)

    return outcome


def answer_question(settings, paths, question, first_answer=None, context=None):
    """
    Question -> checked, executed, self-corrected SQL result.

    first_answer replaces the first model answer (tests inject a
    broken query to check recovery); context is the conversation
    context from chat_memory.sql_context().

    Returns:
        {"status", "question", "sql", "plan", "frame", "row_count",
         "truncated", "attempts", "message", "seconds"}
        status: answered | empty | schema | refused | blocked | failed
        A schema answer also has "schema_summary" and "suggestions".
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

    # Step 1a: a request to change the data is refused before any SQL
    # (ask_in_chat already catches it; this covers direct callers such as
    # the evaluation).
    if is_write_request(question):
        outcome.update(status="refused", message=WRITE_REPLY)
        return finish(outcome, started, settings, paths)

    # Step 1b: questions about the structure are answered without SQL.
    if is_schema_question(question):
        return finish(answer_from_schema(outcome, dataset), started, settings, paths)

    # Step 1c: the prompt must fit the model's context.
    messages, schema_detail = fit_messages(settings, dataset, question, context)

    if messages is None:
        outcome["message"] = ("This dataset has too many columns for the model "
                              "to read at once. Upload a file with fewer columns.")
        return finish(outcome, started, settings, paths)

    if schema_detail != "full":
        logger.warning("Schema text shortened to %r to fit num_ctx=%d (%s)",
                       schema_detail, settings["num_ctx"], paths["dataset_id"])

    tried_sql = set()            # every query already run, normalised
    empty_retry_used = False
    group_shape = top_per_group(question)
    nudged = set()               # each kind of correction at most once
    # "Plot most movies by country": a chart was asked for, so one row
    # (LIMIT 1 for "most") leaves nothing to draw.
    wants_chart = bool(CHART_REQUEST.search(question))

    for attempt_number in range(1, settings["max_retries"] + 2):
        # The question's time budget is used up: start no new model call.
        if past_deadline(settings):
            outcome["message"] = ("This question took too long to answer. Try a "
                                  "simpler or more specific question.")
            outcome["timed_out"] = True
            return finish(outcome, started, settings, paths)

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

        # Step 3: a refusal is final, unless the question names a column:
        # "price of a laptop" refused although list_price exists (one retry).
        hidden = (column_under_longer_name(question, dataset_columns(dataset["profile"]))
                  if not answer["can_answer"] else None)

        if hidden and "refusal" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("refusal")
            attempt.update(outcome="shape_check", error="refused, but a column fits")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": False, "sql": ""})},
                {"role": "user", "content": f"The schema has {hidden[1]}, which holds "
                 f"the {hidden[0]} the question asks about. Answer with it; refuse only "
                 "if it really is something else."},
            ]
            continue

        if not answer["can_answer"]:
            attempt["outcome"] = "refused"
            outcome.update(status="refused", message=answer["plan"])
            return finish(outcome, started, settings, paths)

        # Step 3b: the model reached for SQLite's catalog (pragma_table_info,
        # sqlite_master), which is always blocked: answer from the profile.
        if mentions_schema_sql(sql):
            return finish(answer_from_schema(outcome, dataset, attempt),
                          started, settings, paths)

        # Step 3c: "the most X for every Y" answered without ranking inside
        # the groups returns every pair or one overall row: one correction.
        if (group_shape and "group" not in nudged and attempt_number <= settings["max_retries"]
                and not re.search(r"\bover\s*\(", sql, re.IGNORECASE)
                and (re.search(r"\blimit\b", sql, re.IGNORECASE)
                     or not re.search(r"\bgroup\s+by\b", sql, re.IGNORECASE))):
            nudged.add("group")
            attempt.update(outcome="shape_check", error="not one row per group and item")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": "This does not cover every group. "
                 + top_per_group_hint(group_shape)},
            ]
            continue

        if (wants_chart and "chart" not in nudged and attempt_number <= settings["max_retries"]
                and re.search(r"\blimit\s+1\s*;?\s*$", sql, re.IGNORECASE)):
            nudged.add("chart")
            attempt.update(outcome="shape_check", error="chart asked, one row kept")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": "A chart was asked for, so return every "
                 "group (not LIMIT 1), sorted from highest to lowest, so the chart can "
                 "compare them."},
            ]
            continue

        # Step 3d: COUNT per a column that is different on every row is 1
        # for every group: rank by a number column instead (one correction).
        counted = count_per_unique_column(sql, dataset["profile"])

        if counted and "count" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("count")
            column, numbers = counted
            attempt.update(outcome="shape_check", error=f"COUNT per unique {column}")
            hint = (f" To find the top {column}, ORDER BY the number column that "
                    f"measures it ({', '.join(numbers)}) DESC, or use MAX()."
                    if numbers else "")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"Every {column} is on exactly one row, so "
                 f"grouping by {column} gives one row per {column} and COUNT, SUM or AVG "
                 "per group says nothing. For one overall figure remove the GROUP BY."
                 + hint},
            ]
            continue

        # Step 3f: COUNT(units_sold) is the number of orders, not the units.
        counted_number = count_of_number_column(sql, dataset["profile"])

        if (counted_number and "sum" not in nudged
                and attempt_number <= settings["max_retries"]):
            nudged.add("sum")
            attempt.update(outcome="shape_check", error=f"COUNT({counted_number})")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"COUNT({counted_number}) counts rows that "
                 f"have a value; it never adds them up. For a total use "
                 f"SUM({counted_number}); for a number of rows use COUNT(*)."},
            ]
            continue

        # Step 3k: "which group has the lowest total" compared single rows.
        if (single_row_comparison(question, sql) and "groups" not in nudged
                and attempt_number <= settings["max_retries"]):
            nudged.add("groups")
            attempt.update(outcome="shape_check", error="MIN/MAX per group for a comparison")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": "MIN or MAX per group compares single rows. "
                 "To compare the groups, use SUM(<measure>) per group (AVG if the "
                 "question says average) and order by it."},
            ]
            continue

        # Step 3n: "per decade" grouped per year.
        if (decade_without_decades(question, sql) and "decade" not in nudged
                and attempt_number <= settings["max_retries"]):
            nudged.add("decade")
            attempt.update(outcome="shape_check", error="per year, not per decade")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": "This groups by year. A decade is "
                 "(CAST(strftime('%Y', <date column>) AS INTEGER) / 10) * 10: use that "
                 "expression in SELECT and GROUP BY."},
            ]
            continue

        # Step 3o: a histogram with home-made buckets.
        if (bucketed_histogram(question, sql) and "histogram" not in nudged
                and attempt_number <= settings["max_retries"]):
            nudged.add("histogram")
            attempt.update(outcome="shape_check", error="histogram bucketed in SQL")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": "For a histogram return the raw values, one "
                 "row per record: SELECT <number column> FROM <table> (with any filter "
                 "the question asks). The chart makes even bins; no CASE or GROUP BY."},
            ]
            continue

        # Step 3r: "have notes" read as "notes IS NULL".
        if (has_read_as_missing(question, sql) and "has" not in nudged
                and attempt_number <= settings["max_retries"]):
            nudged.add("has")
            attempt.update(outcome="shape_check", error="'has' read as IS NULL")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": "The question asks for rows that HAVE the "
                 "value, so use <column> IS NOT NULL (for a number column, > 0), not "
                 "IS NULL."},
            ]
            continue

        # Step 3p: "after 2020" that includes 2020.
        after_year = after_year_includes_it(question, sql)

        if after_year and "after" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("after")
            attempt.update(outcome="shape_check", error=f"after {after_year} includes it")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"After {after_year} starts in "
                 f"{int(after_year) + 1}: use >= '{int(after_year) + 1}-01-01' (or "
                 f"strftime('%Y', ...) > '{after_year}')."},
            ]
            continue

        # Step 3q: a split list table matched with IN on the wrong column.
        in_key = child_matched_on_other_column(sql, dataset["profile"])

        if in_key and "in_key" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("in_key")
            attempt.update(outcome="shape_check", error=f"IN not on {in_key}")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"Match the tables on {in_key}: ... WHERE "
                 f"{in_key} IN (SELECT {in_key} FROM <list table> WHERE ...)."},
            ]
            continue

        # Step 3l: a full date compared to a bare year ('2010').
        year_column = date_compared_to_year(sql, dataset["profile"])

        if year_column and "year" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("year")
            attempt.update(outcome="shape_check", error=f"{year_column} compared to a year")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"{year_column} holds full dates "
                 f"('YYYY-MM-DD'), so comparing it to a year like '2010' as text is wrong. "
                 f"Compare strftime('%Y', {year_column}) to the year instead "
                 f"(BETWEEN '2000' AND '2010' keeps both years)."},
            ]
            continue

        # Step 3m: numbers shown in a grouped query without adding them up.
        loose_number = unaggregated_numbers(sql, dataset["profile"])

        if loose_number and "aggregate" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("aggregate")
            attempt.update(outcome="shape_check", error=f"{loose_number} not aggregated")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"With GROUP BY, {loose_number} must be "
                 "aggregated, or each group shows one arbitrary row's value. Use SUM, AVG, "
                 "MIN or MAX (for a ratio per group: SUM(a) * 1.0 / SUM(b))."},
            ]
            continue

        # Step 3s: "performance by department" without performance in the SQL.
        dropped = breakdown_column_dropped(question, sql, dataset["profile"])

        if dropped and "breakdown" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("breakdown")
            attempt.update(outcome="shape_check", error=f"{dropped} left out")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"The question asks about {dropped}, a text "
                 f"column, per group. Count each {dropped} value in each group: "
                 f"SELECT <group>, {dropped}, COUNT(*) AS n ... GROUP BY <group>, {dropped}."},
            ]
            continue

        # Step 3t: grouped by a column the result does not show.
        hidden = grouped_but_not_shown(sql)

        if hidden and "shown" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("shown")
            attempt.update(outcome="shape_check", error=f"GROUP BY {hidden} not selected")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"The query groups by {hidden} but does not "
                 f"show it, so the rows cannot be told apart. SELECT every GROUP BY "
                 f"column, {hidden} included."},
            ]
            continue

        # Step 3v: an aggregate with no FROM counts one empty row.
        if (aggregate_without_from(sql) and "from" not in nudged
                and attempt_number <= settings["max_retries"]):
            nudged.add("from")
            attempt.update(outcome="shape_check", error="aggregate without FROM")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": "A SELECT with COUNT / SUM / AVG but no FROM "
                 "reads no table (COUNT(*) is 1). Give every such SELECT its FROM <table>."},
            ]
            continue

        # Step 3u: a percentage with whole-number division (65.6 -> 65).
        if (whole_number_division(question, sql) and "division" not in nudged
                and attempt_number <= settings["max_retries"]):
            nudged.add("division")
            attempt.update(outcome="shape_check", error="whole-number division")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": "SQLite divides whole numbers into a whole "
                 "number (65.6 becomes 65). Multiply by 100.0 (or 1.0) before dividing."},
            ]
            continue

        # Step 3h: a split list table joined on the wrong key matches nothing.
        join_key = wrong_join_key(sql, dataset["profile"])

        if join_key and "join" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("join")
            attempt.update(outcome="shape_check", error=f"join not on {join_key}")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"Join the tables on {join_key} = {join_key} "
                 "(both sides), exactly as the schema note says; other columns never match."},
            ]
            continue

        # Step 3i: column = 'value' that no row has ('laptop', 'Spielberg').
        missing = missing_text_value(sql, dataset["profile"], db_path)

        if missing and "value" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("value")
            column, value, hint = missing
            attempt.update(outcome="shape_check", error=f"no {column} = '{value}'")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"No row has {column} = '{value}'. {hint}"},
            ]
            continue

        # Step 3j: grouped by a column the question does not ask about.
        stray_group = ungrouped_question(question, sql, dataset["profile"])

        if stray_group and "grouping" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("grouping")
            attempt.update(outcome="shape_check", error=f"GROUP BY {stray_group} not asked")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"The question does not ask for a breakdown by "
                 f"{stray_group}. Remove that GROUP BY and give the figure the question "
                 "asks for (filters stay in WHERE)."},
            ]
            continue

        # Step 3g: "how many X were sold" counted orders instead of items.
        quantities = counts_rows_for_a_quantity(question, sql, dataset["profile"])

        if quantities and "sold" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("sold")
            attempt.update(outcome="shape_check", error="COUNT(*) for a quantity sold")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": "COUNT(*) gives the number of orders (rows), "
                 "but the question asks how many items were sold: use SUM of the "
                 f"quantity column (one of {', '.join(quantities)})."},
            ]
            continue

        # Step 3e: "per year" grouped by the full date: one group per day.
        date_column = raw_date_grouping(question, sql, dataset["profile"])

        if date_column and "date" not in nudged and attempt_number <= settings["max_retries"]:
            nudged.add("date")
            attempt.update(outcome="shape_check", error=f"grouped by full {date_column}")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"{date_column} is a full date, so this "
                 f"groups by day. Group by strftime('%Y', {date_column}) for years "
                 f"(strftime('%Y-%m', {date_column}) for months), in GROUP BY and in "
                 "PARTITION BY."},
            ]
            continue

        # A query tried before (not only the last one) would fail the same way.
        repeated = normalize_sql(sql) in tried_sql
        tried_sql.add(normalize_sql(sql))

        # Step 4: static check, then execution.
        sql_check = check_sql(sql, allowed)

        if sql_check["ok"]:
            result = execute_sql(db_path, sql, allowed)
            error = result["error"]
        else:
            result, error = None, sql_check["reason"]

        # Step 5a': a listing whose names repeat: each once, with a count.
        if (error is None and result["row_count"] > 0 and "listing" not in nudged
                and attempt_number <= settings["max_retries"]
                and repeated_listing(sql, result["frame"])):
            nudged.add("listing")
            column = result["frame"].columns[0]
            attempt.update(outcome="shape_check", error=f"{column} repeats")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(
                    {"plan": answer["plan"], "can_answer": True, "sql": sql})},
                {"role": "user", "content": f"The result repeats the same {column} "
                 f"on many rows. List each {column} once with how many rows it has: "
                 f"SELECT {column}, COUNT(*) ... GROUP BY {column} ORDER BY COUNT(*) DESC."},
            ]
            continue

        # Step 5a: success with rows. "The most X per Y": the top item of
        # each group is kept here, in code (keep_top_per_group).
        if error is None and result["row_count"] > 0:
            frame = result["frame"]

            if group_shape:
                frame = keep_top_per_group(frame, group_shape)

            attempt["outcome"] = "ok"
            outcome.update(status="answered", frame=frame, row_count=len(frame),
                           truncated=result["truncated"])
            return finish(outcome, started, settings, paths)

        # Step 5b: empty result: one second look, then accept.
        if error is None:
            if empty_retry_used or repeated:
                attempt["outcome"] = "empty_accepted"
                outcome.update(status="empty", frame=result["frame"],
                               message="The query ran and no rows match.")
                return finish(outcome, started, settings, paths)

            attempt["outcome"] = "empty"
            empty_retry_used = True
            feedback = EMPTY_RESULT_FEEDBACK

        # Step 5c: error: stop if the model repeats any earlier query.
        else:
            attempt.update(outcome="sql_check_failed" if not sql_check["ok"]
                           else "sql_error", error=error)

            if repeated:
                break

            feedback = error_feedback(error, dataset["profile"])

        # Step 6: send the attempt and the feedback back to the model.
        messages = messages + [
            {"role": "assistant", "content": json.dumps(
                {"plan": answer["plan"], "can_answer": True, "sql": sql})},
            {"role": "user", "content": feedback},
        ]

    # The SQL error stays in the attempts ("How this was answered").
    outcome["message"] = ("I couldn't build a working query for that. Try rephrasing it, or "
                          "split it into simpler questions.")
    outcome["error"] = ("Could not produce a working query: "
                          + (outcome["attempts"][-1].get("error") or "unknown error"))

    return finish(outcome, started, settings, paths)


def finish(outcome, started, settings, paths):
    """Add timing, write the trace and return the outcome."""

    outcome["seconds"] = round(time.perf_counter() - started, 2)

    write_trace(paths["trace_log"], {
        "time": now(),
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

def clarify_outcome(question, reply):
    """A reply that asks the user to rephrase; no SQL was run."""

    return {"status": "clarify", "question": question, "message": reply,
            "attempts": [], "row_count": 0, "sql": None, "plan": None,
            "frame": None, "standalone_question": None, "uses_history": False}


def stored_row_noun(paths):
    """What one row is, as named for the preview ("movies"), else "rows"."""

    stored = paths["root"] / "row_noun.json"

    return read_json(stored)["noun"] if stored.is_file() else "rows"


def overview_outcome(paths, dataset, question, focus, intent):
    """
    "Tell me about the data", "trends?", "sales?", "what": a short overview
    and questions that were run and checked before being offered. No model
    call, no SQL guessed from an open-ended message.
    """

    suggestions = overview_suggestions(paths["database"], dataset["profile"], focus,
                                       stored_row_noun(paths))

    if intent == "overview":
        topic = f" about {' and '.join(focus)}" if focus else ""
        message = (f"{overview_text(dataset['profile'])}\n\nThat question is open-ended, "
                   f"so here are questions{topic} that work well with this data; click "
                   "one, or ask your own.")
    else:
        message = ("I'm not sure what you'd like to know. Ask a question about the "
                   "data, or click one of these.")

    return {**clarify_outcome(question, message), "suggestions": suggestions}


OPTION_WORDS = {"fill_median": "the median", "fill_mean": "the mean", "fill_zero": "0",
                "fill_most_common": "the most common value", "fill_unknown": "\"Unknown\""}


def match_column(target, columns):
    """The column a phrase names ("units sold" -> units_sold), else None."""

    text = " ".join(re.findall(r"[a-z0-9]+", target.lower()))

    if not text:
        return None

    for column in columns:                           # the whole name first
        if column.lower().replace("_", " ") == text or column.lower() == text:
            return column

    words = set(text.split())
    scored = [(len(words & set(column.lower().split("_"))), column) for column in columns]
    best = max(scored, default=(0, None))

    return best[1] if best[0] else None


def count_zeros(db_path, table, column):
    """Rows where a number column is exactly 0."""

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    try:
        quoted_table = '"' + table.replace('"', '""') + '"'
        quoted_column = '"' + column.replace('"', '""') + '"'
        return connection.execute(
            f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_column} = 0").fetchone()[0]
    finally:
        connection.close()


def cleaning_outcome(paths, dataset, question, request):
    """
    "Fill age with the median": checked against the review's options, then
    a reply with a confirm button; the confirmed form ("Apply: ...")
    re-cleans the original upload with that choice, like the review card.
    0, null, empty, blank and similar words in the request all mean
    "missing": a request that names 0 also fills the column's 0s.
    """

    columns = dataset_columns(dataset["profile"])
    column = match_column(request["target"], columns)

    if column is None:
        return clarify_outcome(question, "Which column should I change? Name it as in the "
                                         f"Data tab, e.g. \"fill {columns[0]} with the median\".")

    table, info = next((name, table_info["columns"][column])
                       for name, table_info in dataset["profile"]["tables"].items()
                       if column in table_info["columns"])
    rows = dataset["profile"]["tables"][table]["rows"]
    report = read_json(paths["cleaning_report"])
    choices = read_json(paths["cleaning_choices"]) if paths["cleaning_choices"].is_file() else {}
    item = next((item for item in build_review(report, choices)
                 if item["column"] == column and item["issue"] == request["issue"]), None)

    # How many values count as missing here: empty ones, plus 0s if the request names 0.
    empty = int(info.get("missing", 0))
    numeric = info["kind"] in ("integer", "float")
    zeros = (count_zeros(paths["database"], table, column)
             if request.get("zero_condition") and numeric else 0)

    if not empty and not zeros:
        what = "no empty values and no 0s" if request.get("zero_condition") else "no missing values"
        return clarify_outcome(question, f"{column} has {what} to fill or drop.")

    options = (item["options"] if item else
               [{"value": value, "label": label} for value, label in
                (("fill_median", "Fill with the median"), ("fill_mean", "Fill with the mean"),
                 ("fill_zero", "Fill with 0"), ("drop_rows", "Drop those rows"))])
    allowed = [option["value"] for option in options]

    if request["option"] not in allowed:
        labels = ", ".join(option["label"].lower() for option in options
                           if option["value"] != "leave")
        return clarify_outcome(question, f"That does not fit {column}. Options for it: {labels}.")

    parts = [f"{empty} empty"] * bool(empty) + [f"{zeros} with 0"] * bool(zeros)
    which = " and 0" if zeros else ""
    action = (f"drop the rows with empty{which} {column}" if request["option"] == "drop_rows"
              else f"fill the empty{which} values of {column} with "
                   f"{OPTION_WORDS[request['option']]}")
    detail = f" ({' and '.join(parts)} of {rows} rows)"

    if not request["confirmed"]:
        note = ("" if zeros or not request.get("zero_condition") or not numeric
                else f" {column} has no values of 0, so only its empty values change.")
        return {**clarify_outcome(
            question, f"This will {action}{detail}.{note} Answers already in your chats are "
                      "not re-run, and the original values stay in the raw table. Click to apply."),
            "status": "confirm", "suggestions": [f"Apply: {action}"]}

    # Applied: the same re-clean as the review card's "Apply".
    column_choices = choices.setdefault(table, {}).setdefault(column, {})
    column_choices[request["issue"]] = request["option"]
    if zeros:
        column_choices["zeros_as_missing"] = True

    status = read_status(paths)
    upload = paths["input_dir"] / status["file_name"]
    ingest_dataset(paths, upload, status["file_name"], choices)

    done = [action_["detail"].replace(" (your choice)", "")
            for table_report in read_json(paths["cleaning_report"])
            for action_ in table_report["actions"]
            if action_["column"] == column and "your choice" in action_["detail"]
            and action_["action"] in ("zeros_as_missing", "missing_filled", "missing_rows_dropped")]
    return {**clarify_outcome(
        question, f"Done: {' '.join(done) or action} Ask again to see answers on the updated "
                  "data; Data → Change cleaning decisions undoes it."),
        "status": "cleaned"}


def redraw_last_answer(settings, paths, dataset, last_turn, question, chart):
    """
    "Make it a pie chart": re-run the previous answer's stored SQL
    (checked and read-only, as for a download) so it can be drawn as the
    requested chart. No new SQL is written.
    """

    started = time.perf_counter()
    allowed = dataset["allowed_tables"]
    sql = last_turn["sql"]
    outcome = {"status": "failed", "question": question, "sql": sql,
               "plan": f"Redraw the previous answer as a {chart.replace('_', ' ')} chart.",
               "frame": None, "row_count": 0, "truncated": False, "attempts": [],
               "message": None,
               "standalone_question": last_turn["standalone_question"] or last_turn["question"],
               "uses_history": True}

    sql_check = check_sql(sql, allowed)
    result = (execute_sql(paths["database"], sql, allowed) if sql_check["ok"]
              else {"ok": False, "error": sql_check["reason"]})

    if not result["ok"]:
        outcome["message"] = f"The previous answer could not be redrawn: {result['error']}"
        return finish(outcome, started, settings, paths)

    outcome.update(status="answered" if result["row_count"] else "empty",
                   frame=result["frame"], row_count=result["row_count"],
                   truncated=result["truncated"])

    return finish(outcome, started, settings, paths)


def show_rows(settings, paths, dataset, question, preview):
    """
    "top 5 rows": the first (or last) rows of the main table, through the
    same check and read-only connection as any answer. No model call.
    """

    started = time.perf_counter()
    tables = dataset["profile"]["tables"]
    table = next(iter(tables))           # the uploaded table; split tables follow it
    total = tables[table]["rows"]
    count = preview["rows"]
    offset = max(total - count, 0) if preview["from_end"] else 0
    sql = f'SELECT * FROM "{table}" LIMIT {count}' + (f" OFFSET {offset}" if offset else "")
    where = "last" if preview["from_end"] else "first"
    sections = None

    # "first 5 and last 5 rows": both parts in one read-only query, shown
    # as two tables (the labels travel with the answer's chart spec).
    if preview.get("last_rows"):
        last = preview["last_rows"]
        start = max(total - last, 0)
        sql = (f'SELECT * FROM (SELECT * FROM "{table}" LIMIT {count}) UNION ALL '
               f'SELECT * FROM (SELECT * FROM "{table}" LIMIT {last} OFFSET {start})')
        where = f"first {count} and last"
        count = last
        sections = [{"title": f"First {preview['rows']} rows (1–{min(preview['rows'], total)})",
                     "rows": min(preview["rows"], total)},
                    {"title": f"Last {last} rows ({start + 1}–{total})",
                     "rows": min(last, total)}]
    outcome = {"status": "failed", "question": question, "sql": sql,
               "plan": f"Show the {where} {count} rows of {table}.",
               "frame": None, "row_count": 0, "truncated": False, "attempts": [],
               "message": None, "standalone_question": question, "uses_history": False}

    allowed = dataset["allowed_tables"]
    sql_check = check_sql(sql, allowed)
    result = (execute_sql(paths["database"], sql, allowed) if sql_check["ok"]
              else {"ok": False, "error": sql_check["reason"]})

    if not result["ok"]:
        outcome["message"] = f"The rows could not be read: {result['error']}"
        return finish(outcome, started, settings, paths)

    outcome.update(status="answered" if result["row_count"] else "empty",
                   frame=result["frame"], row_count=result["row_count"],
                   truncated=result["truncated"], row_sections=sections,
                   row_preview=(f"The {where} {count} of {total:,} rows of {table}, as "
                                "stored after cleaning." if sections else
                                f"The {where} {result['row_count']} of {total:,} rows "
                                f"of {table}, as stored after cleaning."))

    return finish(outcome, started, settings, paths)


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
        dataset = load_dataset(paths)
        columns = dataset_columns(dataset["profile"])
        history = (recent_history(paths["chats"], chat_id)
                   if use_memory and chat_id is not None else [])

        # Step 1b: not every message is a data question ("what", "that's
        # wrong", "make it a pie chart"); those never reach the rewrite.
        intent = classify_message(check["question"], columns)

        # "Describe the data" has a better answer: the column table.
        if intent["intent"] == "overview" and is_schema_question(check["question"]):
            intent = {**intent, "intent": "data"}

        if intent["intent"] == "chart_change" and not history:
            intent = {**intent, "intent": "clarify", "reply": NO_EARLIER_ANSWER}

        if intent["intent"] == "cleaning_request":
            outcome = cleaning_outcome(paths, dataset, check["question"], intent["cleaning"])

            if chat_id is None:
                chat_id = create_chat(paths["chats"], check["question"])

        elif intent["intent"] == "separate_requests":
            outcome = {**clarify_outcome(
                check["question"],
                f"That asks for {len(intent['parts'])} different results, and one answer "
                "can show only one. Click each to ask it on its own."),
                "suggestions": intent["parts"]}

            if chat_id is None:
                chat_id = create_chat(paths["chats"], check["question"])

        elif intent["intent"] in {"overview", "vague"}:
            outcome = overview_outcome(paths, dataset, check["question"],
                                       intent.get("focus") or [], intent["intent"])

            # An overview's questions are worth keeping: it starts a chat.
            # A bare "what" does not.
            if chat_id is None and intent["intent"] == "overview":
                chat_id = create_chat(paths["chats"], check["question"])

            if chat_id is None:
                outcome.update(seconds=round(time.perf_counter() - started, 2),
                               chat_id=None, position=None)
                return outcome

        elif intent["intent"] in {"feedback", "clarify", "write_request"}:
            outcome = clarify_outcome(check["question"], intent["reply"])

            # A request to change the data is refused, not clarified.
            if intent["intent"] == "write_request":
                outcome["status"] = "refused"

            if chat_id is None:        # nothing to store: no new chat for it
                outcome.update(seconds=round(time.perf_counter() - started, 2),
                               chat_id=None, position=None)
                return outcome

        elif intent["intent"] == "chart_change":
            outcome = redraw_last_answer(settings, paths, dataset, history[-1],
                                         check["question"], intent["chart"])

        elif intent["intent"] == "row_preview":
            if chat_id is None:
                chat_id = create_chat(paths["chats"], check["question"])

            outcome = show_rows(settings, paths, dataset, check["question"],
                                intent["preview"])

        else:
            if chat_id is None:
                chat_id = create_chat(paths["chats"], check["question"])

            # Step 2: rewrite with history ("Other" spelled out first).
            asked = spell_out_other(check["question"], history)
            rewrite = rewrite_question(settings, history, asked, columns)
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
         "summary", "caveats", "suggestions"}
    """

    frame = outcome.get("frame")
    no_figures = {"light": None, "dark": None}

    # Structure answer: the column table, its sentence and question ideas.
    if outcome["status"] == "schema":
        return {"chart": {"type": "table", "x": None, "y": None, "color": None,
                          "title": None, "note": None},
                "figures": no_figures,
                "summary": {"summary": outcome["schema_summary"], "verified": True,
                            "source": "schema", "attempts": 0, "rejected_numbers": []},
                "caveats": [], "suggestions": outcome["suggestions"]}

    # Nothing to show: the message explains why (blocked / refused / failed).
    if outcome["status"] not in {"answered", "empty"} or frame is None:
        return {"chart": {"type": "none"}, "figures": no_figures,
                "summary": {"summary": outcome.get("message"), "verified": True,
                            "source": "message", "attempts": 0,
                            "rejected_numbers": []},
                "caveats": [], "suggestions": outcome.get("suggestions") or []}

    # "top 5 rows": a table and a fixed sentence; no chart, no model call.
    if outcome.get("row_preview"):
        return {"chart": {"type": "table", "x": None, "y": None, "color": None,
                          "title": None, "note": None,
                          "sections": outcome.get("row_sections")},
                "figures": no_figures,
                "summary": {"summary": outcome["row_preview"], "verified": True,
                            "source": "template", "attempts": 0, "rejected_numbers": []},
                "caveats": [], "suggestions": []}

    question = outcome.get("standalone_question") or outcome["question"]
    spec = choose_chart(frame, question, outcome["question"])
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
        "suggestions": [],
    }


def run_question(settings, paths, chat_id, question, use_memory=True):
    """
    Everything the API needs for one question in a chat: guardrails,
    follow-up rewrite, SQL with retries, stored turn, chart, summary.
    """

    # One time budget for every model call this question makes.
    settings = {**settings, "deadline": time.monotonic() + ASK_TIME_BUDGET_SECONDS}

    outcome = ask_in_chat(settings, paths, chat_id, question, use_memory)
    presented = present_answer(settings, outcome, load_dataset(paths)["profile"])

    # Stored with the turn so the chat redraws with its summary and chart.
    if outcome.get("position") is not None:
        save_presentation(paths["chats"], outcome["chat_id"], outcome["position"],
                          presented)

    return {**outcome, **presented}
