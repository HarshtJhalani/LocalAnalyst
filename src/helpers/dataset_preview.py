# ============================================================
# FILE: src/helpers/dataset_preview.py
# PURPOSE: First look at a freshly uploaded dataset
#
# The first rows of each cleaned table and a short plain-English
# description built from the profile (no model call), so a user sees
# which columns exist before asking for numbers or charts.
# ============================================================

import json
import logging
import re
import sqlite3

from src.helpers.model_client import MODEL_ERRORS, chat_json

logger = logging.getLogger(__name__)

PREVIEW_ROWS = 5
MAX_LISTED = 6


def preview_rows(db_path, table_name, limit=PREVIEW_ROWS):
    """The first rows of one table, opened read-only."""

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    try:
        quoted = '"' + table_name.replace('"', '""') + '"'
        cursor = connection.execute(f"SELECT * FROM {quoted} LIMIT ?", (int(limit),))
        columns = [item[0] for item in cursor.description]
        rows = [list(row) for row in cursor.fetchall()]

    finally:
        connection.close()

    return {"columns": columns, "rows": rows}


def short_number(value):
    """Round floats for reading; leave integers alone."""

    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")

    return f"{value:,}" if isinstance(value, int) else str(value)


def listed(names):
    """'a, b and c', cut to MAX_LISTED names."""

    names = list(names)
    extra = len(names) - MAX_LISTED

    if extra > 0:
        names = names[:MAX_LISTED] + [f"{extra} more"]

    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def measure_columns(table):
    """
    Number columns worth adding up or averaging. Identifiers are left
    out: the join key of a split list column, consecutive whole numbers
    (row_id, serial numbers) and number codes (PIN, ZIP).
    """

    link_key = (table.get("list_of") or {}).get("key")

    return [name for name, info in table["columns"].items()
            if info["kind"] in ("integer", "float") and name != link_key
            and not is_row_number(info, table["rows"])
            and not looks_like_code(info, table["rows"])]


def is_row_number(info, rows):
    """1, 2, 3 ... (or any run of consecutive whole numbers): an ID, not an amount."""

    return (info["kind"] == "integer" and info.get("unique") == rows
            and info.get("min") is not None
            and info["max"] - info["min"] + 1 == rows)


def looks_like_code(info, rows):
    """
    PIN / ZIP / account codes stored as numbers: whole numbers that all
    have the same 5+ digits and repeat across rows (amounts rarely do).
    """

    if info["kind"] != "integer" or info.get("min") is None or info["min"] < 0:
        return False

    digits = len(str(int(info["min"])))

    return (digits >= 5 and digits == len(str(int(info["max"])))
            and info.get("unique", rows) < rows)


def describe_table(table):
    """A few sentences about one profiled table."""

    columns = table["columns"]
    by_kind = {}

    for name, info in columns.items():
        kind = "category" if info.get("is_category") else info["kind"]
        by_kind.setdefault(kind, []).append(name)

    lines = [f"{table['rows']:,} rows and {len(columns)} columns."]

    numbers = measure_columns(table)

    if numbers:
        lines.append(f"Numbers: {listed(numbers)}.")

    for name in by_kind.get("date", []) + by_kind.get("datetime", []):
        info = columns[name]
        lines.append(f"Dates in {name} run from {info['min']} to {info['max']}.")

    for name in by_kind.get("category", [])[:3]:
        top = [str(value) for value, _ in columns[name].get("top_values") or []][:4]
        if top:
            lines.append(f"{name} has {columns[name]['unique']} values, "
                         f"e.g. {listed(top)}.")

    other = by_kind.get("text", []) + by_kind.get("boolean", [])

    if other:
        lines.append(f"Other columns: {listed(other)}.")

    gaps = [f"{name} ({info['missing_pct']:g}%)" for name, info in columns.items()
            if info.get("missing")]

    if gaps:
        lines.append(f"Missing values in {listed(gaps)}.")

    if numbers:
        info = columns[numbers[0]]
        if "min" in info:
            lines.append(f"For example {numbers[0]} ranges from {short_number(info['min'])} "
                         f"to {short_number(info['max'])}.")

    return " ".join(lines)


def readable(name):
    """'content_rating' -> 'content rating'; 'genre_value' -> 'genre'."""

    return name.removesuffix("_value").replace("_", " ")


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def chart_categories(table):
    """
    Text columns that could group a chart: 2 to 25 groups and not mostly
    empty, fewest groups first ("region" before "customer"). Whether the
    chart is readable is checked on the real result (checked_suggestions).
    """

    columns = table["columns"]
    names = [name for name, info in columns.items()
             if info["kind"] == "text" and 2 <= info["unique"] <= 25
             and info.get("missing_pct", 0) <= 30
             and (info.get("is_category") or info["unique"] < table["rows"])]

    return sorted(names, key=lambda name: (columns[name]["unique"] < 3,
                                           columns[name]["unique"]))


# What each chart needs from its result to be worth showing.
CHART_ROWS = {"bar": (2, 25), "pie": (2, 8), "line": (3, 400), "scatter": (5, 5000)}


def candidate(question, sql, chart):
    return {"question": question, "sql": sql, "chart": chart}


def candidate_questions(table_name, table):
    """
    Every starter question this table could support, best first, each
    with the SQL that answers it. "{rows}" is replaced by what one row is
    ("movies", "orders") or "rows". Decided from types and counts only.
    """

    columns = table["columns"]
    t = quoted(table_name)
    link = table.get("list_of")

    # A split list column: counting single values is the point of it.
    if link:
        value = quoted(link["value"])
        return [candidate(f"How many {{rows}} per {readable(link['value'])}, as a bar chart",
                          f"SELECT {value}, COUNT(*) FROM {t} GROUP BY {value}", "bar")]

    numbers = measure_columns(table)
    categories = chart_categories(table)
    dates = [name for name, info in columns.items()
             if info["kind"] in ("date", "datetime") and info.get("min")]
    labels = [name for name, info in columns.items()
              if info["kind"] == "text" and info.get("unique") == table["rows"]]
    found = []

    # A small table: one bar per row, named by its label.
    if numbers and labels and table["rows"] <= 25:
        n, label = quoted(numbers[0]), quoted(labels[0])
        found.append(candidate(f"{readable(numbers[0]).capitalize()} for each "
                               f"{readable(labels[0])}, as a bar chart",
                               f"SELECT {label}, {n} FROM {t} WHERE {n} IS NOT NULL", "bar"))

    for number in numbers[:2]:
        for category in categories[:2]:
            # Always an average: adding up a price or a rating is meaningless,
            # and types alone cannot tell a price from a sales amount.
            n, c = quoted(number), quoted(category)
            found.append(candidate(
                f"Average {readable(number)} by {readable(category)}, as a bar chart",
                f"SELECT {c}, AVG({n}) FROM {t} WHERE {c} IS NOT NULL GROUP BY {c}",
                "bar"))

    for date in dates[:1]:
        span = columns[date].get("span_days") or 0
        period, pattern = (("month", "%Y-%m") if span and span < 730 else ("year", "%Y"))
        d = quoted(date)
        found.append(candidate(
            f"How many {{rows}} per {period} of {readable(date)}, as a line chart",
            f"SELECT strftime('{pattern}', {d}) AS p, COUNT(*) FROM {t} "
            f"WHERE {d} IS NOT NULL GROUP BY p", "line"))

    for category in categories:
        c = quoted(category)
        found.append(candidate(f"Share of {{rows}} by {readable(category)}, as a pie chart",
                               f"SELECT {c}, COUNT(*) FROM {t} WHERE {c} IS NOT NULL "
                               f"GROUP BY {c}", "pie"))

    if len(numbers) >= 2:
        a, b = quoted(numbers[0]), quoted(numbers[1])
        found.append(candidate(f"{readable(numbers[0]).capitalize()} against "
                               f"{readable(numbers[1])}, as a scatter plot",
                               f"SELECT {a}, {b} FROM {t} WHERE {a} IS NOT NULL "
                               f"AND {b} IS NOT NULL", "scatter"))

    return found


def result_is_chartable(rows, chart):
    """
    The result makes a useful chart: enough rows but not too many for its
    kind, and the values are not all empty or all the same.
    """

    low, high = CHART_ROWS[chart]

    if not low <= len(rows) <= high:
        return False

    values = [row[-1] for row in rows]
    labels = [row[0] for row in rows]

    if all(value is None for value in values) or len(set(values)) < 2 and chart != "bar":
        return False

    # A bar or pie with one label hiding most of the result ('Unknown').
    if chart in ("bar", "pie") and len(set(labels)) < len(labels):
        return False

    return True


def checked_suggestions(db_path, candidates, row_noun="rows", wanted=3):
    """
    Run each candidate's SQL (read-only) and keep the first `wanted`
    whose result is chartable, at most one per chart type where
    possible. A candidate that errors or gives a useless chart is
    dropped, so the worst case is fewer suggestions, never a bad one.
    """

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    kept = []

    try:
        for item in candidates[:12]:
            try:
                rows = connection.execute(item["sql"]).fetchmany(
                    CHART_ROWS[item["chart"]][1] + 1)
            except sqlite3.Error:
                continue

            if result_is_chartable(rows, item["chart"]):
                kept.append({**item, "question": item["question"].format(rows=row_noun)})

    finally:
        connection.close()

    # Variety first: one per chart type, then any others that passed.
    by_type, rest = [], []
    for item in kept:
        (rest if any(other["chart"] == item["chart"] for other in by_type)
         else by_type).append(item)

    return [item["question"] for item in (by_type + rest)[:wanted]]


def suggested_questions(db_path, table_name, table, row_noun="rows"):
    """Up to three checked starter questions for one table."""

    return checked_suggestions(db_path, candidate_questions(table_name, table), row_noun,
                               wanted=3)


def dataset_suggestions(tables):
    """
    Three starter questions for the whole dataset: a split list column's
    question first (it is what the plain table cannot answer), then the
    uploaded table's.
    """

    ordered = sorted(tables, key=lambda table: not table["is_list"])
    questions = []

    for table in ordered:
        for question in table["suggestions"]:
            if question not in questions:
                questions.append(question)

    return questions[:3]


# ------------------------------------------------------------
# What one row is ("movies", "orders"): one short model call per
# dataset, stored, so suggestions read "How many movies per genre". The
# model only names; it never decides what is suggested. Anything that
# is not one or two plain words falls back to "rows".
# ------------------------------------------------------------

ROW_NOUN_PROMPT = (
    "You name what one row of a table is. Answer with the plural noun only, "
    "one or two lowercase words (e.g. \"orders\", \"movies\", \"employees\"). "
    "If unsure, answer \"rows\".")
ROW_NOUN_SCHEMA = {"type": "object", "properties": {"noun": {"type": "string"}},
                   "required": ["noun"]}
ROW_NOUN_PATTERN = re.compile(r"^[a-z]{3,20}( [a-z]{3,20})?$")
ROW_NOUN_TIMEOUT = 30


def name_rows(settings, table_name, columns, sample_rows):
    """One model call -> 'movies'; 'rows' when it fails or answers oddly."""

    sample = [dict(zip(columns, row)) for row in sample_rows[:3]]
    message = (f"Table: {table_name}\nColumns: {', '.join(columns)}\n"
               f"Sample rows: {json.dumps(sample, default=str)[:1500]}")

    try:
        answer, _ = chat_json({**settings, "request_timeout": ROW_NOUN_TIMEOUT},
                              [{"role": "system", "content": ROW_NOUN_PROMPT},
                               {"role": "user", "content": message}], ROW_NOUN_SCHEMA)
    except (ValueError, *MODEL_ERRORS) as exc:
        logger.info("Row noun not named for %s: %s", table_name, exc)
        return "rows"

    noun = str(answer.get("noun", "")).strip().lower()

    return noun if ROW_NOUN_PATTERN.match(noun) else "rows"


def overview_suggestions(db_path, profile, focus=(), row_noun="rows", wanted=3):
    """
    Checked starter questions for the whole dataset, or only those about
    the focus words ("sales" -> questions naming total sales) when any fit.
    """

    candidates = []

    # The split list table first: it answers what the plain table cannot.
    tables = sorted(profile["tables"].items(), key=lambda item: not item[1].get("list_of"))

    for name, table in tables:
        candidates += candidate_questions(name, table)

    if focus:
        focused = [item for item in candidates
                   if any(word in item["question"].lower() for word in focus)]
        candidates = focused or candidates

    return checked_suggestions(db_path, candidates, row_noun, wanted)


def overview_text(profile):
    """One short paragraph: the uploaded table's description."""

    name, table = next(iter(profile["tables"].items()))

    return describe_table(table)


def dataset_preview(db_path, profile, limit=PREVIEW_ROWS, row_noun="rows"):
    """Preview, description and checked starter questions for every table."""

    return [{"table_name": name,
             "description": describe_table(table),
             "suggestions": suggested_questions(db_path, name, table, row_noun),
             "is_list": bool(table.get("list_of")),
             **preview_rows(db_path, name, limit)}
            for name, table in profile["tables"].items()]
