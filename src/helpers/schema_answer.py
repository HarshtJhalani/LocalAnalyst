
# ============================================================
# FILE: src/helpers/schema_answer.py
# PURPOSE: Answer questions about the data's structure ("what columns
#          are there?") from the profile, without SQL or the model
#
# The SQL guardrail blocks pragma_* / sqlite_master on purpose, so a
# model asked for column names can only fail. These questions are
# recognised by rules and answered from profile.json (cleaned tables
# only, never *_raw), with suggested questions built from column types.
#
#   is_schema_question()   question -> bool (conservative)
#   mentions_schema_sql()  model SQL that reads the catalog -> bool
#   schema_answer()        profile -> {"frame", "summary", "suggestions"}
# ============================================================

import re

import pandas as pd

from src.helpers.chart_builder import format_number


# ============================================================
# 1. RECOGNISING STRUCTURE QUESTIONS
# ============================================================

STRUCTURE_WORDS = r"(columns?|fields?|variables|attributes|headers|tables?)"

# Each pattern asks about the structure itself. "Which column has the
# highest sales" asks about values and must not match, so "which" only
# matches when followed by an existence phrase.
SCHEMA_PATTERNS = [
    rf"\bwhat\s+(are\s+)?(the\s+|all\s+)?(available\s+)?{STRUCTURE_WORDS}\s*(are\s+there|exist|do\s+(i|we)\s+have|are\s+available|does\s+(it|the\s+\w+)\s+have|\?|$|in\b)",
    r"\b(column|field|table)\s+names\b",
    rf"\b(list|show|name|give|tell)(\s+me)?(\s+all)?(\s+of)?(\s+the)?\s+{STRUCTURE_WORDS}\b",
    rf"\bwhich\s+{STRUCTURE_WORDS}\s+(are\s+there|exist|do\s+(i|we)\s+have|are\s+available)\b",
    r"\bhow\s+many\s+(columns|fields|tables)\b",
    r"\bdescribe\s+(the\s+|this\s+|my\s+)?(data|dataset|table|file)\b",
    r"\bwhat\s+(data|information)\s+do\s+(i|we)\s+have\b",
    r"\bwhat('?s|\s+is)\s+in\s+(the\s+|this\s+|my\s+)?(data|dataset|table|file)\b",
    r"\bschema\b",
]


def is_schema_question(question):
    """True for questions about the data's structure, not its values."""

    text = re.sub(r"\s+", " ", str(question or "")).strip().lower()

    return any(re.search(pattern, text) for pattern in SCHEMA_PATTERNS)


def mentions_schema_sql(sql):
    """Model SQL that reads SQLite's catalog (always blocked)."""

    return bool(re.search(r"\bpragma_\w+|\bsqlite_(master|schema|temp_master)\b",
                          str(sql or ""), re.IGNORECASE))


# ============================================================
# 2. THE COLUMN TABLE
# ============================================================

def column_type(column):
    """'integer', 'integer (currency)', 'text (category)' ..."""

    kind = column["kind"]

    if column.get("unit"):
        return f"{kind} ({column['unit']})"

    if kind == "text" and column.get("is_category"):
        return "text (category)"

    return kind


def example_value(column):
    """A range for numbers and dates, sample values for text."""

    if column.get("min") is not None and column.get("max") is not None:
        low, high = column["min"], column["max"]

        if column["kind"] in {"integer", "float"}:
            low, high = format_number(low), format_number(high)

        return f"{low} to {high}"

    if column["kind"] == "boolean":
        return "true / false"

    examples = column.get("examples") or [value for value, _ in column.get("top_values") or []]

    return ", ".join(str(value) for value in examples[:3])


def column_rows(profile, allowed_tables):
    rows = []

    for table_name, table in profile["tables"].items():
        if table_name not in allowed_tables:
            continue                      # *_raw and anything not queryable

        for name, column in table["columns"].items():
            rows.append({
                "table": table_name,
                "column": name,
                "type": column_type(column),
                "example": example_value(column),
                "missing": f"{column['missing']} ({column['missing_pct']}%)",
            })

    return rows


# ============================================================
# 3. SUGGESTED QUESTIONS
# ============================================================

def spoken(name):
    """total_sales -> total sales"""

    return name.replace("_", " ")


def total_of(name):
    """units_sold -> 'Total units sold'; total_sales -> 'Total sales'"""

    words = spoken(name)

    return words.capitalize() if words.startswith("total") else f"Total {words}"


def suggestions_for(table, limit=5):
    """
    Questions this table can answer, from its column types:
    category + number -> bar, date + number -> line, two numbers -> scatter.
    """

    columns = table["columns"]
    numbers = [name for name, column in columns.items()
               if column["kind"] in {"integer", "float"}
               and not name.endswith("_id") and not column.get("is_unique")]
    # Totals and amounts first (what people usually add up), then money,
    # then a rate such as unit_price or discount_pct.
    numbers.sort(key=lambda name: (
        not re.search(r"total|amount|sales|revenue|value|count|quantity|units", name),
        columns[name].get("unit") != "currency",
        bool(re.search(r"price|rate|pct|percent|ratio", name))))
    # Categories with few groups make readable charts (region before customer).
    categories = sorted((name for name, column in columns.items()
                         if column["kind"] == "text" and column.get("is_category")
                         and column.get("unique", 0) >= 2),
                        key=lambda name: columns[name]["unique"])
    dates = [name for name, column in columns.items()
             if column["kind"] in {"date", "datetime"}]

    suggestions = []

    if numbers and categories:
        suggestions.append(f"{total_of(numbers[0])} by {spoken(categories[0])}")

    if numbers and dates:
        suggestions.append(f"{total_of(numbers[0])} per month")

    if len(numbers) >= 2:
        suggestions.append(f"{spoken(numbers[0]).capitalize()} vs {spoken(numbers[1])}")

    if categories:
        suggestions.append(f"How many rows per {spoken(categories[0])}?")

    if numbers and len(categories) >= 2:
        suggestions.append(f"Average {spoken(numbers[0])} by {spoken(categories[1])}")

    if numbers:
        suggestions.append(f"Top 5 rows by {spoken(numbers[0])}")

    return suggestions[:limit]


# ============================================================
# 4. THE ANSWER
# ============================================================

def schema_answer(profile, allowed_tables):
    """
    Returns:
        {"frame": one row per column, "summary": str, "suggestions": [str]}
    """

    frame = pd.DataFrame(column_rows(profile, allowed_tables),
                         columns=["table", "column", "type", "example", "missing"])
    tables = [name for name in profile["tables"] if name in allowed_tables]

    sentences = []
    for name in tables:
        columns = list(profile["tables"][name]["columns"])
        sentences.append(f"{name} has {len(columns)} columns: {', '.join(columns)}.")

    suggestions = []
    for name in tables:
        for suggestion in suggestions_for(profile["tables"][name]):
            if suggestion not in suggestions:
                suggestions.append(suggestion)

    return {"frame": frame, "summary": " ".join(sentences),
            "suggestions": suggestions[:5]}
