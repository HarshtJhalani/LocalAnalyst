
# ============================================================
# FILE: src/helpers/answer_summary.py
# PURPOSE: Plain-language summary with a number check, and caveats
#          (06_charts_summary.ipynb, steps 6-7)
#
# The model writes one to three sentences from the question, SQL and up
# to SUMMARY_MAX_ROWS result rows. Every number in the summary must come
# from the result or the question (formatting / rounding tolerated);
# one retry, then a fixed template. Caveats are added by code, not the
# model, from the cleaning flags of the columns the SQL uses.
#
#   summarize()       -> {"summary", "verified", "source", "attempts",
#                         "rejected_numbers"}
#   build_caveats()   sql + profile -> list of warnings
# ============================================================

import json
import math
import numbers
import re

import numpy

from config import MAX_RESULT_ROWS, SUMMARY_MAX_ROWS
from src.helpers.chart_builder import format_number, readable
from src.helpers.model_client import MODEL_ERRORS, chat_json, past_deadline


# ============================================================
# 1. CAVEATS FROM THE CLEANING FLAGS
# ============================================================


def caveat_text(flag):
    column, issue, detail = flag["column"], flag["issue"], flag["detail"]

    if issue == "missing_values":
        count = detail.split(" of ")[0]
        return (f"{readable(column)} has {count} missing value(s); they are "
                "left out of sums, averages and counts of that column.")

    if issue == "outliers":
        examples = re.search(r"\[(.*?)\]", detail)
        shown = examples.group(1) if examples else "one value"
        return (f"{readable(column)} contains an extreme value ({shown}) that "
                "strongly affects totals and averages.")

    if issue == "similar_categories":
        pairs = re.search(r"\[(.*)\]", detail)
        return (f"{readable(column)} may hold the same category under two "
                f"spellings: {pairs.group(1) if pairs else detail}.")

    if issue == "partly_numeric":
        return f"{readable(column)} is stored as text because some values are not numbers."

    if issue == "decimal_comma":
        return (f"{readable(column)} is stored as text because its numbers use a "
                "decimal comma, so it cannot be summed or averaged.")

    if issue == "ambiguous_dates":
        return f"{readable(column)} dates could be day-first or month-first, so they are kept as text."

    return None


def build_caveats(sql, profile, truncated=False):
    """Warnings for the columns this query uses."""

    caveats = []

    for table in profile["tables"].values():
        if not re.search(rf"\b{re.escape(table['table_name'])}\b", sql):
            continue

        for flag in table["flags"]:
            if flag["column"] and re.search(rf"\b{re.escape(flag['column'])}\b", sql):
                text = caveat_text(flag)

                if text and text not in caveats:
                    caveats.append(text)

    if truncated:
        caveats.append(f"Only the first {MAX_RESULT_ROWS:,} rows are shown.")

    return caveats


# ============================================================
# 2. NUMBER CHECK
# ============================================================


ALWAYS_ALLOWED_NUMBERS = set(range(0, 13))


NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")


def numbers_in_text(text):
    """'Delhi had 1,234.5 sales' -> [1234.5]"""

    return [float(match.replace(",", ""))
            for match in NUMBER_PATTERN.findall(str(text))]


def allowed_numbers(frame, question):
    """Every number the summary may legitimately mention."""

    allowed = set(ALWAYS_ALLOWED_NUMBERS) | {len(frame)}
    allowed.update(numbers_in_text(question))

    for column in frame.columns:
        for value in frame[column].dropna():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                allowed.add(float(value))
            else:
                allowed.update(numbers_in_text(value))

    return allowed


def number_is_allowed(number, allowed):
    for value in allowed:
        # A share may be written as a percentage (0.25 -> 25%); any other
        # value times 100 would be an invented number.
        candidates = (value, value * 100) if 0 < abs(value) <= 1 else (value,)

        for candidate in candidates:
            if math.isclose(number, candidate, rel_tol=0.005, abs_tol=0.01):
                return True
    return False


def unverified_numbers(summary, frame, question):
    allowed = allowed_numbers(frame, question)
    return [number for number in numbers_in_text(summary)
            if not number_is_allowed(abs(number), {abs(v) for v in allowed})]


# ============================================================
# 3. SUMMARY
# ============================================================


SUMMARY_PROMPT = """You explain database query results to a non-technical reader.

Write one to three short sentences that directly answer the question.
- Use only numbers that appear in the result table, copied exactly as written there (keep the commas where they are).
- Do not abbreviate (no "k" or "million"), add up, subtract or otherwise calculate new numbers.
- Do not add currency symbols or units ($, Rs, kg); the data does not say which apply.
- Do not guess causes or mention SQL, tables or columns by their technical names."""


# Added only when the result shows a '(blank)' cell: as a standing rule
# the model copied its wording into answers that had no blank at all
# ("No recorded units sold" for North | 3).
BLANK_LABEL_RULE = ("\n- A '(blank)' cell is a group whose value is missing; call that "
                    "group \"no recorded <label>\" and still give its numbers.")


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def display_value(value):
    """
    A cell as the summary should write it. Numbers come pre-formatted
    (585250 -> 585,250) so the model copies them instead of inserting
    separators itself, which small models get wrong ("5,852,50").
    """

    # A blank label must stay visible, or the model shifts the numbers of
    # the rows that follow onto the wrong labels.
    if value is None or (isinstance(value, numbers.Number) and math.isnan(value)):
        return "(blank)"

    if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, numbers.Number):
        return str(value)

    if 0 < abs(value) < 1:
        return f"{value:.4g}"            # keep small shares readable

    return format_number(value)


def result_as_text(frame):
    """Result rows as a ' | ' table (commas belong to the numbers)."""

    shown = frame.head(SUMMARY_MAX_ROWS)
    lines = [" | ".join(str(column) for column in shown.columns)]
    lines += [" | ".join(display_value(value) for value in row)
              for row in shown.itertuples(index=False)]
    text = "\n".join(lines) + "\n"

    if len(frame) > len(shown):
        text += f"... ({len(frame)} rows in total)\n"

    return text


CURRENCY_SYMBOLS = "$€£₹¥"


def drop_invented_currency(summary, frame):
    """'$1,250' -> '1,250' unless the result itself shows that symbol."""

    shown = result_as_text(frame)

    for symbol in CURRENCY_SYMBOLS:
        if symbol not in shown:
            summary = re.sub(rf"{re.escape(symbol)}\s?(?=-?\d)", "", summary)

    return summary


def summary_problem(summary, frame):
    """
    What is wrong with a summary the number check cannot catch, as a
    correction for the model; None when it reads as a description.
    """

    # '2.06 | cut | color ...': the result table copied back.
    if "|" in summary or summary.count("\n") >= 3:
        return "Do not copy the table. Describe the result in one to three plain sentences."

    if not frame.empty and re.search(r"\bno (data|rows?|results?)\b", summary, re.IGNORECASE):
        return (f"The result has {len(frame)} rows, so data did match. A '(blank)' "
                "label is a group without a value. Describe the rows.")

    # The answer lost its numbers although the result has some.
    shown = frame.head(SUMMARY_MAX_ROWS)
    result_has_numbers = any(numbers_in_text(display_value(value))
                             for row in shown.itertuples(index=False) for value in row)

    if result_has_numbers and not numbers_in_text(summary):
        return "Answer the question with the numbers from the result table."

    return None


def template_summary(frame, spec):
    """Fallback that cannot invent anything."""

    if frame.empty:
        return "No data matched the question."

    if spec["type"] == "stat":
        return f"{readable(spec['y'])}: {format_number(frame[spec['y']].iloc[0])}."

    if spec["type"] == "bar":
        ranked = frame.sort_values(spec["x"], ascending=False)
        top = ranked.iloc[0]
        text = (f"{top[spec['y']]} has the highest {readable(spec['x']).lower()} "
                f"({format_number(top[spec['x']])})")

        if len(ranked) > 1:
            low = ranked.iloc[-1]
            text += (f" and {low[spec['y']]} the lowest "
                     f"({format_number(low[spec['x']])})")

        return f"{text}, across {len(frame)} groups."

    return f"The result has {len(frame)} rows; see the table and chart."


def summarize(settings, question, sql, frame, spec):
    """
    Returns:
        {"summary", "verified", "source", "attempts", "rejected_numbers"}
        source: model | template
    """

    if frame.empty:
        return {"summary": template_summary(frame, spec), "verified": True,
                "source": "template", "attempts": 0, "rejected_numbers": []}

    result_text = result_as_text(frame)
    system = SUMMARY_PROMPT + (BLANK_LABEL_RULE if "(blank)" in result_text else "")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nSQL: {sql}\n\n"
                                    f"Result:\n{result_text}"},
    ]
    rejected = []

    for attempt in (1, 2):
        # Out of time for this question: use the template below.
        if past_deadline(settings):
            break

        try:
            answer, _ = chat_json(settings, messages, SUMMARY_SCHEMA)

        except (ValueError, *MODEL_ERRORS):
            break                        # fall back to the template below

        summary = drop_invented_currency(answer["summary"].strip(), frame)

        problem = summary_problem(summary, frame)

        if problem:
            messages += [{"role": "assistant", "content": json.dumps(answer)},
                         {"role": "user", "content": problem}]
            continue

        bad = unverified_numbers(summary, frame, question)

        if not bad:
            return {"summary": summary, "verified": True, "source": "model",
                    "attempts": attempt, "rejected_numbers": rejected}

        rejected.extend(bad)
        messages += [
            {"role": "assistant", "content": json.dumps(answer)},
            {"role": "user", "content":
                f"These numbers are not in the result: "
                f"{[format_number(number) for number in bad]}. Rewrite the summary, "
                "copying every number character for character from the result "
                "table above."},
        ]

    return {"summary": template_summary(frame, spec), "verified": True,
            "source": "template", "attempts": 2, "rejected_numbers": rejected}
