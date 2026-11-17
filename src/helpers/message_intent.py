
# ============================================================
# FILE: src/helpers/message_intent.py
# PURPOSE: Sort a chat message before any SQL is written
#
# Not every message is a question about the data. Without this step
# "what" or "that is not a pie chart" went through the follow-up
# rewrite, which filled in the missing parts from the last turn, and
# the same query was answered again.
#
#   write_request "delete the North rows"     -> refused (read-only)
#   vague         "what", "huh", "?", "ok"    -> ask to rephrase
#   feedback      "that's wrong"              -> ask what should change
#   chart_change  "make it a pie chart"       -> redraw the last result
#   data          everything else             -> the normal SQL path
#
# Rules only (no model call): a message that names a column of the
# dataset is always a data question.
# ============================================================

import re

from src.helpers.chart_builder import requested_chart


# Words that carry no question on their own.
FILLER_WORDS = {
    "what", "whats", "huh", "hmm", "hm", "ok", "okay", "k", "yes", "yeah", "yep",
    "no", "nope", "why", "how", "so", "and", "then", "hi", "hello", "hey", "thanks",
    "thank", "you", "thx", "cool", "nice", "great", "wow", "lol", "again", "sorry",
    "explain", "more", "continue", "go", "on", "please", "pls", "the", "a", "is",
    "it", "this", "that", "sure", "fine", "right", "really", "wait", "um", "uh",
}

# The message points back at the previous answer.
REFERS_BACK = re.compile(
    r"\b(it|that|this|those|these|above|previous|last|same|instead|again|"
    r"make|change|convert|turn|redraw|draw|show|switch|not)\b", re.IGNORECASE)

# The message says the last answer was wrong, without saying what to do.
FEEDBACK = re.compile(
    r"\b(wrong|incorrect|not (right|correct|what i (asked|wanted|meant))|"
    r"that'?s not|that is not|doesn'?t (look|seem) right|mistake|makes no sense|"
    r"not what|fix (it|this|that|the (chart|graph|plot|answer)))\b", re.IGNORECASE)

CHART_WORDS = {"chart", "graph", "plot", "pie", "donut", "doughnut", "bar", "bars",
               "column", "columns", "line", "lines", "area", "stacked", "scatter",
               "histogram", "vertical", "as", "a", "an", "of", "for", "the", "in",
               "to", "into", "instead", "please", "pls"}

# A request to change the data: a write verb that opens the message
# (after an optional "please" / "can you"). Analytical wording that
# starts the same way ("change in sales", "add up the totals") is not
# a write request.
WRITE_REQUEST = re.compile(
    r"^\s*(please\s+|can you\s+|could you\s+|would you\s+|i want to\s+|"
    r"i'?d like to\s+|let'?s\s+|go ahead and\s+)?"
    r"(delete|remove|drop|truncate|update|insert|modify|edit|change|set|replace|"
    r"rename|overwrite|erase|clear|wipe|add|fix|correct)\b"
    r"(?!\s+(in|over|from|between|of|up|by|per|across|for|vs|versus)\b)", re.IGNORECASE)

# Words about the picture, not the data: "change the chart colours" or
# "fix the chart" is about presentation and is never a write request.
PRESENTATION = re.compile(r"\b(chart|graph|plot|bars?|lines?|pie|donut|colou?rs?|"
                          r"title|axis|labels?|legend)\b", re.IGNORECASE)

WRITE_REPLY = ("I can only read this data, not change it: the dataset is opened "
               "read-only. To change values, edit the file and upload it again. "
               "You can ask to see the rows you meant, for example \"Show the "
               "North rows\".")

VAGUE_REPLY = ("I'm not sure what you'd like to know. Ask a question about the "
               "data, for example \"Total sales by region\" or \"How many orders "
               "per month?\".")

FEEDBACK_REPLY = ("Sorry about that. Tell me what should change: the numbers, "
                  "the grouping (\"by product instead\"), a filter (\"only 2024\"), "
                  "or the chart (\"as a pie chart\").")

NO_EARLIER_ANSWER = ("There is no earlier answer in this chat to redraw. Ask a "
                     "question first, for example \"Units sold by sales rep as a "
                     "pie chart\".")


def words_of(message):
    """Lower-case words, punctuation removed."""

    return re.findall(r"[a-z0-9']+", message.lower().replace("_", " "))


def column_tokens(columns):
    """Every column name and its parts: total_sales -> total_sales, total, sales."""

    tokens = set()

    for column in columns:
        name = str(column).lower()
        tokens.add(name)
        tokens.update(part for part in name.split("_") if len(part) > 2)

    return tokens


# Words the sorting itself relies on. A column named "last", "change" or
# "chart_type" must not turn "make it a pie chart instead" or "change
# it to bars" into a data question, so these never count as naming a
# column on their own.
INTENT_WORDS = FILLER_WORDS | CHART_WORDS | {
    "those", "these", "above", "previous", "last", "same", "instead", "make",
    "change", "convert", "turn", "redraw", "draw", "show", "switch", "not",
    "wrong", "incorrect", "correct", "mistake", "type", "chart", "graph", "plot",
}


def names_a_column(message, columns):
    """
    True when the message names a column (or a clear part of one).
    Words in INTENT_WORDS do not count, unless the full column name with
    its underscores is written ("chart_type").
    """

    text = message.lower()
    tokens = column_tokens(columns) - INTENT_WORDS
    words = set(words_of(message))

    return any(token in words or ("_" in token and token in text) for token in tokens)


def is_write_request(message):
    """True for a request to change the data ("delete the North rows")."""

    return bool(WRITE_REQUEST.search(message)) and not PRESENTATION.search(message)


def classify_message(message, columns):
    """
    Sort one chat message.

    Returns:
        {"intent": "data" | "write_request" | "vague" | "feedback" | "chart_change",
         "chart": requested chart type or None,
         "reply": text for vague / feedback, else None}
    """

    words = words_of(message)
    chart = requested_chart(message)
    result = {"intent": "data", "chart": chart, "reply": None}

    # "Delete the North rows": refused before any SQL, even when it names
    # a column ("update units_sold ...").
    if is_write_request(message):
        return {**result, "intent": "write_request", "reply": WRITE_REPLY}

    # Anything that names a column is a data question.
    if names_a_column(message, columns):
        return result

    # "pie chart for the above", "make it a donut", "that is not a pie chart".
    if chart and (REFERS_BACK.search(message)
                  or all(word in CHART_WORDS for word in words)):
        return {**result, "intent": "chart_change"}

    # "that's wrong", "not what I asked".
    if FEEDBACK.search(message):
        return {**result, "intent": "feedback", "reply": FEEDBACK_REPLY}

    # "what", "huh?", "ok", "?": only filler words. A short message with a
    # value in it ("Laptop only") is a follow-up filter, not vague.
    if not words or all(word in FILLER_WORDS for word in words):
        return {**result, "intent": "vague", "reply": VAGUE_REPLY}

    return result
