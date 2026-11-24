
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
    r"make|change|convert|turn|redraw|draw|show|switch|not|now)\b", re.IGNORECASE)

# The message says the last answer was wrong, without saying what to do.
FEEDBACK = re.compile(
    r"\b(wrong|incorrect|not (right|correct|what i (asked|wanted|meant))|"
    r"that'?s not|that is not|doesn'?t (look|seem) right|mistake|makes no sense|"
    r"not what|fix (it|this|that|the (chart|graph|plot|answer)))\b", re.IGNORECASE)

# "top 5 rows", "show me the first 10 records", "sample of the data",
# "head": the stored rows themselves, not a question. Every word must be
# one of these (or a number), so "top 5 rows of that" or "top 5
# products" stay data questions.
PREVIEW_WORDS = {
    "please", "pls", "can", "could", "you", "show", "give", "display", "list", "print",
    "see", "view", "me", "the", "a", "what", "are", "is", "top", "first", "last",
    "bottom", "head", "tail", "sample", "some", "few", "random", "of", "from", "in",
    "my", "this", "your", "rows", "row", "records", "record", "entries", "lines",
    "data", "dataset", "table", "file", "sheet",
}
PREVIEW_TARGET = {"rows", "row", "records", "record", "entries", "lines", "data",
                  "dataset", "table", "head", "tail", "sample"}
PREVIEW_DEFAULT_ROWS = 5
PREVIEW_MAX_ROWS = 50


def row_preview_request(message):
    """
    {"rows": n, "from_end": bool} when the message only asks to see
    stored rows ("top 5 rows", "last 10 records", "show the data"), else None.
    """

    tokens = re.findall(r"[a-z]+|\d+", message.lower())
    words = [token for token in tokens if not token.isdigit()]
    counts = [int(token) for token in tokens if token.isdigit()]
    start_words, end_words = {"top", "first", "head"}, {"last", "bottom", "tail"}
    both = bool(start_words & set(words)) and bool(end_words & set(words))

    if (not words or len(counts) > (2 if both else 1) or not PREVIEW_TARGET & set(words)
            or not all(word in PREVIEW_WORDS or word == "and" for word in words)):
        return None

    fit = lambda count: max(1, min(count, PREVIEW_MAX_ROWS))

    # "first 5 rows and last 5 rows", "first and last 5 rows", "top 3 and bottom 2".
    if both:
        first_at = min(tokens.index(word) for word in start_words if word in tokens)
        last_at = min(tokens.index(word) for word in end_words if word in tokens)
        counts = counts or [PREVIEW_DEFAULT_ROWS]
        first, last = (counts * 2)[:2] if first_at < last_at else (counts * 2)[1::-1]
        return {"rows": fit(first), "from_end": False, "last_rows": fit(last)}

    count = counts[0] if counts else PREVIEW_DEFAULT_ROWS

    return {"rows": fit(count), "from_end": bool(end_words & set(words))}


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


# Open-ended messages ("tell me about the data", "something interesting",
# "trends?", "compare", "best one?"): no single query answers them, and a
# guessed one is worse than an overview with questions that work.
OPEN_ENDED_TRIGGERS = {
    "tell", "about", "overview", "summary", "summarize", "summarise", "describe",
    "description", "explore", "insight", "insights", "interesting", "something",
    "anything", "trend", "trends", "compare", "comparison", "analysis", "analyze",
    "analyse", "stats", "statistics", "info", "information", "details",
    "everything", "best", "worst", "top", "good", "important", "key", "main",
    "patterns", "pattern", "overall", "general", "quick", "notable", "useful", "ask",
}
OPEN_ENDED_WORDS = OPEN_ENDED_TRIGGERS | FILLER_WORDS | {
    "me", "show", "give", "some", "any", "there", "can", "i", "do", "ask", "could",
    "would", "should", "look", "at", "which", "of", "in", "for", "to", "whats",
    "what's", "are", "here", "all", "one", "ones", "data", "dataset", "table", "file",
    "find", "see", "get", "an", "my", "your", "us", "let's", "lets", "tell's",
}
MAX_TOPIC_WORDS = 2

# A word asking for a specific number makes it a real question ("Total
# sales?", "average discount"), never an overview.
AGGREGATE_WORDS = {"total", "totals", "sum", "average", "avg", "mean", "median", "count",
                   "number", "max", "maximum", "min", "minimum", "how", "many", "much",
                   "highest", "lowest", "most", "least", "per", "by"}


def open_ended(words, columns):
    """
    None, or {"focus": [column tokens]} when the message is open-ended:
    every word is from OPEN_ENDED_WORDS (at least one a trigger), or the
    message is just a topic ("sales?", "region trends") naming up to two
    columns with nothing else to ask.
    """

    if not words or any(word.isdigit() or word in AGGREGATE_WORDS for word in words):
        return None

    tokens = column_tokens(columns) - INTENT_WORDS
    other = [word for word in words if word not in OPEN_ENDED_WORDS]

    if not other:
        if any(word in OPEN_ENDED_TRIGGERS for word in words):
            return {"focus": []}
        return None

    if len(other) <= MAX_TOPIC_WORDS and all(word in tokens for word in other):
        return {"focus": other}

    return None


# "score histogram and movies per decade": two separate requests, each
# with its own grouping or chart word. One query cannot answer both.
REQUEST_WORDS = re.compile(
    r"\b(by|per|each|every|histogram|chart|plot|graph|distribution|trend|over time|"
    r"breakdown|pie|bar|line|scatter)\b", re.IGNORECASE)
GROUPING_WORDS = re.compile(r"\b(by|per|each|every|histogram|distribution|trend|"
                            r"over time|breakdown)\b", re.IGNORECASE)
PRESENTATION_SUFFIX = re.compile(
    r"\b(as|in|into|with) (an? )?(\w+ )?(\w+ )?(chart|graph|plot|table)s?\b", re.IGNORECASE)
FRAGMENT_START = re.compile(r"^(by|per|for|in|over|across|each|every|with)\b", re.IGNORECASE)
REQUEST_SPLIT = re.compile(r"\s*(?:,\s*and\s+|\s+and\s+(?:also\s+)?|\s*&\s*|;\s*|\s+plus\s+|"
                           r"\s+as well as\s+)", re.IGNORECASE)


def separate_requests(message):
    """
    The parts of a message that asks for two or more separate results,
    each one complete on its own (it has its own "by", "per", "histogram",
    "chart" ...); else []. "Total sales by region and product" is one
    request: "product" alone asks for nothing.
    """

    parts = [part.strip(" .?!,") for part in REQUEST_SPLIT.split(message)]
    parts = [part for part in parts if part]

    # "... as a bar chart" says how to draw the whole message, not a request.
    asked = [PRESENTATION_SUFFIX.sub("", part).strip() for part in parts]

    if len(parts) < 2 or not all(REQUEST_WORDS.search(part) for part in asked):
        return []

    # "sales by region and by product": a part that starts with its grouping
    # word, or is one word, is a fragment of a single request.
    if any(len(part.split()) < 2 or FRAGMENT_START.match(part) for part in asked):
        return []

    # "bar chart of score and movies per decade": a part that only names a
    # chart shares the grouping of the part that has one: one chart, two
    # measures. Every part naming its own chart ("pie chart of countries &
    # bar chart of genres"), or its own grouping, is a separate request.
    grouped = [bool(GROUPING_WORDS.search(part)) for part in asked]

    if any(grouped) and not all(grouped):
        return []

    return [part[0].upper() + part[1:] for part in parts]


# "fill age with the median", "replace missing salary with the mean",
# "drop rows with missing email": a cleaning choice, applied through the
# review step (the cleaner re-cleans the original upload), never SQL.
# "Apply: ..." is the confirmed form the reply's button sends.
FILL_REQUEST = re.compile(
    r"^\s*(?P<apply>apply:\s*)?(?:please\s+|can you\s+|could you\s+)?"
    r"(?:fill|replace|impute|substitute|set)\b(?P<target>.*?)\bwith\s+(?:the\s+|an?\s+)?"
    r"(?P<how>median|mean|average|zero|0|most common|most frequent|mode|unknown)\b",
    re.IGNORECASE)
# Only rows MISSING a value: "drop rows with missing email", "remove rows
# where salary is empty". "Delete rows where region is North" is not one
# (it deletes real data and stays refused).
DROP_MISSING = re.compile(
    r"^\s*(?P<apply>apply:\s*)?(?:please\s+)?(?:drop|remove|delete)\s+(?:the\s+|all\s+)?"
    r"rows?\s+(?:with|where|that have|having|that are missing|missing)\s+"
    r"(?:(?:a\s+)?(?:no|missing|empty|blank|null)\s+(?P<target>[\w ]+?)"
    r"|(?P<target2>[\w ]+?)\s+(?:is|are)\s+(?:missing|empty|blank|null))\s*[.?!]*$",
    re.IGNORECASE)
# Any word order: "fill age with the median", "fill median age in rows
# where age is empty", "impute missing ages using the mean", "use the
# median for missing age". It must start with the action.
FILL_START = re.compile(
    r"^\s*(?P<apply>apply:\s*)?(?:please\s+|can you\s+|could you\s+)?"
    r"(?:fill(?:\s+in|\s+up)?|impute|replace|substitute|put|use|set)\b(?P<rest>.*)$",
    re.IGNORECASE)
FILL_METHOD = re.compile(r"\b(median|mean|average|most common|most frequent|mode|unknown)\b",
                         re.IGNORECASE)
FILL_FILLER = re.compile(
    r"\b(the|a|an|missing|empty|blank|null|nan|none|na|n/a|values?|in|into|of|for|to|"
    r"column|all|any|rows?|where|which|that|is|are|has|have|with|using|by|as|it|them|"
    r"its|their|gaps?|holes?|nulls|blanks|=|equals?|no|instead|and|or|both)\b|[^\w\s]",
    re.IGNORECASE)
FILL_OPTIONS = {"median": "fill_median", "mean": "fill_mean", "average": "fill_mean",
                "zero": "fill_zero", "0": "fill_zero", "most common": "fill_most_common",
                "most frequent": "fill_most_common", "mode": "fill_most_common",
                "unknown": "fill_unknown"}


def cleaning_request(message):
    """
    {"target": text naming the column, "issue": "missing_values",
    "option": review option, "confirmed": bool}, or None.
    """

    fill = FILL_START.match(message)

    if fill:
        rest = fill.group("rest")
        how = FILL_METHOD.search(rest)
        zero_fill = re.search(r"\bwith\s+(?:a\s+)?(?:0|zero|zeros|zeroes)\b", rest, re.IGNORECASE)

        if how or zero_fill:
            option = FILL_OPTIONS[how.group(1).lower()] if how else "fill_zero"
            # 0 named anywhere except as the fill value ("where age is 0", "the
            # empty and 0 values") means 0s count as missing, like empty and null.
            without_fill = re.sub(r"\bwith\s+(?:a\s+)?(?:0|zero|zeros|zeroes)\b", " ", rest,
                                  flags=re.IGNORECASE)
            zero_condition = bool(re.search(r"\b(?:0|zero|zeros|zeroes)\b", without_fill,
                                            re.IGNORECASE))
            target = FILL_FILLER.sub(" ", FILL_METHOD.sub(" ", rest))
            target = re.sub(r"\b(?:0|zero|zeros|zeroes)\b", " ", target, flags=re.IGNORECASE)
            target = " ".join(dict.fromkeys(target.split()))     # "age ... age" -> "age"

            if target:
                return {"target": target, "issue": "missing_values", "option": option,
                        "confirmed": bool(fill.group("apply")),
                        "zero_condition": zero_condition}

    drop = DROP_MISSING.match(message)
    target = drop and (drop.group("target") or drop.group("target2") or "").strip()

    if target:
        return {"target": target, "issue": "missing_values", "option": "drop_rows",
                "confirmed": bool(drop.group("apply"))}

    return None


def classify_message(message, columns):
    """
    Sort one chat message.

    Returns:
        {"intent": "data" | "write_request" | "row_preview" | "overview" | "vague"
                   | "feedback" | "chart_change",
         "focus": column words an overview should centre on,
         "preview": {"rows", "from_end"} for row_preview,
         "chart": requested chart type or None,
         "reply": text for vague / feedback, else None}
    """

    words = words_of(message)
    chart = requested_chart(message)
    result = {"intent": "data", "chart": chart, "reply": None}

    # "Fill age with the median": a cleaning choice, not a write (checked
    # first: "replace missing salary with the mean" opens with a write verb).
    cleaning = cleaning_request(message)

    if cleaning:
        return {**result, "intent": "cleaning_request", "cleaning": cleaning}

    # "Delete the North rows": refused before any SQL, even when it names
    # a column ("update units_sold ...").
    if is_write_request(message):
        return {**result, "intent": "write_request", "reply": WRITE_REPLY}

    # "score histogram and movies per decade": one question at a time.
    parts = separate_requests(message)

    if parts:
        return {**result, "intent": "separate_requests", "parts": parts}

    # "sales?", "tell me about the data", "trends?": an overview with
    # questions that work, not a guessed query.
    overview = open_ended(words, columns)

    if overview and not chart:
        return {**result, "intent": "overview", **overview}

    # Anything that names a column is a data question.
    if names_a_column(message, columns):
        return result

    # "top 5 rows": the stored rows, no model call.
    preview = row_preview_request(message)

    if preview:
        return {**result, "intent": "row_preview", "preview": preview}

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
