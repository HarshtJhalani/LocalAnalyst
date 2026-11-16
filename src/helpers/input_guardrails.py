
# ============================================================
# FILE: src/helpers/input_guardrails.py
# PURPOSE: Checks that run before the model is called
#          (05_self_correction_guardrails.ipynb, step 3)
#
#   safe_filename()        keep only a plain file name (no folders, no '..')
#   check_table_limits()   row / column caps per table
#   check_question()       length, prompt injection, raw SQL as a question
#
# Each check returns {"ok", "reason", ...} so the API can turn a
# refusal into a 400; safe_filename() raises ValueError.
# ============================================================

import re
import unicodedata

from config import (
    MAX_QUESTION_LENGTH,
    MAX_TABLE_COLUMNS,
    MAX_TABLE_ROWS,
    MIN_QUESTION_LENGTH,
)


# ============================================================
# 1. PATTERNS
# ============================================================


INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|above|prior|earlier|your)\s+"
    r"(instructions|rules|prompts?|messages?)",
    r"\bdisregard\b.{0,40}\b(instructions|rules|prompt)",
    r"\b(system|developer|hidden)\s+prompt\b",
    r"\breveal\b.{0,40}\b(prompt|instructions|rules)\b",
    r"\byou\s+are\s+now\b",
    r"\bjailbreak\b",
    r"\bpretend\s+(to\s+be|you\s+are)\b",
]


SQL_STATEMENT_PATTERNS = [
    r"^\s*(select|with)\b.*\bfrom\b.*(\bwhere\b|\bgroup\s+by\b|\border\s+by\b"
    r"|\bjoin\b|;|\*|=|\bas\s*\()",
    r"^\s*select\s+\*",
    r"^\s*insert\s+into\b",
    r"^\s*update\s+\w+\s+set\b",
    r"^\s*delete\s+from\s+\w+\s*(;|$|\bwhere\b)",
    r"^\s*(drop|alter|create)\s+(table|view|index|trigger)\b",
    r"^\s*(attach|detach)\s+(database\b|')",
    r"^\s*pragma\s+\w+",
]


# ============================================================
# 2. UPLOAD CHECKS
# ============================================================


def safe_filename(filename):
    """
    Keep only a plain file name: no folders, no '..', no odd characters.
    'C:\\x\\..\\Sales Report.csv' -> 'Sales_Report.csv'
    """

    name = re.split(r"[\\/]", str(filename or ""))[-1]
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")

    if not name or "." not in name:
        raise ValueError(f"Invalid file name: {filename!r}")

    return name[:120]


def check_table_limits(table_name, frame):
    """Reject tables too large to handle comfortably."""

    rows, columns = frame.shape

    if rows > MAX_TABLE_ROWS:
        return {"ok": False,
                "reason": f"{table_name} has {rows:,} rows; the limit is "
                          f"{MAX_TABLE_ROWS:,}."}

    if columns > MAX_TABLE_COLUMNS:
        return {"ok": False,
                "reason": f"{table_name} has {columns} columns; the limit is "
                          f"{MAX_TABLE_COLUMNS}."}

    return {"ok": True, "reason": None}


# ============================================================
# 3. QUESTION CHECK
# ============================================================


def check_question(question):
    """
    Validate a user question before it reaches the model.

    Returns:
        {"ok", "reason", "question"}  question is the cleaned text
    """

    # Step 1: normalise: drop control characters, collapse spaces.
    text = "".join(char for char in str(question or "")
                   if unicodedata.category(char)[0] != "C" or char in "\n\t")
    text = re.sub(r"\s+", " ", text).strip()

    result = {"ok": False, "reason": None, "question": text}

    # Step 2: length.
    if len(text) < MIN_QUESTION_LENGTH:
        result["reason"] = "Please type a question about your data."
        return result

    if len(text) > MAX_QUESTION_LENGTH:
        result["reason"] = (f"The question is {len(text)} characters; please "
                            f"keep it under {MAX_QUESTION_LENGTH}.")
        return result

    # Step 3: prompt injection.
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            result["reason"] = ("The question tries to change the assistant's "
                                "instructions, so it was not sent.")
            return result

    # Step 4: raw SQL instead of a question.
    if any(re.search(pattern, text, re.IGNORECASE | re.DOTALL)
           for pattern in SQL_STATEMENT_PATTERNS):
        result["reason"] = ("Please ask in plain English; SQL statements are "
                            "not accepted as questions.")
        return result

    result["ok"] = True
    return result
