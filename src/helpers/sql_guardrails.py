
# ============================================================
# FILE: src/helpers/sql_guardrails.py
# PURPOSE: Static check of the model's SQL before it runs
#          (05_self_correction_guardrails.ipynb, step 4)
#
# One statement, starts with SELECT / WITH, no write or admin keywords,
# only the dataset's cleaned tables (or CTE names). String literals and
# comments are removed first. Errors are written for the model: they
# say what to change. The read-only connection and the authorizer in
# sql_executor.py are what actually enforce safety.
#
#   check_sql()   sql + allowed tables -> {"ok", "reason"}
# ============================================================

import re


# ============================================================
# 1. FORBIDDEN KEYWORDS
# ============================================================


FORBIDDEN_SQL_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "attach",
    "detach", "pragma", "vacuum", "reindex", "analyze", "begin", "commit",
    "rollback", "savepoint", "release", "truncate", "load_extension",
}


# ============================================================
# 2. READING THE QUERY
# ============================================================


def strip_literals_and_comments(sql):
    """Replace string literals with '' and drop comments."""

    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)           # 'text' -> ''
    sql = re.sub(r"--[^\n]*", " ", sql)                   # -- comment
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL) # /* comment */

    return sql


def referenced_tables(code_only):
    """Names after FROM / JOIN and in comma lists after FROM."""

    names = set()

    for match in re.finditer(
        r"\b(?:from|join)\s+((?:[\"`\[]?\w+[\"`\]]?(?:\s+(?:as\s+)?\w+)?\s*,\s*)*"
        r"[\"`\[]?\w+[\"`\]]?)",
        code_only, re.IGNORECASE,
    ):
        for part in match.group(1).split(","):
            name = part.strip().split()[0].strip('"`[]')
            names.add(name)

    return names


def cte_names(code_only):
    """Names defined by WITH name AS ( ... ), name2 AS ( ... )."""

    return {match.group(1) for match in re.finditer(
        r"(?:\bwith(?:\s+recursive)?|,)\s*(\w+)\s*(?:\([^)]*\))?\s+as\s*\(",
        code_only, re.IGNORECASE)}


# ============================================================
# 3. CHECK
# ============================================================


def check_sql(sql, allowed_tables):
    """
    Static check of one generated query.

    Returns:
        {"ok", "reason"}
    """

    if not sql or not sql.strip():
        return {"ok": False, "reason": "The query is empty."}

    code_only = strip_literals_and_comments(sql).strip().rstrip(";").strip()
    lowered = code_only.lower()

    # Step 1: one statement only.
    if ";" in code_only:
        return {"ok": False,
                "reason": "Only one statement is allowed; remove the ';' and "
                          "everything after it."}

    # Step 2: read-only statement.
    if not re.match(r"^(select|with)\b", lowered):
        return {"ok": False,
                "reason": "The query must start with SELECT or WITH."}

    words = set(re.findall(r"[a-z_]+", lowered))
    forbidden = sorted(words & FORBIDDEN_SQL_KEYWORDS)

    if forbidden:
        return {"ok": False,
                "reason": f"The query uses {', '.join(forbidden).upper()}; "
                          "only reading data is allowed."}

    # Step 3: known tables only (CTE names are fine).
    allowed = {table.lower() for table in allowed_tables}
    unknown = sorted(name for name in referenced_tables(code_only)
                     if name.lower() not in allowed
                     and name.lower() not in {cte.lower() for cte in cte_names(code_only)})

    if unknown:
        return {"ok": False,
                "reason": f"Unknown table(s) {unknown}. Use only: "
                          f"{', '.join(sorted(allowed_tables))}."}

    return {"ok": True, "reason": None}
