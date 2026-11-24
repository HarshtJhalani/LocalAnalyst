# ============================================================
# FILE: src/helpers/cleaning_review.py
# PURPOSE: The review step after upload
#
# Safe fixes (names, types, spaces, duplicates, date formats) run on
# upload with no questions. Judgment calls are left as flags in the
# cleaning report; this module turns them into review items, each with
# options and a suggested pick, and checks the user's choices before the
# file is cleaned again with them (data_cleaner.load_and_clean).
#
# Everything is decided from the flags (values, types, counts), never
# from column names, so it works the same for any file. No model call.
#
# Choices shape: {table: {column: {issue: option}}}; for
# "similar_categories" the option may also be {variant: spelling}.
# ============================================================

import json
import logging
import re

from src.helpers.model_client import MODEL_ERRORS, chat_json

logger = logging.getLogger(__name__)

NUMBER_KINDS = {"integer", "float"}
DOT_THOUSANDS = re.compile(r"^-?\d{1,3}(\.\d{3})+$")

LABELS = {
    "leave": "Leave empty",
    "fill_median": "Fill with the median",
    "fill_mean": "Fill with the mean",
    "fill_zero": "Fill with 0",
    "fill_most_common": "Fill with the most common value",
    "fill_unknown": "Fill with \"Unknown\"",
    "drop_rows": "Drop those rows",
    "keep": "Keep as they are",
    "set_null": "Set them to empty",
    "merge": "Merge each group into one spelling",
    "keep_text": "Keep as text",
    "convert": "Convert to numbers (others become empty)",
    "keep_dates": "Use the guessed dates",
    "keep_numbers": "Use the repaired numbers",
    "empty": "Leave those values empty",
    "text": "Keep the column as text",
    "split": "Split into a separate table (one row per value)",
}

TITLES = {
    "missing_values": "Missing values",
    "outliers": "Extreme values",
    "similar_categories": "Look-alike values",
    "partly_numeric": "Mostly numbers, stored as text",
    "decimal_comma": "Numbers with a decimal comma",
    "dates_guessed": "Dates read by best guess",
    "numbers_guessed": "Numbers with typos, read by best guess",
    "list_column": "Several values in one cell",
}


def missing_item(flag, rows):
    """Options depend on the column type; the suggestion on how much is missing."""

    kind = flag.get("kind", "text")
    share = flag.get("count", 0) / max(rows, 1)

    if kind in NUMBER_KINDS:
        options = ["leave", "fill_median", "fill_mean", "fill_zero", "drop_rows"]
        suggested, reason = (("fill_median", "Few values are missing; the median "
                              "is not pulled by extreme values.") if share <= 0.10 else
                             ("leave", "Too many values are missing to fill without "
                              "changing totals and averages."))
    elif kind == "text":
        options = ["leave", "fill_unknown", "fill_most_common", "drop_rows"]
        suggested, reason = ("fill_unknown", "Empty text becomes its own group, "
                             "so counts by this column still add up.")
    else:
        options = ["leave", "fill_most_common", "drop_rows"]
        suggested, reason = ("leave", "Guessing a date or yes/no value would "
                             "invent data.")

    return options, suggested, reason


def review_item(table, flag, rows):
    """One flag -> one review item, or None when there is nothing to decide."""

    issue = flag["issue"]

    if issue not in TITLES or flag.get("column") is None:
        return None

    extra = {}

    if issue == "missing_values":
        options, suggested, reason = missing_item(flag, rows)
    elif issue == "outliers":
        options, suggested, reason = (["keep", "set_null", "drop_rows"], "keep",
                                      "Extreme values may be real; check them first.")
    elif issue == "similar_categories":
        options, suggested, reason = (["keep", "merge"], "merge",
                                      "Typos split one group into several.")
        extra["groups"] = flag.get("suggested") or {}
    elif issue == "partly_numeric":
        examples = [str(x) for x in flag.get("examples", [])]
        dotted = [x for x in examples if DOT_THOUSANDS.match(x)]
        # Words with no digit at all ("confidential", "TBD", "n/a") stand for a
        # missing number; converting makes them empty, the honest value.
        placeholders = examples and all(not any(ch.isdigit() for ch in x) for x in examples)
        options = ["keep_text", "convert"]
        if dotted:
            suggested, reason = ("convert", "The other values use dots as thousands "
                                 "separators (1.007.598).")
        elif placeholders:
            suggested, reason = ("convert", f"The other values ({', '.join(examples[:3])}) "
                                 "are words standing for a missing number; they become "
                                 "empty and the column can be added up and averaged.")
        else:
            suggested, reason = ("keep_text", "The other values are not numbers; "
                                 "converting would empty them.")
    elif issue == "decimal_comma":
        options, suggested, reason = (["keep_text", "convert"], "convert",
                                      "12,5 means 12.5 here.")
    elif issue == "numbers_guessed":
        options, suggested, reason = (["keep_numbers", "empty"], "keep_numbers",
                                      "Check the repairs below; a typo like '178c' is "
                                      "almost always 178.")
        extra["guesses"] = flag.get("guesses", [])
    elif issue == "dates_guessed":
        options, suggested, reason = (["keep_dates", "text"], "keep_dates",
                                      "Check the guesses below; most are clear.")
        extra["guesses"] = flag.get("guesses", [])
    else:  # list_column
        options, suggested, reason = (["split", "keep"], "split",
                                      "Lets you count each value (e.g. per genre) "
                                      "instead of each combination.")

    return {
        "id": f"{table}|{flag['column']}|{issue}",
        "table": table,
        "column": flag["column"],
        "issue": issue,
        "title": TITLES[issue],
        "detail": flag["detail"],
        "options": [{"value": value, "label": LABELS[value]} for value in options],
        "suggested": suggested,
        "reason": reason,
        **extra,
    }


def build_review(cleaning_report, choices=None):
    """
    Review items for every table, with what the user chose last time.

    Tables added by the cleaning itself (split list columns) have nothing
    to review.
    """

    choices = choices or {}
    items = []

    for report in cleaning_report:
        for flag in report["flags"]:
            item = review_item(report["table_name"], flag, report["rows_before"])

            if item:
                item["chosen"] = (choices.get(item["table"], {})
                                  .get(item["column"], {}).get(item["issue"]))
                items.append(item)

    return items


# ------------------------------------------------------------
# Correct spelling of look-alike values: string similarity alone cannot
# tell "New Zealand" from "New Zeland" when each appears once, so one
# short model call per dataset picks the correct spelling in each group.
# The pick must be one of the group's own values; anything else, or no
# model, keeps the rule's pick. The user still confirms it.
# ------------------------------------------------------------

SPELLING_PROMPT = (
    "Each group lists spellings of what is probably one value in a data column. "
    "For each group pick the correctly spelled value, copied exactly from that "
    "group. Answer as JSON: {\"picks\": [\"value for group 1\", ...]}.")
SPELLING_SCHEMA = {"type": "object",
                   "properties": {"picks": {"type": "array", "items": {"type": "string"}}},
                   "required": ["picks"]}
SPELLING_TIMEOUT = 30


def spelling_groups(mapping):
    """{variant: target} -> [[target, variant, ...], ...]"""

    groups = {}

    for variant, target in mapping.items():
        groups.setdefault(target, [target]).append(variant)

    return list(groups.values())


def pick_spellings(settings, column, mapping):
    """
    The rule's {variant: target} with each group's target replaced by the
    model's pick. Returns the mapping unchanged when the model fails.
    """

    groups = spelling_groups(mapping)
    message = f"Column: {column}\n" + "\n".join(
        f"Group {number}: {json.dumps(group)}" for number, group in enumerate(groups, 1))

    try:
        answer, _ = chat_json({**settings, "request_timeout": SPELLING_TIMEOUT},
                              [{"role": "system", "content": SPELLING_PROMPT},
                               {"role": "user", "content": message}], SPELLING_SCHEMA)
    except (ValueError, *MODEL_ERRORS) as exc:
        logger.info("Spellings not picked for %s: %s", column, exc)
        return mapping

    picks = answer.get("picks") or []
    result = {}

    for number, group in enumerate(groups):
        pick = picks[number] if number < len(picks) else None
        target = pick if pick in group else group[0]          # only the group's own values
        result.update({value: target for value in group if value != target})

    return result


def apply_spellings(items, picked):
    """Put stored picks ({table|column: mapping}) into the review items."""

    for item in items:
        key = f"{item['table']}|{item['column']}"

        if item["issue"] == "similar_categories" and key in picked:
            item["groups"] = picked[key]

    return items


# ------------------------------------------------------------
# Values that mean the same thing ("Unrated" / "Not Rated", "NY" / "New
# York"): spelling rules cannot see synonyms, so one short model call per
# dataset suggests groups for each category column. Only the column's own
# values are accepted; nothing is merged until the user confirms it.
# ------------------------------------------------------------

SYNONYM_PROMPT = """You check one data column for values that are two names for exactly the same thing.

Same thing: "Not Rated" and "Unrated"; "NY" and "New York"; "USA" and "United States"; "N/A" and "Not available".
NOT the same thing (different items, even if related): "North" and "South"; "Delhi" and "Mumbai"; "Laptop" and "Monitor"; "Kiran" and "Meera"; "Germany" and "West Germany"; "PG" and "PG-13"; "Gift wrap" and "Late delivery".

Most columns have no such pair: then answer {"pairs": []}.
Otherwise answer {"pairs": [{"a": "...", "b": "...", "keep": "..."}]}, copying values exactly; keep is a or b."""
SYNONYM_SCHEMA = {
    "type": "object",
    "properties": {"pairs": {"type": "array", "items": {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"},
                       "keep": {"type": "string"}},
        "required": ["a", "b", "keep"]}}},
    "required": ["pairs"],
}
CONFIRM_PROMPT = ("Answer as JSON {\"same\": true} only when the two values are two names for "
                  "exactly the same thing in this column, otherwise {\"same\": false}.")
CONFIRM_SCHEMA = {"type": "object", "properties": {"same": {"type": "boolean"}},
                  "required": ["same"]}
SYNONYM_MAX_VALUES = 30
SYNONYM_MAX_GROUP = 3


def category_values(profile):
    """{(table, column): [values]} for text category columns worth checking."""

    found = {}

    for table_name, table in profile["tables"].items():
        if table.get("list_of"):
            continue

        for name, info in table["columns"].items():
            values = info.get("all_values") or [value for value, _ in info.get("top_values") or []]

            if (info["kind"] == "text" and info.get("is_category")
                    and 3 <= len(values) <= SYNONYM_MAX_VALUES):
                found[(table_name, name)] = [str(value) for value in values]

    return found


def ask_json(settings, system, message, schema):
    answer, _ = chat_json({**settings, "request_timeout": SPELLING_TIMEOUT},
                          [{"role": "system", "content": system},
                           {"role": "user", "content": message}], schema)
    return answer


def column_synonyms(settings, column, values):
    """
    {variant: label to keep} for one column. A pair counts only when the
    model names it, both values exist, a second yes/no question agrees,
    and the groups stay small (a "group" of most values is a misreading).
    """

    answer = ask_json(settings, SYNONYM_PROMPT,
                      f"Column: {column}\nValues: {json.dumps(values, ensure_ascii=False)}",
                      SYNONYM_SCHEMA)
    known = set(values)
    mapping = {}

    for pair in answer.get("pairs") or []:
        a, b, keep = pair.get("a"), pair.get("b"), pair.get("keep")
        keep = {"a": a, "b": b}.get(keep, keep)          # "keep": "a" means value a

        if a == b or a not in known or b not in known or keep not in (a, b):
            continue

        confirm = ask_json(settings, CONFIRM_PROMPT,
                           f"Column: {column}\nValue 1: {a}\nValue 2: {b}", CONFIRM_SCHEMA)

        if confirm.get("same") is True:
            variant = b if keep == a else a
            mapping.setdefault(variant, keep)

    groups = {}
    for variant, keep in mapping.items():
        groups.setdefault(keep, {keep}).add(variant)

    if any(len(group) > SYNONYM_MAX_GROUP for group in groups.values()) \
            or len(mapping) + len(groups) > len(values) / 2:
        return {}

    return mapping


def suggest_synonyms(settings, profile):
    """
    {"table|column": {variant: label to keep}}, one column at a time;
    {} for a column when the model fails or finds nothing.
    """

    suggestions = {}

    for (table, column), values in category_values(profile).items():
        try:
            mapping = column_synonyms(settings, column, values)
        except (ValueError, *MODEL_ERRORS) as exc:
            logger.info("No same-meaning check for %s: %s", column, exc)
            continue

        if mapping:
            suggestions[f"{table}|{column}"] = mapping

    return suggestions


def synonym_items(synonyms, choices=None):
    """Review items for the model's same-meaning groups."""

    choices = choices or {}
    items = []

    for key, mapping in synonyms.items():
        table, column = key.split("|", 1)
        groups = {}

        for variant, keep in mapping.items():
            groups.setdefault(keep, []).append(variant)

        shown = "; ".join(f"{', '.join(variants)} -> {keep}" for keep, variants in groups.items())
        items.append({
            "id": f"{table}|{column}|same_meaning",
            "table": table,
            "column": column,
            "issue": "same_meaning",
            "title": "Values that mean the same thing",
            "detail": f"Suggested as the same meaning: {shown}. Check them: related "
                      "values are not always the same.",
            "options": [{"value": "keep", "label": LABELS["keep"]},
                        {"value": "merge", "label": LABELS["merge"]}],
            "suggested": "merge",
            "reason": "Two labels for one meaning split the counts.",
            "groups": mapping,
            "chosen": choices.get(table, {}).get(column, {}).get("same_meaning"),
        })

    return items


def suggested_choices(items):
    """Every item set to its suggestion (what "Apply suggestions" sends)."""

    choices = {}

    for item in items:
        value = item["suggested"]
        if item["issue"] in ("similar_categories", "same_meaning") and value == "merge":
            value = item["groups"] or "merge"
        choices.setdefault(item["table"], {}).setdefault(item["column"], {})[item["issue"]] = value

    return choices


def validate_choices(choices, items):
    """
    Keep only choices that match a review item and one of its options.

    Raises ValueError naming the first bad entry, so a typo never cleans
    the file differently from what the user meant.
    """

    if not isinstance(choices, dict):
        raise ValueError("choices must be an object: {table: {column: {issue: option}}}.")

    allowed = {(item["table"], item["column"], item["issue"]):
               {option["value"] for option in item["options"]} for item in items}
    clean = {}

    for table, columns in choices.items():
        for column, issues in (columns or {}).items():
            for issue, option in (issues or {}).items():
                key = (table, column, issue)

                if key not in allowed:
                    raise ValueError(f"Nothing to decide for {table}.{column} ({issue}).")

                if isinstance(option, dict) and issue in ("similar_categories", "same_meaning"):
                    option = {str(k): str(v) for k, v in option.items()}
                elif option not in allowed[key]:
                    raise ValueError(f"{option!r} is not an option for {table}.{column} "
                                     f"({issue}); choose from {sorted(allowed[key])}.")

                clean.setdefault(table, {}).setdefault(column, {})[issue] = option

    return clean
