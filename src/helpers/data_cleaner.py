
# ============================================================
# FILE: src/helpers/data_cleaner.py
# PURPOSE: Clean every table of an uploaded file automatically
#          (pipeline stage 2; 01_loading_cleaning.ipynb, steps 3-8)
#
# Applied automatically: header detection, column names, whitespace,
# null placeholders, empty rows / columns, exact duplicates, numbers /
# booleans / dates stored as text, category spelling variants.
# Flagged, never changed: missing values, outliers, ambiguous dates,
# similar categories, partly numeric columns, decimal commas, identifiers
# with leading zeros, constant columns, duplicate rows in a table with
# no key column.
#
#   load_and_clean()   file -> [{"table_name", "cleaned", "raw", "report"}]
# ============================================================

import difflib
import json
import logging
import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd
from dateutil import parser as date_parser

from config import (
    CATEGORY_MAX_UNIQUE_RATIO,
    CURRENCY_SYMBOLS,
    DATE_FORMATS,
    FALSE_VALUES,
    HEADER_MIN_FILLED_RATIO,
    HEADER_SCAN_ROWS,
    NULL_PLACEHOLDERS,
    OUTLIER_GAP_FACTOR,
    OUTLIER_IQR_FACTOR,
    OUTLIER_MAX_SHARE,
    PARTLY_TYPED_RATIO,
    SIMILAR_CATEGORY_MAX_VALUES,
    SIMILAR_CATEGORY_RATIO,
    SINGLE_FAILURE_MIN_RATIO,
    SQL_RESERVED_WORDS,
    TRUE_VALUES,
    TYPE_MATCH_THRESHOLD,
)
from src.helpers.file_loader import read_file


logger = logging.getLogger(__name__)

CURRENCY_PREFIX_PATTERN = re.compile(r"^(rs\.?|inr|usd|eur|gbp)\s*", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
LEADING_ZERO_PATTERN = re.compile(r"^[+-]?0\d+$")      # 00123, not 08.9

# Commas are thousands separators only in these positions: 1,234,567
# (western) or 12,34,567 (Indian). "12,5" or "1.234,56" use a decimal
# comma and must not be read as 125 or 1.23456.
THOUSANDS_PATTERN = re.compile(
    r"^[+-]?(\d{1,3}(,\d{3})+|\d{1,2}(,\d{2})+,\d{3})(\.\d+)?$")
DECIMAL_COMMA_PATTERN = re.compile(r"^[+-]?\d{1,3}(\.\d{3})*,\d+$|^[+-]?\d+,\d+$")


# ============================================================
# 1. NUMBERS STORED AS TEXT
# ============================================================

def parse_number(text):
    """
    Parse a number stored as text.

    Handles currency symbols, thousands separators, percent signs and
    accounting negatives "(1,200)". Internal spaces are refused so phone
    numbers stay text, and so are commas that are not thousands
    separators ("12,5"), because their meaning is not certain.

    Returns:
        (float, unit) or None. unit is "currency", "percent" or None.
    """

    text = str(text).strip()
    unit = None
    negative = False

    # Step 1: accounting negative "(1,200)".
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()

    # Step 2: percent sign.
    if text.endswith("%"):
        unit = "percent"
        text = text[:-1].strip()

    # Step 3: currency symbols and codes.
    for symbol in CURRENCY_SYMBOLS:
        if symbol in text:
            unit = unit or "currency"
            text = text.replace(symbol, "")

    if CURRENCY_PREFIX_PATTERN.match(text):
        unit = unit or "currency"
        text = CURRENCY_PREFIX_PATTERN.sub("", text)

    # Step 4: thousands separators, then a plain number must remain.
    text = text.strip()

    if "," in text:
        if not THOUSANDS_PATTERN.match(text):
            return None

        text = text.replace(",", "")

    if not NUMBER_PATTERN.match(text):
        return None

    number = float(text)

    return (-number if negative else number), unit


# ============================================================
# 2. HEADER ROW AND COLUMN NAMES
# ============================================================

def detect_header_row(grid):
    """
    Index of the header row: the first row (within HEADER_SCAN_ROWS)
    nearly as full as the widest row and made mostly of text.
    """

    filled_counts = grid.notna().sum(axis=1)
    max_filled = int(filled_counts.max())
    min_filled = max(min(2, max_filled),
                     int(HEADER_MIN_FILLED_RATIO * max_filled))

    for index in grid.index[:HEADER_SCAN_ROWS]:
        values = [value for value in grid.loc[index] if pd.notna(value)]

        if len(values) < min_filled:
            continue

        text_values = [value for value in values
                       if parse_number(value) is None]

        if len(text_values) >= HEADER_MIN_FILLED_RATIO * len(values):
            return index

    # Fallback: the first fullest row.
    return int(filled_counts.idxmax())


def split_header(table):
    """Return (header names, body grid, header row index)."""

    grid = table["grid"]
    header_row = 0 if table["header_known"] else detect_header_row(grid)

    headers = list(grid.iloc[header_row])
    body = grid.iloc[header_row + 1:].reset_index(drop=True)
    body.columns = range(len(headers))

    return headers, body, header_row


def clean_column_name(name, position):
    """Turn one header into a lowercase snake_case SQL-safe name."""

    text = "" if name is None or pd.isna(name) else str(name)

    # Step 1: drop units in brackets, keep meaning of a few symbols.
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    text = (text.replace("%", " pct ").replace("#", " number ")
                .replace("&", " and "))

    # Step 2: accents -> ASCII, then snake_case.
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")

    # Step 3: names SQL cannot use bare.
    if not text:
        text = f"column_{position + 1}"

    if text[0].isdigit():
        text = f"col_{text}"

    if text in SQL_RESERVED_WORDS:
        text = f"{text}_value"

    return text


def clean_column_names(headers):
    """
    Clean every header and make the names unique.

    Returns:
        (list of new names, dict of original -> new for changed names)
    """

    names, renamed, used = [], {}, set()

    for position, header in enumerate(headers):
        base = clean_column_name(header, position)
        name = base
        suffix = 2

        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1

        used.add(name)
        names.append(name)

        original = "" if header is None or pd.isna(header) else str(header)

        if original != name:
            renamed[original if original else f"<empty #{position + 1}>"] = name

    return names, renamed


# ============================================================
# 3. REPORT
# ============================================================

def new_report(table_name, source_file):
    """Empty cleaning report for one table."""

    return {
        "table_name": table_name,
        "source_file": source_file,
        "source_info": {},
        "header_row": 0,
        "rows_before": 0,
        "rows_after": 0,
        "columns_before": 0,
        "columns_after": 0,
        "renamed_columns": {},
        "column_types": {},
        "column_units": {},
        "actions": [],
        "flags": [],
    }


def add_action(report, column, action, detail, count=None):
    """Record a change that was applied."""

    report["actions"].append({"column": column, "action": action,
                              "detail": detail, "count": count})


def add_flag(report, column, issue, detail, **data):
    """Record a problem; data holds what the review step needs (counts, pairs)."""

    report["flags"].append({"column": column, "issue": issue, "detail": detail, **data})


def chosen(choices, column, issue, default):
    """The user's choice for one column and issue (see cleaning_review.py)."""

    return (choices or {}).get(column, {}).get(issue, default)


# ============================================================
# 4. CLEANING VALUES
# ============================================================

# UTF-8 text read as Windows-1252: 'piÃ¹' for 'più', 'Ã©' for 'é'.
MOJIBAKE = re.compile("[ÂÃÅ][\u0080-¿‘-›ŒœŠšŸŽžƒˆ˜]")


def repair_mojibake(value):
    """'Per qualche dollaro in piÃ¹' -> 'Per qualche dollaro in più'; else unchanged."""

    if not isinstance(value, str) or not MOJIBAKE.search(value):
        return value

    for encoding in ("cp1252", "latin-1"):
        try:
            return value.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue

    return value


def fix_mixed_encoding(frame, report):
    """Repair text that was saved as UTF-8 inside a Windows-1252 file."""

    repaired = []

    for column in frame.columns:
        original = frame[column]
        fixed = original.map(repair_mojibake)
        changes = original.fillna("") != fixed.fillna("")

        if changes.any():
            repaired += [f"{before!r} -> {after!r}" for before, after
                         in zip(original[changes][:2], fixed[changes][:2])]
            frame[column] = fixed

    if repaired:
        add_action(report, None, "repaired_encoding",
                   f"Repaired garbled accents (mixed file encoding), e.g. {', '.join(repaired[:3])}.",
                   len(repaired))

    return frame


def trim_whitespace(frame, report):
    """Strip and collapse whitespace inside every text cell."""

    changed = 0

    for column in frame.columns:
        original = frame[column]
        trimmed = original.map(
            lambda value: re.sub(r"\s+", " ", value).strip()
            if isinstance(value, str) else value
        )
        changed += int((original.fillna("") != trimmed.fillna("")).sum())
        frame[column] = trimmed

    if changed:
        add_action(report, None, "trimmed_whitespace",
                   "Removed extra spaces inside text values.", changed)

    return frame


def replace_null_placeholders(frame, report):
    """N/A, -, null, ? ... become real nulls."""

    for column in frame.columns:
        is_placeholder = frame[column].map(
            lambda value: isinstance(value, str)
            and value.casefold() in NULL_PLACEHOLDERS
        )
        count = int(is_placeholder.sum())

        if count:
            found = sorted(set(frame.loc[is_placeholder, column]))
            frame.loc[is_placeholder, column] = None
            add_action(report, column, "null_placeholders",
                       f"Treated {found} as missing.", count)

    return frame


def drop_empty_rows_and_columns(frame, report):
    """Remove rows and columns with no values at all."""

    empty_columns = [column for column in frame.columns
                     if frame[column].isna().all()]

    if empty_columns:
        frame = frame.drop(columns=empty_columns)
        add_action(report, None, "dropped_empty_columns",
                   f"Removed empty columns {empty_columns}.", len(empty_columns))

    empty_rows = frame.isna().all(axis=1)

    if empty_rows.any():
        frame = frame.loc[~empty_rows].reset_index(drop=True)
        add_action(report, None, "dropped_empty_rows",
                   "Removed rows with no values.", int(empty_rows.sum()))

    return frame


def drop_duplicate_rows(frame, report):
    """
    Remove exact duplicate rows, keeping the first, when they look
    like accidents: the table has a key (some column is unique once the
    duplicates are gone, so the repeated rows repeated an ID) and the
    repeats are fewer than the rows that remain. Without a key, or when
    most rows are repeats (a plain log of transactions), identical rows
    can be real, so they are only flagged.
    """

    duplicates = frame.duplicated(keep="first")

    if not duplicates.any():
        return frame

    count = int(duplicates.sum())
    deduplicated = frame.loc[~duplicates]
    has_key = count < len(deduplicated) and any(
        deduplicated[column].notna().all() and deduplicated[column].is_unique
        for column in deduplicated.columns)

    if not has_key:
        add_flag(report, None, "duplicate_rows",
                 f"{count} row(s) repeat an earlier row exactly; kept because "
                 "no column identifies a row.")
        return frame

    add_action(report, None, "dropped_duplicate_rows",
               "Removed exact duplicate rows.", count)

    return deduplicated.reset_index(drop=True)


# ============================================================
# 5. DETECTING COLUMN TYPES
# ============================================================

def type_matches(matched, total):
    """
    True when enough values parse: at least TYPE_MATCH_THRESHOLD, or all
    but one when that one leaves at least SINGLE_FAILURE_MIN_RATIO.
    """

    if total == 0:
        return False

    rate = matched / total

    return (rate >= TYPE_MATCH_THRESHOLD
            or (total - matched == 1 and rate >= SINGLE_FAILURE_MIN_RATIO))


def try_boolean(values):
    """Yes/No, Y/N, True/False -> booleans."""

    lowered = values.str.casefold()
    known = lowered.isin(TRUE_VALUES | FALSE_VALUES)

    if not type_matches(int(known.sum()), len(known)):
        return None

    converted = lowered.map(
        lambda value: True if value in TRUE_VALUES
        else False if value in FALSE_VALUES else None
    )

    return {"kind": "boolean", "values": converted,
            "failed": values[~known], "unit": None}


# Typos inside numbers, repaired only in a column that is already mostly
# numbers and only for a few values; every repair is listed for review.
MAX_REPAIRED_SHARE = 0.2


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
NUMBER_SCALES = {"hundred": 100, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
                 "lac": 100_000, "million": 1_000_000, "crore": 10_000_000,
                 "crores": 10_000_000, "billion": 1_000_000_000}
SHORT_SCALE = re.compile(r"^([\d.,]+)\s*(k|K|M|mn|Mn|bn|Bn|lakhs?|lac|cr|crores?)$")
SHORT_SCALES = {"k": 1_000, "m": 1_000_000, "mn": 1_000_000, "bn": 1_000_000_000,
                "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "cr": 10_000_000,
                "crore": 10_000_000, "crores": 10_000_000}


def words_to_number(text):
    """
    'two hundred' -> 200, 'fifty-five' -> 55, 'one thousand two hundred
    and five' -> 1205, 'three lakh' -> 300000, '1.5k' -> 1500, '2 crore'
    -> 20000000. Anything else -> None.
    """

    text = str(text).strip()
    short = SHORT_SCALE.match(text)

    if short:
        number = parse_number(short.group(1))
        return None if number is None else number[0] * SHORT_SCALES[short.group(2).lower()]

    words = re.findall(r"[a-z]+", text.lower().replace("-", " "))

    if not words or re.search(r"[^a-zA-Z\s,-]", text):
        return None

    total, current, seen = 0, 0, False

    for word in words:
        if word == "and":
            continue
        if word in NUMBER_WORDS:
            current += NUMBER_WORDS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in NUMBER_SCALES:
            total += (current or 1) * NUMBER_SCALES[word]
            current = 0
        else:
            return None
        seen = True

    return float(total + current) if seen else None


def repair_number(text):
    """
    One typo'd number -> (number, unit) or None: '178c' -> 178,
    '$ 4o8,035,783' -> 408035783, '9,.0' -> 9.0, '8:8' -> 8.8,
    '++8.7' -> 8.7, '8.7.' -> 8.7, '8,7e-0' -> 8.7. Words ('Inf',
    'Not Applicable') stay None.
    """

    text = str(text).strip()

    if not re.search(r"\d", text):
        return None

    text = re.sub(r"(?<=[\d,])[oO](?=[\d,])", "0", text)       # 4o8 -> 408
    text = re.sub(r"^\++", "", text)                             # ++8.7
    text = re.sub(r"(?<=\d):(?=\d)", ".", text)                  # 8:8
    text = re.sub(r"[.,]{2,}", ".", text)                        # 8..8, 9,.0

    if "," in text and not THOUSANDS_PATTERN.match(text.lstrip("$€£₹¥ ")):
        text = text.replace(",", ".") if text.count(",") == 1 else text

    for candidate in (text, re.sub(r"[A-Za-z.]+$", "", text)):  # 8.7e-0 first, then 178c
        result = parse_number(candidate.strip())
        if result is not None:
            return result

    return None


def repaired(values, parsed):
    """
    Values that did not parse, repaired: (parsed with repairs, [(raw, number)]).
    Nothing is repaired unless at least half the column parsed as is and
    the repairs stay few (a whole column of '12,5' is a convention, not typos).
    """

    failed = parsed.isna()

    if failed.mean() > MAX_REPAIRED_SHARE or parsed.notna().mean() < PARTLY_TYPED_RATIO:
        return parsed, []

    fixes = values[failed].map(repair_number)
    fixed = fixes.notna()

    if not fixed.any():
        return parsed, []

    parsed = parsed.copy()
    parsed[fixes[fixed].index] = fixes[fixed]
    guesses = [(raw, result[0]) for raw, result in zip(values[fixes[fixed].index], fixes[fixed])]

    return parsed, guesses


def try_number(values):
    """
    Numbers stored as text -> integers or floats.

    Returns a result dict, or a dict with kind "partly_numeric" /
    "leading_zero" / "decimal_comma" when the column is only flagged.
    """

    # Step 1: identifiers such as 00123 must keep their zeros.
    if values.str.match(LEADING_ZERO_PATTERN).any():
        if type_matches(int(values.map(parse_number).notna().sum()), len(values)):
            return {"kind": "leading_zero"}
        return None

    # Step 2: parse every value. Written numbers ('two hundred') are read
    # too, and a few typos repaired; both are listed for the review.
    parsed = values.map(parse_number)
    exact = parsed.notna()
    worded = values[~exact].map(words_to_number)
    worded = worded[worded.notna()]
    parsed = parsed.copy()
    parsed[worded.index] = [(number, None) for number in worded]
    word_guesses = [(raw, number) for raw, number in zip(values[worded.index], worded)]
    parsed, guessed = repaired(values, parsed)
    guessed = word_guesses + guessed
    ok = parsed.notna()
    rate = ok.mean()

    if guessed and not type_matches(int(ok.sum()), len(ok)):
        parsed, guessed, ok = parsed.where(exact), [], exact      # repairs did not help
        rate = ok.mean()

    if not type_matches(int(ok.sum()), len(ok)):
        # "12,5" / "1.234,56": numbers with a decimal comma. Guessing
        # the convention could change values silently, so only flag.
        decimal_commas = ~ok & values.str.match(DECIMAL_COMMA_PATTERN)

        if decimal_commas.any() and type_matches(
                int((ok | decimal_commas).sum()), len(ok)):
            return {"kind": "decimal_comma",
                    "example": values[decimal_commas].iloc[0]}

        if rate >= PARTLY_TYPED_RATIO:
            return {"kind": "partly_numeric", "rate": rate,
                    "failed": values[~ok]}
        return None

    # Step 3: integers when every number is whole.
    numbers = parsed[ok].map(lambda result: result[0])
    units = {result[1] for result in parsed[ok] if result[1]}
    is_whole = numbers.map(lambda number: number.is_integer()).all()

    converted = pd.Series(pd.NA, index=values.index, dtype="Float64")
    converted[ok] = numbers

    if is_whole and numbers.abs().max() < 2 ** 53:
        converted = converted.astype("Int64")

    return {
        "kind": "integer" if is_whole else "float",
        "values": converted,
        "failed": values[~ok],
        "unit": "/".join(sorted(units)) if units else None,
        "guessed": [(raw, int(number) if is_whole else number) for raw, number in guessed],
        "repaired_index": [index for index in values.index if ok[index] and not exact[index]],
    }


DOT_THOUSANDS = re.compile(r"^-?\d{1,3}(\.\d{3})+$")


def force_number(values, decimal_comma=False):
    """
    Convert a column the user asked to treat as numbers. "1.007.598" (dot
    thousands) and, when decimal_comma, "12,5" are read; anything else
    that is not a number becomes null.
    """

    def one(text):
        text = str(text).strip()
        if DOT_THOUSANDS.match(text):
            text = text.replace(".", "")
        elif decimal_comma and re.match(r"^-?[\d.]*\d,\d+$", text):
            text = text.replace(".", "").replace(",", ".")
        return parse_number(text)

    parsed = values.map(one)
    ok = parsed.notna()
    numbers = parsed[ok].map(lambda result: result[0])
    is_whole = numbers.map(lambda number: number.is_integer()).all()
    converted = pd.Series(pd.NA, index=values.index, dtype="Float64")
    converted[ok] = numbers

    if is_whole and (numbers.abs().max() if len(numbers) else 0) < 2 ** 53:
        converted = converted.astype("Int64")

    return {"kind": "integer" if is_whole else "float", "values": converted,
            "failed": values[~ok], "unit": None, "by_choice": True}


def swap_day_month(date_format):
    """%d/%m/%Y <-> %m/%d/%Y"""

    return (date_format.replace("%d", "@").replace("%m", "%d")
                       .replace("@", "%m"))


# Month names in other languages that turn up in hand-typed dates.
FOREIGN_MONTHS = {
    "enero": "jan", "febrero": "feb", "marzo": "mar", "abril": "apr",
    "mayo": "may", "junio": "jun", "julio": "jul", "agosto": "aug",
    "septiembre": "sep", "setiembre": "sep", "octubre": "oct",
    "noviembre": "nov", "diciembre": "dec",
    "janvier": "jan", "fevrier": "feb", "mars": "mar", "avril": "apr",
    "mai": "may", "juin": "jun", "juillet": "jul", "aout": "aug",
    "septembre": "sep", "octobre": "oct", "novembre": "nov", "decembre": "dec",
}
DATE_FILLER = re.compile(r"\b(the|of|year|on|day)\b", re.IGNORECASE)
ORDINAL = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)


def tidy_date_text(text):
    """'The 6th of marzo, year 1951' -> '6 mar 1951'; '23 -07-2008' -> '23-07-2008'."""

    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = ORDINAL.sub(r"\1", text)
    text = DATE_FILLER.sub(" ", text)
    text = re.sub(r"\s*([-/.])\s*", r"\1", text)
    words = [FOREIGN_MONTHS.get(word.lower(), word) for word in re.split(r"(\W+)", text)]

    return re.sub(r"\s+", " ", "".join(words)).strip(" ,")


def guess_one_date(text, day_first):
    """One hand-typed date -> Timestamp, or NaT. Two-digit years never land in the future."""

    try:
        parsed = date_parser.parse(tidy_date_text(text), dayfirst=day_first,
                                   default=pd.Timestamp(2000, 1, 1).to_pydatetime())

    except (ValueError, OverflowError, TypeError):
        return pd.NaT

    if parsed.year > date.today().year:
        parsed = parsed.replace(year=parsed.year - 100)

    return pd.Timestamp(parsed)


def try_mixed_dates(values):
    """
    Dates typed in many formats in one column.

    Values matching a known format are read with it. The rest are guessed
    one by one; day-first or month-first follows whichever the column's
    clear cases use (a day above 12 settles it). Returns None unless
    enough values are dates.
    """

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    for date_format in DATE_FORMATS:
        missing = parsed.isna()
        if not missing.any():
            break
        parsed[missing] = pd.to_datetime(values[missing], format=date_format, errors="coerce")

    exact = parsed.notna()

    # Must already look mostly like dates before guessing the rest.
    if exact.mean() < 0.5:
        return None

    day_first = sum(bool(re.match(r"\s*(1[3-9]|2\d|3[01])\D", v)) for v in values) >= \
        sum(bool(re.match(r"\s*\d{1,2}\D+(1[3-9]|2\d|3[01])\D", v)) for v in values)
    guessed = values[~exact].map(lambda v: guess_one_date(v, day_first))
    parsed[~exact] = guessed

    if not type_matches(int(parsed.notna().sum()), len(parsed)):
        return None

    has_time = (parsed.dropna().dt.normalize() != parsed.dropna()).any()

    return {
        "kind": "datetime" if has_time else "date",
        "values": parsed,
        "failed": values[parsed.isna()],
        "unit": None,
        "format": "mixed",
        "guessed": [(raw, str(value.date())) for raw, value in
                    zip(values[~exact], guessed) if pd.notna(value)],
    }


def try_date(values):
    """
    Dates stored as text -> datetimes.

    Every known format is tried; the best one wins. When the day-first
    and month-first versions fit equally well the column is ambiguous.
    """

    results = []

    for date_format in DATE_FORMATS:
        parsed = pd.to_datetime(values, format=date_format, errors="coerce")

        if type_matches(int(parsed.notna().sum()), len(parsed)):
            results.append((parsed.notna().mean(), date_format, parsed))

    if not results:
        return try_mixed_dates(values)

    results.sort(key=lambda result: result[0], reverse=True)
    best_rate, best_format, best_parsed = results[0]

    # Ambiguity: the swapped format fits just as well.
    swapped = swap_day_month(best_format)

    if swapped != best_format and any(
        date_format == swapped and rate == best_rate
        for rate, date_format, _ in results
    ):
        return {"kind": "ambiguous_date", "formats": [best_format, swapped]}

    has_time = (best_parsed.dropna().dt.normalize() != best_parsed.dropna()).any()

    return {
        "kind": "datetime" if has_time else "date",
        "values": best_parsed,
        "failed": values[best_parsed.isna()],
        "unit": None,
        "format": best_format,
    }


def convert_column(frame, column, report, choices=None):
    """Infer one column's type, convert it and record the outcome."""

    values = frame[column].dropna().astype(str)

    if values.empty:
        report["column_types"][column] = "text"
        return frame

    for attempt in (try_boolean, try_number, try_date):
        result = attempt(values)

        if result is None:
            continue

        kind = result["kind"]

        # Flag-only outcomes: keep the text, explain why.
        if kind == "leading_zero":
            add_flag(report, column, "identifier_kept_as_text",
                     f"Values have leading zeros (e.g. {values.iloc[0]!r}), "
                     "so they stay text.")
            break

        if kind == "decimal_comma":
            add_flag(report, column, "decimal_comma",
                     f"Numbers use a decimal comma (e.g. {result['example']!r}); "
                     "kept as text.", example=str(result["example"]))
            if chosen(choices, column, "decimal_comma", "keep_text") != "convert":
                break
            result = force_number(values, decimal_comma=True)
            kind = result["kind"]

        elif kind == "partly_numeric":
            not_numbers = sorted(set(result["failed"]))[:5]
            add_flag(report, column, "partly_numeric",
                     f"{result['rate']:.0%} of values are numbers; kept as text. "
                     f"Not numbers: {not_numbers}", examples=not_numbers)
            if chosen(choices, column, "partly_numeric", "keep_text") != "convert":
                break
            result = force_number(values)
            kind = result["kind"]

        if kind == "ambiguous_date":
            add_flag(report, column, "ambiguous_dates",
                     "Dates fit both day-first and month-first "
                     f"({' / '.join(result['formats'])}); kept as text.")
            break

        # A real conversion. Rows that were null stay null (reindex).
        converted = result["values"].reindex(frame.index)

        if kind == "boolean":
            converted = converted.astype("boolean")

        original = frame[column]
        frame[column] = converted

        detail = f"Converted text to {kind}"

        if result.get("format"):
            detail += f" (format {result['format']})"

        if result["unit"]:
            detail += f"; removed {result['unit']} symbols"
            report["column_units"][column] = result["unit"]

        if result.get("guessed") and kind in ("integer", "float"):
            # Each distinct spelling once ('thirty' -> 30 is not repeated 176 times).
            distinct = list(dict.fromkeys((raw, value) for raw, value in result["guessed"]))
            examples = ", ".join(f"{raw!r} -> {value}" for raw, value in distinct[:5])
            add_flag(report, column, "numbers_guessed",
                     f"{len(result['guessed'])} values were written as words or with "
                     f"typos and were read as numbers: {examples}.",
                     guesses=[[raw, value] for raw, value in distinct[:20]])
            if chosen(choices, column, "numbers_guessed", "keep_numbers") == "empty":
                frame.loc[result["repaired_index"], column] = pd.NA

        elif result.get("guessed"):
            distinct = list(dict.fromkeys((raw, value) for raw, value in result["guessed"]))
            examples = ", ".join(f"{raw!r} -> {value}" for raw, value in distinct[:5])
            add_flag(report, column, "dates_guessed",
                     f"{len(result['guessed'])} dates were in unusual formats and "
                     f"were read by best guess: {examples}.",
                     guesses=[list(pair) for pair in distinct[:20]])
            if chosen(choices, column, "dates_guessed", "keep_dates") == "text":
                frame[column] = original
                report["column_types"][column] = "text"
                return frame

        failed = sorted(set(result["failed"]))

        if failed:
            detail += f"; values that did not parse became null: {failed[:5]}"

        if result.get("by_choice"):
            detail += " (your choice)"

        add_action(report, column, f"converted_to_{kind}", detail + ".", len(values))
        report["column_types"][column] = kind
        return frame

    report["column_types"][column] = "text"

    return frame


# ============================================================
# 6. CATEGORIES AND FLAGS
# ============================================================

def is_category_column(series):
    """Low-cardinality text column (city, product, status ...)."""

    values = series.dropna()

    return (not values.empty
            and values.nunique() <= CATEGORY_MAX_UNIQUE_RATIO * len(values))


def merge_category_variants(frame, report):
    """'delhi', 'DELHI' -> one spelling, 'Delhi'."""

    for column, kind in report["column_types"].items():
        if kind != "text" or not is_category_column(frame[column]):
            continue

        groups = {}

        for value in frame[column].value_counts().index:
            groups.setdefault(value.casefold(), []).append(value)

        mapping = {}

        for variants in groups.values():
            if len(variants) > 1:
                # value_counts is sorted by frequency. Prefer the most
                # common normally cased spelling ("Beverages" over
                # "BEVERAGES"), then any not all lowercase ("HR" over
                # "hr"), then simply the most common.
                proper = [variant for variant in variants
                          if not variant.islower() and not variant.isupper()]
                cased = [variant for variant in variants
                         if not variant.islower()]
                canonical = (proper or cased or variants)[0]

                for variant in variants:
                    if variant != canonical:
                        mapping[variant] = canonical

        if mapping:
            count = int(frame[column].isin(mapping).sum())
            frame[column] = frame[column].replace(mapping)
            add_action(report, column, "merged_category_variants",
                       f"Merged spelling variants: {mapping}.", count)

    return frame


def flag_missing_values(frame, report):
    for column in frame.columns:
        missing = int(frame[column].isna().sum())

        if missing:
            add_flag(report, column, "missing_values",
                     f"{missing} of {len(frame)} values are missing "
                     f"({missing / len(frame):.0%}).",
                     count=missing, kind=report["column_types"].get(column, "text"))


def apply_value_choices(frame, report, choices, outliers_by_column):
    """
    The user's choices for outliers and missing values, in that order
    (an outlier set to null can then be filled). Defaults change nothing.
    """

    for column, outliers in outliers_by_column.items():
        choice = chosen(choices, column, "outliers", "keep")
        hit = frame[column].astype("Float64").isin(outliers).fillna(False).astype(bool)

        if choice == "set_null":
            frame.loc[hit, column] = pd.NA
            add_action(report, column, "outliers_set_null",
                       "Extreme values set to empty (your choice).", int(hit.sum()))
        elif choice == "drop_rows":
            frame = frame[~hit]
            add_action(report, column, "outlier_rows_dropped",
                       "Rows with extreme values dropped (your choice).", int(hit.sum()))

    # "Fill age where age is 0": the user's 0s are missing values too.
    for column in list(frame.columns):
        if chosen(choices, column, "zeros_as_missing", False) is True \
                and pd.api.types.is_numeric_dtype(frame[column]):
            zeros = frame[column] == 0
            zeros = zeros.fillna(False).astype(bool)

            if zeros.any():
                frame.loc[zeros, column] = pd.NA
                add_action(report, column, "zeros_as_missing",
                           f"Treated {int(zeros.sum())} values of 0 as missing (your choice).",
                           int(zeros.sum()))

    for column in list(frame.columns):
        choice = chosen(choices, column, "missing_values", "leave")
        empty = frame[column].isna()

        if choice == "leave" or not empty.any():
            continue

        values = frame[column].dropna()
        fill = None

        if choice == "drop_rows":
            frame = frame[~empty]
            add_action(report, column, "missing_rows_dropped",
                       "Rows with this value missing were dropped (your choice).",
                       int(empty.sum()))
            continue

        if choice in ("fill_median", "fill_mean") and not values.empty:
            number = values.astype(float).median() if choice == "fill_median" \
                else values.astype(float).mean()
            fill = round(number) if report["column_types"].get(column) == "integer" \
                else round(number, 4)
        elif choice == "fill_zero":
            fill = 0
        elif choice == "fill_most_common" and not values.empty:
            fill = values.mode().iloc[0]
        elif choice == "fill_unknown":
            fill = "Unknown"

        if fill is not None:
            frame[column] = frame[column].fillna(fill)
            add_action(report, column, "missing_filled",
                       f"Filled {int(empty.sum())} missing values with {fill!r} "
                       "(your choice).", int(empty.sum()))

    # Values that mean the same thing ("Unrated" / "Not Rated"), as the user
    # confirmed them in the review: last, so a filled "Unknown" can join too.
    for column in list(frame.columns):
        mapping = chosen(choices, column, "same_meaning", "keep")

        if isinstance(mapping, dict):
            mapping = {variant: target for variant, target in mapping.items()
                       if target and variant != target}
            hit = frame[column].isin(mapping)

            if hit.any():
                frame[column] = frame[column].replace(mapping)
                add_action(report, column, "merged_same_meaning",
                           f"Merged values with the same meaning (your choice): {mapping}.",
                           int(hit.sum()))

    return frame


def find_outliers(values):
    """
    Extreme values in a numeric Series.

    Rule 1: outside the IQR fences, only if such values are rare.
    Rule 2: the highest / lowest positive values when they are
            OUTLIER_GAP_FACTOR times away from the next value.
    """

    max_count = max(1, int(OUTLIER_MAX_SHARE * len(values)))
    outliers = set()

    # Rule 1: IQR fences.
    q1, q3 = values.quantile([0.25, 0.75])
    spread = OUTLIER_IQR_FACTOR * (q3 - q1)
    fenced = values[(values < q1 - spread) | (values > q3 + spread)]

    if len(fenced) <= max_count:
        outliers.update(fenced.tolist())

    # Rule 2: a big jump at either end of the sorted positive values.
    ordered = sorted(values[values > 0].unique())

    for position in range(len(ordered) - 1, 0, -1):
        if len(ordered) - position > max_count:
            break
        if ordered[position] >= OUTLIER_GAP_FACTOR * ordered[position - 1]:
            outliers.update(ordered[position:])
            break

    for position in range(min(max_count, len(ordered) - 1)):
        if ordered[position + 1] >= OUTLIER_GAP_FACTOR * ordered[position]:
            outliers.update(ordered[:position + 1])
            break

    return outliers


def flag_outliers(frame, report):
    """Flag extreme values; returns {column: set of outlier values}."""

    outliers_by_column = {}

    for column, kind in report["column_types"].items():
        if kind not in {"integer", "float"}:
            continue

        values = frame[column].dropna().astype(float)

        if len(values) < 8:
            continue

        outliers = find_outliers(values)

        if outliers:
            count = int(values.isin(outliers).sum())
            examples = [
                int(value) if float(value).is_integer() else float(value)
                for value in sorted(outliers, key=abs, reverse=True)[:3]
            ]
            add_flag(report, column, "outliers",
                     f"{count} extreme value(s), e.g. {examples}; kept.",
                     count=count, examples=examples)
            outliers_by_column[column] = outliers

    return outliers_by_column


# Prefixes that make a different (often opposite) word: female, inactive,
# unpaid, nonprofit, disabled, impossible, irregular, illegal, deactivated.
WORD_PREFIXES = ("fe", "un", "in", "non", "non-", "non ", "dis", "im", "ir", "il",
                 "de", "anti", "anti-", "not ", "no ", "pre", "post", "sub", "re")


def prefixed_pair(first, second):
    """True when one value is the other with a word-changing prefix."""

    a, b = first.casefold().strip(), second.casefold().strip()
    short, long = sorted((a, b), key=len)

    return long.endswith(short) and long[:len(long) - len(short)] in WORD_PREFIXES


def flag_similar_categories(frame, report, choices=None):
    for column, kind in report["column_types"].items():
        if kind != "text" or not is_category_column(frame[column]):
            continue

        values = sorted(frame[column].dropna().unique())

        if len(values) > SIMILAR_CATEGORY_MAX_VALUES:
            continue

        pairs = []

        for position, first in enumerate(values):
            for second in values[position + 1:]:
                # "Product 1" vs "Product 2" differ only in digits: separate
                # items. "Italy1" vs "Italy" (one has none) is a typo.
                if (re.sub(r"\d", "", first) == re.sub(r"\d", "", second)
                        and re.search(r"\d", first) and re.search(r"\d", second)):
                    continue

                # "Male" / "Female", "Active" / "Inactive", "Paid" / "Unpaid":
                # a word and its prefixed form are different values, not typos.
                if prefixed_pair(first, second):
                    continue

                ratio = difflib.SequenceMatcher(
                    None, first.casefold(), second.casefold()).ratio()

                if ratio >= SIMILAR_CATEGORY_RATIO:
                    pairs.append((first, second))

        if pairs:
            add_flag(report, column, "similar_categories",
                     f"Possibly the same category: {pairs[:5]}; not merged.",
                     pairs=[list(pair) for pair in pairs[:20]],
                     suggested=similar_groups(frame, column, pairs[:20]))

            choice = chosen(choices, column, "similar_categories", "keep")

            if choice == "merge":
                merge_similar(frame, report, column, similar_groups(frame, column, pairs))
            elif isinstance(choice, dict) and choice:
                merge_similar(frame, report, column, choice)


def similar_groups(frame, column, pairs):
    """
    {variant: suggested spelling} for each group of look-alike values.
    The review step shows these and lets the user pick another spelling.
    """

    counts = frame[column].value_counts()
    group_of = {}

    for first, second in pairs:
        group = group_of.get(first, {first}) | group_of.get(second, {second})
        for value in group:
            group_of[value] = group

    mapping = {}

    for group in {frozenset(group) for group in group_of.values()}:
        # Most common wins; on a tie, the spelling closest to the others
        # ("New Zealand" over "New Zesland").
        keep = max(group, key=lambda value: (
            counts.get(value, 0),
            sum(difflib.SequenceMatcher(None, value.casefold(), other.casefold()).ratio()
                for other in group),
            value))
        mapping.update({value: keep for value in group if value != keep})

    return mapping


def merge_similar(frame, report, column, mapping):
    """Replace look-alike values by the chosen spelling ({variant: target})."""

    mapping = {variant: target for variant, target in mapping.items()
               if target and variant != target}

    if not mapping:
        return

    count = int(frame[column].isin(mapping).sum())
    frame[column] = frame[column].replace(mapping)
    add_action(report, column, "merged_similar_categories",
               f"Merged look-alike values (your choice): {mapping}.", count)


def flag_constant_columns(frame, report):
    for column in frame.columns:
        if len(frame) > 1 and frame[column].nunique(dropna=False) == 1:
            add_flag(report, column, "constant_column",
                     f"Every row has the same value ({frame[column].iloc[0]!r}).")


# ============================================================
# 7. FULL CLEANING
# ============================================================

def make_table_name(file_path, sheet, sheet_count):
    """sales.csv -> sales; book.xlsx sheet 'Q1' -> book_q1"""

    name = clean_column_name(Path(file_path).stem, 0)

    if sheet is not None and sheet_count > 1:
        name = f"{name}_{clean_column_name(sheet, 0)}"

    return name


def clean_table(table, table_name, source_file, choices=None):
    """
    Clean one grid table. choices: {column: {issue: option}} from the
    review step; without them only the safe automatic fixes run.

    Returns:
        (cleaned DataFrame, raw DataFrame, report dict)
    """

    report = new_report(table_name, source_file)
    report["source_info"] = table["source_info"]

    # Step 1: header row and column names.
    headers, body, header_row = split_header(table)
    names, renamed = clean_column_names(headers)
    body.columns = names

    report.update(header_row=header_row, renamed_columns=renamed,
                  rows_before=len(body), columns_before=len(names))

    if header_row:
        add_action(report, None, "skipped_title_rows",
                   f"Header found on row {header_row + 1}; rows above it "
                   "were dropped.", header_row)

    # Step 2: raw copy keeps the original values.
    raw = body.copy()

    # Step 3: value cleaning.
    frame = body.copy()
    frame = fix_mixed_encoding(frame, report)
    frame = trim_whitespace(frame, report)
    frame = replace_null_placeholders(frame, report)
    frame = drop_empty_rows_and_columns(frame, report)
    frame = drop_duplicate_rows(frame, report)

    if frame.empty:
        raise ValueError(f"{table_name}: no data rows left after cleaning.")

    # Step 4: types, then categories.
    for column in frame.columns:
        frame = convert_column(frame, column, report, choices)

    frame = merge_category_variants(frame, report)

    # Step 5: flags only.
    flag_missing_values(frame, report)
    outliers_by_column = flag_outliers(frame, report)
    flag_similar_categories(frame, report, choices)
    flag_constant_columns(frame, report)

    # Step 6: the user's choices from the review step.
    frame = apply_value_choices(frame, report, choices, outliers_by_column)

    if frame.empty:
        raise ValueError(f"{table_name}: your choices removed every row.")

    report["rows_after"] = len(frame)
    report["columns_after"] = len(frame.columns)

    logger.info("Cleaned %s | rows %d -> %d | columns %d -> %d | "
                "%d action(s), %d flag(s)",
                table_name, report["rows_before"], report["rows_after"],
                report["columns_before"], report["columns_after"],
                len(report["actions"]), len(report["flags"]))

    return frame, raw, report


LIST_SEPARATOR = re.compile(r"\s*[,;|]\s*")


def list_items(value):
    """'Crime, Drama' or '["vip", "online"]' (a JSON list) -> its items."""

    text = str(value).strip()

    if text.startswith("[") and text.endswith("]"):
        try:
            items = json.loads(text)
            if isinstance(items, list):
                return [str(item).strip() for item in items if str(item).strip()]
        except ValueError:
            pass

    return [x for x in LIST_SEPARATOR.split(text) if x]


def is_json_list_column(values):
    """Every cell is a JSON list: unambiguous structure, whatever the row count."""

    return bool(len(values)) and values.map(
        lambda v: str(v).strip().startswith("[") and str(v).strip().endswith("]")
        and isinstance(json_or_none(v), list)).all()


def json_or_none(value):
    try:
        return json.loads(str(value))
    except ValueError:
        return None


def is_list_column(series):
    """
    Text holding several short values per cell ("Crime, Drama"), where
    the parts repeat across rows far more than whole cells do; or a
    column of JSON lists ('["vip", "online"]').
    """

    values = series.dropna().astype(str)

    if is_json_list_column(values):
        return True

    if len(values) < 10:
        return False

    has_separator = values.str.contains(r"[,;|]").mean()
    parts = values.map(list_items)
    distinct_parts = {x.lower() for row in parts for x in row}
    average_part = sum(len(x) for x in distinct_parts) / max(len(distinct_parts), 1)

    return (has_separator >= 0.3
            and len(distinct_parts) <= 0.6 * values.nunique()
            and average_part <= 25)


def split_list_columns(frame, report, choices=None):
    """
    For each list column, a second table with one row per value:
    <table>_<column>(row_id, <column>_value). The cleaned table gains a row_id
    column (1, 2, 3 ...) to join on. The original column stays as it is.
    """

    extra = []
    list_columns = []

    for column in [c for c in frame.columns
                   if report["column_types"].get(c) == "text" and is_list_column(frame[c])]:
        values = frame[column].dropna().astype(str)
        example = values[values.str.contains(r"[,;|]")].iloc[0]
        add_flag(report, column, "list_column",
                 f"Holds several values per row (e.g. {example!r}).", example=example)

        if chosen(choices, column, "list_column", "split") == "split":
            list_columns.append(column)

    # A file may already have a "row_id" column: never reuse it as the key.
    key = "row_id"
    while key in frame.columns and list_columns:
        key = "_" + key

    for column in list_columns:
        if key not in frame.columns:
            frame.insert(0, key, range(1, len(frame) + 1))
            report["column_types"] = {key: "integer", **report["column_types"]}

        # A different column name than the parent's, so a query can never
        # mix them up ("ambiguous column name: genre").
        value = f"{column}_value"
        child = (frame[[key, column]].dropna()
                 .assign(**{value: lambda f: f[column].astype(str).map(
                     list_items)})
                 [[key, value]]
                 .explode(value).reset_index(drop=True))
        name = f"{report['table_name']}_{column}"

        child_report = new_report(name, report["source_file"])
        child_report.update(rows_before=len(child), rows_after=len(child),
                            columns_before=2, columns_after=2,
                            column_types={key: "integer", value: "text"},
                            list_of={"table": report["table_name"], "column": column,
                                     "key": key, "value": value})
        add_action(child_report, None, "split_list_column",
                   f"One row per value of {report['table_name']}.{column}; "
                   f"join on {key}.", len(child))
        example = next(flag["example"] for flag in report["flags"]
                       if flag["column"] == column and flag["issue"] == "list_column")
        add_action(report, column, "split_list_column",
                   f"Holds several values per row (e.g. {example!r}); "
                   f"table {name} has one row per value, joined on {key}.")

        extra.append({"table_name": name, "cleaned": child, "raw": child.copy(),
                      "report": child_report})

    report["columns_after"] = len(frame.columns)

    return frame, extra


# Options that leave the data as the automatic cleaning had it.
DEFAULT_OPTIONS = {"same_meaning": "keep", "numbers_guessed": "keep_numbers",
                   "missing_values": "leave",
                   "outliers": "keep",
                   "similar_categories": "keep", "partly_numeric": "keep_text",
                   "decimal_comma": "keep_text", "dates_guessed": "keep_dates",
                   "list_column": "split"}


def mark_resolved(report, choices):
    """
    Flags the user's choice dealt with get "resolved": True, so the model
    is not warned about a problem that is gone (votes no longer TEXT).
    """

    for flag in report["flags"]:
        default = DEFAULT_OPTIONS.get(flag["issue"])
        choice = chosen(choices, flag["column"], flag["issue"], default)
        flag["resolved"] = default is not None and choice != default
        # The user made a choice for it, even the suggested one ("keep the
        # guessed dates"): no longer open in the Data tab. Warnings to the
        # model follow "resolved" (kept extreme values still skew totals).
        flag["decided"] = (choices or {}).get(flag["column"], {}).get(flag["issue"]) is not None


def load_and_clean(file_path, table_base_name=None, choices=None):
    """
    Read a file and clean every table in it.

    table_base_name names the tables (default: the file name), so an
    uploaded file keeps its original name even when stored elsewhere.

    Returns:
        list of {"table_name", "cleaned", "raw", "report"}
    """

    file_path = Path(file_path)
    tables = read_file(file_path)
    base = table_base_name or file_path.name
    results = []

    for table in tables:
        table_name = make_table_name(base, table["sheet"], len(tables))
        table_choices = (choices or {}).get(table_name)
        cleaned, raw, report = clean_table(table, table_name, Path(base).name, table_choices)
        cleaned, extra = split_list_columns(cleaned, report, table_choices)
        mark_resolved(report, table_choices)
        results.append({"table_name": table_name, "cleaned": cleaned,
                        "raw": raw, "report": report})
        results.extend(extra)

    if not results:
        raise ValueError(f"{file_path.name} contains no tables with data.")

    return results
