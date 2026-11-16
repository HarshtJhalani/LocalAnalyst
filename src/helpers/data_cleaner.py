
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
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

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
LEADING_ZERO_PATTERN = re.compile(r"^[+-]?0\d")

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


def add_flag(report, column, issue, detail):
    """Record a problem that was left unchanged."""

    report["flags"].append({"column": column, "issue": issue, "detail": detail})


# ============================================================
# 4. CLEANING VALUES
# ============================================================

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

    # Step 2: parse every value.
    parsed = values.map(parse_number)
    ok = parsed.notna()
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
    }


def swap_day_month(date_format):
    """%d/%m/%Y <-> %m/%d/%Y"""

    return (date_format.replace("%d", "@").replace("%m", "%d")
                       .replace("@", "%m"))


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
        return None

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


def convert_column(frame, column, report):
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
                     "kept as text.")
            break

        if kind == "partly_numeric":
            add_flag(report, column, "partly_numeric",
                     f"{result['rate']:.0%} of values are numbers; kept as text. "
                     f"Not numbers: {sorted(set(result['failed']))[:5]}")
            break

        if kind == "ambiguous_date":
            add_flag(report, column, "ambiguous_dates",
                     "Dates fit both day-first and month-first "
                     f"({' / '.join(result['formats'])}); kept as text.")
            break

        # A real conversion. Rows that were null stay null (reindex).
        converted = result["values"].reindex(frame.index)

        if kind == "boolean":
            converted = converted.astype("boolean")

        frame[column] = converted

        detail = f"Converted text to {kind}"

        if result.get("format"):
            detail += f" (format {result['format']})"

        if result["unit"]:
            detail += f"; removed {result['unit']} symbols"
            report["column_units"][column] = result["unit"]

        failed = sorted(set(result["failed"]))

        if failed:
            detail += f"; values that did not parse became null: {failed[:5]}"

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
                     f"({missing / len(frame):.0%}).")


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
                     f"{count} extreme value(s), e.g. {examples}; kept.")


def flag_similar_categories(frame, report):
    for column, kind in report["column_types"].items():
        if kind != "text" or not is_category_column(frame[column]):
            continue

        values = sorted(frame[column].dropna().unique())

        if len(values) > SIMILAR_CATEGORY_MAX_VALUES:
            continue

        pairs = []

        for position, first in enumerate(values):
            for second in values[position + 1:]:
                # "Product 1" vs "Product 2" differ only in digits.
                if re.sub(r"\d", "", first) == re.sub(r"\d", "", second):
                    continue

                ratio = difflib.SequenceMatcher(
                    None, first.casefold(), second.casefold()).ratio()

                if ratio >= SIMILAR_CATEGORY_RATIO:
                    pairs.append((first, second))

        if pairs:
            add_flag(report, column, "similar_categories",
                     f"Possibly the same category: {pairs[:5]}; not merged.")


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


def clean_table(table, table_name, source_file):
    """
    Clean one grid table.

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
    frame = trim_whitespace(frame, report)
    frame = replace_null_placeholders(frame, report)
    frame = drop_empty_rows_and_columns(frame, report)
    frame = drop_duplicate_rows(frame, report)

    if frame.empty:
        raise ValueError(f"{table_name}: no data rows left after cleaning.")

    # Step 4: types, then categories.
    for column in frame.columns:
        frame = convert_column(frame, column, report)

    frame = merge_category_variants(frame, report)

    # Step 5: flags only.
    flag_missing_values(frame, report)
    flag_outliers(frame, report)
    flag_similar_categories(frame, report)
    flag_constant_columns(frame, report)

    report["rows_after"] = len(frame)
    report["columns_after"] = len(frame.columns)

    logger.info("Cleaned %s | rows %d -> %d | columns %d -> %d | "
                "%d action(s), %d flag(s)",
                table_name, report["rows_before"], report["rows_after"],
                report["columns_before"], report["columns_after"],
                len(report["actions"]), len(report["flags"]))

    return frame, raw, report


def load_and_clean(file_path, table_base_name=None):
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
        cleaned, raw, report = clean_table(table, table_name, Path(base).name)
        results.append({"table_name": table_name, "cleaned": cleaned,
                        "raw": raw, "report": report})

    if not results:
        raise ValueError(f"{file_path.name} contains no tables with data.")

    return results
