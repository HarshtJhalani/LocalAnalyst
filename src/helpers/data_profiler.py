
# ============================================================
# FILE: src/helpers/data_profiler.py
# PURPOSE: Profile cleaned tables (EDA) and build the schema text
#          the model sees
#          (pipeline stage 4; 02_sqlite_profiling.ipynb, steps 5-6)
#
#   profile_dataset()     every table -> ranges, missing values, top
#                         categories, correlations, cleaning flags
#   build_schema_text()   compact schema: types, exact category
#                         spellings, ranges, units, warnings
# ============================================================

import logging
import math

import pandas as pd

from config import (
    CATEGORY_MAX_UNIQUE_RATIO,
    CORRELATION_MIN_ABS,
    CORRELATION_MIN_ROWS,
    SCHEMA_ALL_VALUES_LIMIT,
    SCHEMA_SAMPLE_VALUES,
    SQLITE_TYPES,
    TOP_VALUES,
)


logger = logging.getLogger(__name__)


# ============================================================
# 1. COLUMN PROFILES
# ============================================================

def to_plain(value):
    """numpy / pandas scalars -> JSON-safe Python values."""

    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value) if value.is_integer() else round(value, 4)

    return value


def profile_numeric(series):
    values = series.dropna().astype(float)

    if values.empty:
        return {}

    return {
        "min": to_plain(values.min()),
        "max": to_plain(values.max()),
        "mean": to_plain(values.mean()),
        "median": to_plain(values.median()),
        "std": to_plain(values.std()) if len(values) > 1 else None,
        "p25": to_plain(values.quantile(0.25)),
        "p75": to_plain(values.quantile(0.75)),
        "sum": to_plain(values.sum()),
    }


def profile_date(series):
    values = series.dropna()

    if values.empty:
        return {}

    return {
        "min": values.min().strftime("%Y-%m-%d"),
        "max": values.max().strftime("%Y-%m-%d"),
        "span_days": int((values.max() - values.min()).days),
    }


def profile_boolean(series):
    values = series.dropna()

    return {"true": int(values.sum()), "false": int((~values).sum())}


def profile_text(series):
    values = series.dropna().astype(str)

    if values.empty:
        return {}

    counts = values.value_counts()

    return {
        "is_category": bool(values.nunique() <= CATEGORY_MAX_UNIQUE_RATIO * len(values)),
        "is_unique": bool(values.nunique() == len(values)),
        "top_values": [[value, int(count)]
                       for value, count in counts.head(TOP_VALUES).items()],
        "all_values": sorted(counts.index)
        if len(counts) <= SCHEMA_ALL_VALUES_LIMIT else None,
        "examples": list(dict.fromkeys(values))[:SCHEMA_SAMPLE_VALUES],
        "avg_length": to_plain(values.str.len().mean()),
    }


def profile_column(series, kind, unit):
    """Profile one column according to its kind."""

    missing = int(series.isna().sum())
    profile = {
        "kind": kind,
        "sqlite_type": SQLITE_TYPES[kind],
        "unit": unit,
        "missing": missing,
        "missing_pct": round(100 * missing / len(series), 1) if len(series) else 0,
        "unique": int(series.nunique(dropna=True)),
    }

    if kind in {"integer", "float"}:
        profile.update(profile_numeric(series))
    elif kind in {"date", "datetime"}:
        profile.update(profile_date(series))
    elif kind == "boolean":
        profile.update(profile_boolean(series))
    else:
        profile.update(profile_text(series))

    return profile


# ============================================================
# 2. TABLE AND DATASET PROFILES
# ============================================================

def find_correlations(frame, column_types):
    """Pairs of numeric columns with |Pearson r| >= CORRELATION_MIN_ABS."""

    numeric = [column for column, kind in column_types.items()
               if kind in {"integer", "float"}]

    if len(numeric) < 2:
        return []

    matrix = frame[numeric].astype(float).corr(min_periods=CORRELATION_MIN_ROWS)
    pairs = []

    for position, first in enumerate(numeric):
        for second in numeric[position + 1:]:
            value = matrix.loc[first, second]

            if pd.notna(value) and abs(value) >= CORRELATION_MIN_ABS:
                pairs.append({"columns": [first, second], "r": round(float(value), 3)})

    return sorted(pairs, key=lambda pair: abs(pair["r"]), reverse=True)


def profile_table(table, raw_table_name):
    """Full profile for one cleaned table."""

    frame = table["cleaned"]
    report = table["report"]
    column_types = report["column_types"]
    units = report.get("column_units", {})

    return {
        "table_name": table["table_name"],
        "raw_table": raw_table_name,
        "source_file": report["source_file"],
        "rows": len(frame),
        "columns": {
            column: profile_column(frame[column], kind, units.get(column))
            for column, kind in column_types.items()
        },
        "correlations": find_correlations(frame, column_types),
        "flags": report["flags"],
        "cleaning": {
            "rows_before": report["rows_before"],
            "rows_after": report["rows_after"],
            "actions": report["actions"],
        },
    }


def profile_dataset(tables, written):
    """
    Profile every table of one dataset.

    Args:
        written: write_dataset() result (gives each raw table's name)
    """

    return {
        "tables": {
            table["table_name"]: profile_table(
                table, written[table["table_name"]]["raw_table"])
            for table in tables
        }
    }


def allowed_tables(profile):
    """Cleaned tables a query may read (raw tables are not exposed)."""

    return set(profile["tables"])


# ============================================================
# 3. SCHEMA TEXT FOR THE MODEL
# ============================================================

def format_number(value):
    """No thousands separators: the model must not copy commas into SQL."""

    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def describe_column(name, column, detail="full"):
    """
    One schema line, e.g. city TEXT | values: 'Delhi', 'Pune'

    detail "compact" keeps what a query cannot be written without (the
    date format, 1 / 0 booleans, exact category spellings) and drops
    ranges, examples and missing counts; "names" keeps name and type.
    """

    kind = column["kind"]
    parts = [f"{name} {column['sqlite_type']}"]

    if detail == "names":
        return "  " + parts[0]

    full = detail == "full"

    if kind in {"integer", "float"}:
        if column["unit"] == "percent":
            parts.append("percent (10 means 10%)")
        elif column["unit"] == "currency" and full:
            parts.append("currency amount")

        if full:
            parts.append(f"range {format_number(column.get('min'))} to "
                         f"{format_number(column.get('max'))}")

    elif kind == "date":
        parts.append(f"date 'YYYY-MM-DD', {column.get('min')} to {column.get('max')}"
                     if full else "date 'YYYY-MM-DD'")

    elif kind == "datetime":
        parts.append(f"datetime 'YYYY-MM-DD HH:MM:SS', {column.get('min')} "
                     f"to {column.get('max')}"
                     if full else "datetime 'YYYY-MM-DD HH:MM:SS'")

    elif kind == "boolean":
        parts.append("boolean stored as 1 = true, 0 = false" if full
                     else "boolean 1 / 0")

    else:
        # Unique and never missing: a key the model can join or count on.
        if full and column.get("is_unique") and not column["missing"]:
            parts.append("unique per row")

        if column.get("all_values"):
            parts.append("values: " + ", ".join(repr(v) for v in column["all_values"]))
        elif full and column.get("examples"):
            parts.append("e.g. " + ", ".join(repr(v) for v in column["examples"]))

    if full and column["missing"]:
        parts.append(f"{column['missing']} missing")

    return "  " + " | ".join(parts)


def describe_flag(flag):
    """Cleaning flags rewritten as warnings for the model."""

    column, issue = flag["column"], flag["issue"]
    detail = flag["detail"].replace("; kept.", "").replace("; not merged.", "")

    if issue == "missing_values":
        return None                      # already on the column line
    if issue == "duplicate_rows":
        return detail                    # about the table, not one column
    if issue == "decimal_comma":
        return f"{column}: numbers stored as TEXT with a decimal comma"
    if issue in {"outliers", "similar_categories"}:
        return f"{column}: {detail}"
    if issue == "ambiguous_dates":
        return f"{column}: dates are TEXT in an unknown day/month order"
    if issue == "partly_numeric":
        return f"{column}: stored as TEXT because some values are not numbers"
    if issue == "identifier_kept_as_text":
        return f"{column}: identifier with leading zeros, compare as TEXT"

    return f"{column}: {detail}"


def build_schema_text(profile, detail="full"):
    """
    Compact schema of every cleaned table in one dataset.

    detail: "full" (stored as schema.txt), or the shorter "compact" /
    "names" used when the full text does not fit the model's context
    (see describe_column; both leave out the warnings).
    """

    blocks = []

    for table_name, table in profile["tables"].items():
        lines = [f"TABLE {table_name} ({table['rows']} rows)"]
        lines += [describe_column(name, column, detail)
                  for name, column in table["columns"].items()]

        warnings = ([text for text in map(describe_flag, table["flags"]) if text]
                    if detail == "full" else [])

        if warnings:
            lines.append("  Warnings:")
            lines += [f"  - {warning}" for warning in warnings]

        blocks.append("\n".join(lines))

    blocks.append(
        "SQL notes: SQLite dialect. Text comparisons are case-sensitive. "
        "For median / standard deviation / percentile use "
        "median_of(json_group_array(col)), stdev_of(json_group_array(col)), "
        "percentile_of(json_group_array(col), 90)."
    )

    return "\n\n".join(blocks)
