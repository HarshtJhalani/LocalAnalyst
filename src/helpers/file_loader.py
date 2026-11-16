
# ============================================================
# FILE: src/helpers/file_loader.py
# PURPOSE: Read an uploaded file into text grids
#          (pipeline stage 1; 01_loading_cleaning.ipynb, step 2)
#
# Every reader returns a list of tables. CSV and Excel tables come back
# as a grid (no header yet, every cell text or None) so the header row
# can be detected; JSON and Parquet put their column names in row 0.
#
#   read_file()   validate a file and read it into grid tables
# ============================================================

import csv
import io
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    CSV_DELIMITERS,
    MAX_FILE_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    TEXT_ENCODINGS,
)


logger = logging.getLogger(__name__)


# ============================================================
# 1. VALIDATION AND CELL TEXT
# ============================================================

def validate_file(file_path):
    """Check the file exists, has a supported type and is not too big."""

    # Step 1: the file must exist.
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Step 2: supported extension only.
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {extension!r}. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Step 3: size limit, and not empty.
    size_mb = file_path.stat().st_size / (1024 * 1024)

    if size_mb == 0:
        raise ValueError(f"File is empty: {file_path.name}")

    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File is {size_mb:.1f} MB; the limit is {MAX_FILE_SIZE_MB} MB."
        )

    return file_path


def cell_to_text(value):
    """
    Turn one cell into stripped text, or None when it is empty.

    Excel dates become ISO text and whole floats lose their ".0", so
    every source is cleaned by the same text rules.
    """

    # Missing values from Parquet / JSON: NaT in a date column, NA in a
    # nullable integer or boolean column.
    if value is None or value is pd.NaT or value is pd.NA:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, (pd.Timestamp, datetime)):
        if (value.hour, value.minute, value.second) == (0, 0, 0):
            return value.strftime("%Y-%m-%d")
        return value.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)

    text = str(value).strip()

    return text if text else None


def rows_to_grid(rows):
    """Pad ragged rows into a rectangular DataFrame of text cells."""

    width = max((len(row) for row in rows), default=0)
    padded = [
        [cell_to_text(cell) for cell in row] + [None] * (width - len(row))
        for row in rows
    ]

    return pd.DataFrame(padded, dtype=object)


def frame_to_grid(frame):
    """Put a frame's column names back as row 0 of a text grid."""

    rows = [list(frame.columns)] + frame.astype(object).values.tolist()

    return rows_to_grid(rows)


# ============================================================
# 2. TEXT FILES (CSV / TSV)
# ============================================================

def read_text(file_path):
    """Decode a text file with the first encoding that works."""

    raw_bytes = Path(file_path).read_bytes()

    for encoding in TEXT_ENCODINGS:
        try:
            return raw_bytes.decode(encoding), encoding

        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode {Path(file_path).name} as text.")


def detect_delimiter(text, file_path):
    """
    Pick the delimiter that splits the most lines into the same number
    of fields. Title lines above the header do not confuse it.
    """

    if Path(file_path).suffix.lower() == ".tsv":
        return "\t"

    lines = [line for line in text.splitlines() if line.strip()][:50]
    best_delimiter, best_score = ",", 0

    for delimiter in CSV_DELIMITERS:
        counts = [line.count(delimiter) for line in lines]
        non_zero = [count for count in counts if count > 0]

        if not non_zero:
            continue

        # Lines sharing the most common field count.
        common_count = max(set(non_zero), key=non_zero.count)
        score = non_zero.count(common_count)

        if score > best_score:
            best_delimiter, best_score = delimiter, score

    return best_delimiter


def read_csv_file(file_path):
    """CSV / TSV -> one grid table."""

    text, encoding = read_text(file_path)
    delimiter = detect_delimiter(text, file_path)
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))

    logger.info("Read %s | encoding=%s | delimiter=%r | lines=%d",
                Path(file_path).name, encoding, delimiter, len(rows))

    return [{
        "sheet": None,
        "grid": rows_to_grid(rows),
        "header_known": False,
        "source_info": {"encoding": encoding, "delimiter": delimiter},
    }]


# ============================================================
# 3. EXCEL, JSON AND PARQUET
# ============================================================

def read_excel_file(file_path):
    """Excel -> one grid table per non-empty sheet."""

    try:
        sheets = pd.read_excel(file_path, sheet_name=None, header=None,
                               dtype=object, engine="openpyxl")

    except Exception as exc:
        raise ValueError(f"{Path(file_path).name} is not a readable Excel file: {exc}") from exc

    tables = []

    for sheet_name, sheet in sheets.items():
        # rows_to_grid keeps empty cells as None (not NaN).
        grid = rows_to_grid(sheet.astype(object).values.tolist())

        if grid.isna().all().all():
            logger.info("Skipping empty sheet %r", sheet_name)
            continue

        tables.append({
            "sheet": str(sheet_name),
            "grid": grid,
            "header_known": False,
            "source_info": {"sheet": str(sheet_name)},
        })

    logger.info("Read %s | sheets=%d", Path(file_path).name, len(tables))

    return tables


def read_json_file(file_path):
    """
    JSON / JSON Lines -> one table.

    Accepts a list of records, a dict holding one list of records
    (e.g. {"customers": [...]}) or one record per line. Nested objects
    are flattened into columns such as address_city.
    """

    text, _ = read_text(file_path)
    name = Path(file_path).name

    # Step 1: whole document first, then JSON Lines.
    try:
        data = json.loads(text)

    except json.JSONDecodeError:
        try:
            data = [json.loads(line) for line in text.splitlines()
                    if line.strip()]

        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not valid JSON: {exc}") from exc

    # Step 2: find the list of records.
    if isinstance(data, dict):
        record_lists = [
            value for value in data.values()
            if isinstance(value, list) and value
            and all(isinstance(item, dict) for item in value)
        ]
        data = record_lists[0] if len(record_lists) == 1 else [data]

    if not isinstance(data, list) or not data:
        raise ValueError(f"{name} has no records to load.")

    # Step 3: flatten nested objects.
    frame = pd.json_normalize(data, sep="_")

    logger.info("Read %s | records=%d", name, len(frame))

    return [{"sheet": None, "grid": frame_to_grid(frame),
             "header_known": True, "source_info": {}}]


def read_parquet_file(file_path):
    """Parquet -> one table (types are re-inferred like any other file)."""

    try:
        frame = pd.read_parquet(file_path)

    except Exception as exc:
        raise ValueError(f"{Path(file_path).name} is not a readable Parquet file: {exc}") from exc

    logger.info("Read %s | rows=%d", Path(file_path).name, len(frame))

    return [{"sheet": None, "grid": frame_to_grid(frame),
             "header_known": True, "source_info": {}}]


# ============================================================
# 4. ENTRY POINT
# ============================================================

def read_file(file_path):
    """Validate a file and read it into a list of grid tables."""

    file_path = validate_file(file_path)
    extension = file_path.suffix.lower()

    if extension in {".csv", ".tsv"}:
        return read_csv_file(file_path)

    if extension in {".xlsx", ".xlsm"}:
        return read_excel_file(file_path)

    if extension in {".json", ".jsonl"}:
        return read_json_file(file_path)

    return read_parquet_file(file_path)
