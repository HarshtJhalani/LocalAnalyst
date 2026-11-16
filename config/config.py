
# ============================================================
# FILE: config/config.py
# PURPOSE: Central configuration for the Text-to-SQL Data Analyst
#
#   1. Project paths
#   2. Per-dataset workspace paths (datasets/<dataset_id>/...)
#   3. Ollama and model profiles
#   4. Upload and cleaning settings
#   5. Profiling and schema text
#   6. Question, SQL and execution limits
#   7. Chart and summary settings
#   8. Chat memory
#
# Everything here is plain constants and functions; no classes.
# ============================================================

import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# config/config.py -> project root is one level up.
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Sample files, dev questions and the held-out evaluation sets.
INPUT_DIR = DATA_DIR / "input_data"
EVAL_DIR = INPUT_DIR / "eval"
DEV_QUESTIONS_PATH = INPUT_DIR / "dev_questions.json"

# Notebook and evaluation outputs.
OUTPUT_DIR = DATA_DIR / "output_data"
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation"

# Environment file (OLLAMA_URL, MODEL_PROFILE, DATASETS_DIR). Loaded
# before any os.getenv() below, or the values in .env would be ignored.
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

# API dataset workspaces, see get_dataset_paths().
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", BASE_DIR / "datasets"))

# Application log
APP_LOG = DATA_DIR / "logs" / "app.log"

# Server-generated dataset IDs, e.g. DS_20261114_183012_A3F9
DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# ============================================================
# 2. PER-DATASET WORKSPACE PATHS
# ============================================================

def generate_dataset_id():
    """New unique dataset ID: DS_<date>_<time>_<4 hex>."""

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"DS_{stamp}_{secrets.token_hex(2).upper()}"


def validate_dataset_id(dataset_id):
    """Reject IDs that could escape the datasets folder."""

    if not DATASET_ID_PATTERN.fullmatch(str(dataset_id)):
        raise ValueError(f"Invalid dataset ID: {dataset_id!r}")

    return str(dataset_id)


def get_dataset_paths(dataset_id, datasets_dir=None):
    """
    Every path of one dataset workspace.

    datasets/<dataset_id>/
        input/                  the uploaded file, as received
        dataset.db              cleaned + raw tables (read-only to queries)
        cleaning_report.json    one report per table
        profile.json            EDA profile per table
        schema.txt              compact schema the model sees
        chats.db                chats and turns
        logs/traces.jsonl       one line per answered question
        status.json             ingestion stage and counts
    """

    dataset_id = validate_dataset_id(dataset_id)
    root = Path(datasets_dir or DATASETS_DIR) / dataset_id

    return {
        "dataset_id": dataset_id,
        "root": root,
        "input_dir": root / "input",
        "database": root / "dataset.db",
        "cleaning_report": root / "cleaning_report.json",
        "profile": root / "profile.json",
        "schema": root / "schema.txt",
        "chats": root / "chats.db",
        "logs_dir": root / "logs",
        "trace_log": root / "logs" / "traces.jsonl",
        "status": root / "status.json",
    }


def create_dataset_directories(paths):
    """Create the workspace folders of one dataset."""

    for key in ("root", "input_dir", "logs_dir"):
        try:
            paths[key].mkdir(parents=True, exist_ok=True)

        except OSError as exc:
            raise OSError(f"Could not create {paths[key]}: {exc}") from exc

    return paths


# ============================================================
# 3. OLLAMA AND MODEL PROFILES
# ============================================================

# Inside Docker: http://host.docker.internal:11434
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# auto: detect; gpu / cpu: force a profile.
MODEL_PROFILE = os.getenv("MODEL_PROFILE", "auto").strip().lower()

# Same model file runs on GPU or CPU; the profile picks the size.
MODEL_PROFILES = {
    "gpu": {
        "model": "qwen2.5:7b-instruct",
        "num_ctx": 8192,
        "max_retries": 3,
        "request_timeout": 120,
    },
    "cpu": {
        "model": "qwen2.5:3b-instruct",
        "num_ctx": 4096,
        "max_retries": 2,
        "request_timeout": 300,
    },
}

# A model "fits" the GPU when at least this share is in GPU memory.
GPU_FIT_SHARE = 0.99

LLM_TEMPERATURE = 0
KEEP_ALIVE = "30m"
CONNECT_TIMEOUT = 10

# Loading a model into memory (cold disk, or the 7B on a CPU) takes far
# longer than a connection check.
MODEL_LOAD_TIMEOUT = 300

# Cap on generated tokens: a runaway answer becomes invalid JSON (and is
# retried) instead of running into the request timeout.
MAX_OUTPUT_TOKENS = 512

# Prompt size guard. Ollama silently drops the start of a prompt that
# does not fit num_ctx (the rules and examples), so the prompt is
# estimated first and the schema text shortened when needed.
# Measured on the SQL prompt: about 3 characters per token.
PROMPT_CHARS_PER_TOKEN = 3
RETRY_TOKEN_RESERVE = 400         # room for the retry messages


# ============================================================
# 4. UPLOAD AND CLEANING SETTINGS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".csv", ".tsv", ".xlsx", ".xlsm", ".json", ".jsonl", ".parquet"
}
MAX_FILE_SIZE_MB = 50
MAX_TABLE_ROWS = 1_000_000
MAX_TABLE_COLUMNS = 200

# Text decoding attempts, in order (latin-1 never fails).
TEXT_ENCODINGS = ["utf-8-sig", "cp1252", "latin-1"]
CSV_DELIMITERS = [",", ";", "\t", "|"]

# Header detection
HEADER_SCAN_ROWS = 20
HEADER_MIN_FILLED_RATIO = 0.8

# A column is converted when at least TYPE_MATCH_THRESHOLD of its
# non-null values parse, or all but one when the rest is at least
# SINGLE_FAILURE_MIN_RATIO; from PARTLY_TYPED_RATIO up it is only flagged.
TYPE_MATCH_THRESHOLD = 0.95
SINGLE_FAILURE_MIN_RATIO = 0.9
PARTLY_TYPED_RATIO = 0.5

# Categories
CATEGORY_MAX_UNIQUE_RATIO = 0.5
SIMILAR_CATEGORY_RATIO = 0.8
SIMILAR_CATEGORY_MAX_VALUES = 100

# Outliers: outside the IQR fences when rare, or a big jump at either end.
OUTLIER_IQR_FACTOR = 3.0
OUTLIER_MAX_SHARE = 0.05
OUTLIER_GAP_FACTOR = 10

NULL_PLACEHOLDERS = {
    "", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--",
    "?", "missing", "#n/a", "not available", "undefined",
}
TRUE_VALUES = {"yes", "y", "true", "t"}
FALSE_VALUES = {"no", "n", "false", "f"}

CURRENCY_SYMBOLS = "$€£₹¥"

DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%Y",
    "%m/%d/%Y", "%d.%m.%Y", "%d/%m/%y", "%m/%d/%y", "%d-%b-%Y",
    "%d %b %Y", "%b %d, %Y", "%d %B %Y", "%B %d, %Y",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M",
]

# Column names that would need quoting in SQL get a "_value" suffix.
SQL_RESERVED_WORDS = {
    "all", "and", "any", "as", "asc", "between", "by", "case", "cast",
    "check", "column", "create", "default", "delete", "desc", "distinct",
    "drop", "else", "end", "exists", "from", "group", "having", "in",
    "index", "insert", "into", "is", "join", "key", "like", "limit",
    "not", "null", "offset", "on", "or", "order", "primary",
    "references", "select", "set", "table", "then", "to", "union",
    "update", "values", "when", "where",
    # Words the SQL check refuses anywhere in a query (sql_guardrails.py);
    # a column with one of these names could never be queried.
    "alter", "analyze", "attach", "begin", "commit", "detach",
    "load_extension", "pragma", "reindex", "release", "rollback",
    "savepoint", "truncate", "vacuum",
}


# ============================================================
# 5. PROFILING AND SCHEMA TEXT
# ============================================================

RAW_TABLE_SUFFIX = "_raw"
INSERT_BATCH_SIZE = 5000

# SQLite column type per kind found by the cleaner.
SQLITE_TYPES = {
    "integer": "INTEGER",
    "float": "REAL",
    "boolean": "INTEGER",
    "date": "TEXT",
    "datetime": "TEXT",
    "text": "TEXT",
}

TOP_VALUES = 5
CORRELATION_MIN_ABS = 0.5
CORRELATION_MIN_ROWS = 5

SCHEMA_ALL_VALUES_LIMIT = 15      # list every category up to this many
SCHEMA_SAMPLE_VALUES = 3          # examples for other text columns


# ============================================================
# 6. QUESTION, SQL AND EXECUTION LIMITS
# ============================================================

MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 500

QUERY_TIMEOUT_SECONDS = 10
MAX_RESULT_ROWS = 1000
PROGRESS_CHECK_STEPS = 10_000     # SQLite VM steps between time checks


# ============================================================
# 7. CHART AND SUMMARY SETTINGS
# ============================================================

MAX_BARS = 25
MAX_SERIES = 8
MAX_SMALL_MULTIPLES = 3
MIN_SCATTER_ROWS = 5
MIN_HISTOGRAM_ROWS = 10
BAR_SLOT_PX = 36                  # bar ~22 px thick with BAR_GAP 0.4
BAR_GAP = 0.4
CHART_WIDTH = 760

SUMMARY_MAX_ROWS = 20             # result rows shown to the model


# ============================================================
# 8. CHAT MEMORY
# ============================================================

MEMORY_TURNS = 3                  # answered turns given to the model
HISTORY_PREVIEW_ROWS = 5          # result rows shown per earlier turn
STORED_PREVIEW_ROWS = 20          # result rows saved with each turn
CHAT_TITLE_LENGTH = 60
