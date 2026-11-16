
# ============================================================
# FILE: src/helpers/logging_config.py
# PURPOSE: Configure application-wide logging and write traces
#
#   setup_logging()   console + file logging on the root logger;
#                     called once at application startup
#   write_trace()     append one JSON line to a dataset's trace log
# ============================================================

import json
import logging
from pathlib import Path

from config import APP_LOG


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================
# 1. APPLICATION LOGGING
# ============================================================

def setup_logging(log_level=logging.INFO, log_file=APP_LOG):
    """
    Configure console and file logging on the root logger.

    Calling it again replaces the existing handlers.

    Returns:
        Path: The log file in use.
    """

    log_file = Path(log_file)

    # Step 1: make sure the log folder exists.
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        raise OSError(
            f"Could not create log directory {log_file.parent}: {exc}"
        ) from exc

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Step 2: remove handlers from a previous call.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Step 3: console and file handlers.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")

    except OSError as exc:
        raise OSError(f"Could not open log file {log_file}: {exc}") from exc

    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # The Ollama client logs every HTTP request at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info("Logging initialized. Log file: %s", log_file)

    return log_file


# ============================================================
# 2. TRACES
# ============================================================

def write_trace(trace_path, trace):
    """Append one question's trace as a JSON line."""

    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with trace_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
