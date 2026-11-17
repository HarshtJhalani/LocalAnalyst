
# ============================================================
# FILE: src/api/upload_dataset.py
# PURPOSE: POST /datasets
#
# Upload one data file (CSV, TSV, Excel, JSON, Parquet). It is cleaned
# automatically, written to SQLite and profiled; the response lists
# every table with what the cleaning changed and flagged.
#
#   A. validate the file name and type   -> 400
#   B. create datasets/<dataset_id>/     save the file (size cap -> 413)
#   C. clean, store and profile          -> 400 bad file, 500 failures
#   D. return the dataset summary
# ============================================================

import logging
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import (
    DATASETS_DIR,
    MAX_FILE_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    create_dataset_directories,
    generate_dataset_id,
    get_dataset_paths,
)
from src.helpers.input_guardrails import safe_filename
from src.helpers.pipeline import ingest_dataset, read_json

from src.api.common import save_upload


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


# ============================================================
# 1. HELPERS
# ============================================================

def discard_workspace(paths, reason):
    """Remove a dataset folder that never became usable."""

    shutil.rmtree(paths["root"], ignore_errors=True)
    logger.warning("Discarded workspace for %s: %s", paths["dataset_id"], reason)


def cleaning_overview(paths):
    """What the cleaning changed and flagged, per table."""

    overview = []

    for report in read_json(paths["cleaning_report"]):
        overview.append({
            "table_name": report["table_name"],
            "rows_before": report["rows_before"],
            "rows_after": report["rows_after"],
            "applied": [action["detail"] if action["column"] is None
                        else f"[{action['column']}] {action['detail']}"
                        for action in report["actions"]],
            "flagged": [f"[{flag['column']}] {flag['detail']}"
                        for flag in report["flags"]],
        })

    return overview


# ============================================================
# 2. ROUTE
# ============================================================

@router.post("", status_code=201, summary="Upload a data file and prepare it for questions")
def upload_dataset(
    file: UploadFile = File(
        ..., description="CSV, TSV, Excel (.xlsx/.xlsm), JSON / JSON Lines or Parquet."
    ),
):
    """
    Clean the file automatically (header row, column names, types, null
    placeholders, duplicates, category spellings), store every table in
    SQLite and profile it. Returns the dataset ID used by every other
    route.
    """

    # ----------------------------------------------------
    # A. Validate the file name and type
    # ----------------------------------------------------
    try:
        file_name = safe_filename(file.filename)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    extension = "." + file_name.rsplit(".", 1)[-1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {extension!r}. Supported: "
                   f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # ----------------------------------------------------
    # B. Create the workspace and store the file
    # ----------------------------------------------------
    paths = get_dataset_paths(generate_dataset_id(), datasets_dir=DATASETS_DIR)

    try:
        create_dataset_directories(paths)
        upload_path = paths["input_dir"] / file_name
        save_upload(file, upload_path, MAX_FILE_SIZE_MB * 1024 * 1024)

    except ValueError as exc:
        discard_workspace(paths, str(exc))
        raise HTTPException(status_code=413, detail=str(exc))

    except OSError as exc:
        discard_workspace(paths, str(exc))
        logger.exception("Could not store upload for %s", paths["dataset_id"])
        raise HTTPException(status_code=500, detail="Could not store the uploaded file.")

    # ----------------------------------------------------
    # C. Clean, store and profile
    # ----------------------------------------------------
    try:
        summary = ingest_dataset(paths, upload_path, file_name)

    except (ValueError, FileNotFoundError) as exc:
        # The file itself is the problem (empty, unreadable, no rows ...).
        discard_workspace(paths, str(exc))
        raise HTTPException(status_code=400, detail=f"Could not use {file_name}: {exc}")

    except Exception as exc:
        # Kept with status "failed" so the failure can be inspected.
        logger.exception("Ingestion failed for %s", paths["dataset_id"])
        raise HTTPException(
            status_code=500,
            detail={"dataset_id": paths["dataset_id"],
                    "error": f"{type(exc).__name__}: {exc}"},
        )

    # ----------------------------------------------------
    # D. Respond with the summary
    # ----------------------------------------------------
    return {
        "dataset_id": summary["dataset_id"],
        "file_name": summary["file_name"],
        "tables": summary["tables"],
        "cleaning": cleaning_overview(paths),
        "schema_text": summary["schema_text"],
    }
