
# ============================================================
# FILE: src/api/list_datasets.py
# PURPOSE: GET /datasets
#
# List every uploaded dataset with its file name, ID, status and tables,
# newest first. Folders whose ingestion never started, or whose
# status.json is unreadable, are skipped.
# ============================================================

import logging

from fastapi import APIRouter, HTTPException

from config import DATASETS_DIR, get_dataset_paths
from src.helpers.pipeline import read_status


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


# ============================================================
# 1. HELPERS
# ============================================================

def summarize_dataset(folder):
    """
    The list entry for one dataset folder.

    Returns:
        dict | None: None when the folder is not a usable dataset.
    """

    try:
        paths = get_dataset_paths(folder.name, datasets_dir=DATASETS_DIR)

    except ValueError:
        logger.debug("Ignoring non-dataset folder: %s", folder.name)
        return None

    try:
        status = read_status(paths)

    except (ValueError, OSError) as exc:
        logger.warning("Skipping dataset %s: unreadable status.json (%s)",
                       folder.name, exc)
        return None

    if status is None:
        return None

    return {
        "dataset_id": folder.name,
        "file_name": status.get("file_name"),
        "status": status.get("status"),
        "tables": [{"table_name": table["table_name"], "rows": table["rows"]}
                   for table in status.get("tables", [])],
        "created_at": status.get("created_at"),
        "error": status.get("error"),
    }


# ============================================================
# 2. ROUTE
# ============================================================

@router.get("", summary="List all uploaded datasets")
def list_datasets():
    """Every dataset with its file name, ID, status and tables, newest first."""

    if not DATASETS_DIR.is_dir():
        return []

    try:
        folders = sorted((path for path in DATASETS_DIR.iterdir() if path.is_dir()),
                         reverse=True)

    except OSError as exc:
        logger.exception("Could not list datasets folder %s", DATASETS_DIR)
        raise HTTPException(status_code=500,
                            detail=f"Could not read the datasets folder: {exc}")

    datasets = [entry for entry in map(summarize_dataset, folders) if entry]

    logger.info("Listed %d dataset(s).", len(datasets))

    return datasets
