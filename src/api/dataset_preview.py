# ============================================================
# FILE: src/api/dataset_preview.py
# PURPOSE: GET /datasets/{dataset_id}/preview
#
# The first rows of every cleaned table, a short description and a few
# starter questions. No model call, so it returns at once after upload.
# ============================================================

import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Query, Request

from src.helpers.dataset_preview import (
    dataset_preview,
    dataset_suggestions,
    name_rows,
    preview_rows,
)
from src.helpers.pipeline import load_dataset, read_json, write_json

from src.api.common import get_existing_dataset


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


def row_noun(request, paths, profile):
    """
    What one row of the uploaded table is, named once by the model and
    stored in row_noun.json; "rows" while no model is loaded.
    """

    stored = paths["root"] / "row_noun.json"

    if stored.is_file():
        return read_json(stored)["noun"]

    settings = getattr(request.app.state, "model_settings", None)

    if settings is None:
        return "rows"

    table = next(iter(profile["tables"]))
    first = preview_rows(paths["database"], table, 3)
    noun = name_rows(settings, table, first["columns"], first["rows"])
    write_json(stored, {"noun": noun})

    return noun


@router.get("/{dataset_id}/preview", summary="First rows and a short description of a dataset")
def preview_dataset(
    request: Request,
    dataset_id: str,
    rows: int = Query(5, ge=1, le=50, description="How many rows per table."),
):
    """409 while the dataset is not ready. Starter questions are checked
    against the data before they are offered."""

    paths, status = get_existing_dataset(dataset_id)

    if status.get("status") != "completed":
        raise HTTPException(status_code=409, detail="This dataset is not ready yet.")

    try:
        profile = load_dataset(paths)["profile"]
        tables = dataset_preview(paths["database"], profile, rows,
                                 row_noun(request, paths, profile))

    except (FileNotFoundError, ValueError, OSError, sqlite3.Error) as exc:
        logger.error("Could not preview dataset %s: %s", dataset_id, exc)
        raise HTTPException(status_code=500, detail="Dataset files are unreadable.")

    return {"dataset_id": dataset_id, "tables": tables,
            "suggestions": dataset_suggestions(tables)}
