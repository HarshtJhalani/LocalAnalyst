# ============================================================
# FILE: src/api/download_table.py
# PURPOSE: GET /datasets/{dataset_id}/tables/{table_name}/download
#
# Download a whole cleaned table (after the automatic cleaning and the
# user's review choices) as CSV or Excel. Only cleaned tables can be
# downloaded; the *_raw copies are not exposed, as for questions.
# Read through a read-only connection; no SQL from the user.
# ============================================================

import logging
import sqlite3

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.helpers.pipeline import load_dataset

from src.api.common import require_ready_dataset
from src.api.download_result import MEDIA_TYPES, to_file_bytes


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/{dataset_id}/tables/{table_name}/download",
            summary="Download a cleaned table as CSV or Excel")
def download_table(
    dataset_id: str,
    table_name: str,
    file_format: str = Query("csv", alias="format", pattern="^(csv|xlsx)$",
                             description="csv or xlsx"),
):
    """Every row of one cleaned table; 404 for a table the dataset does not have."""

    paths, _ = require_ready_dataset(dataset_id)
    allowed = load_dataset(paths)["allowed_tables"]

    # Only names from the profile reach the query, so the quoting below
    # never sees user text it has not already matched exactly.
    if table_name not in allowed:
        raise HTTPException(status_code=404,
                            detail=f"No cleaned table {table_name!r} in this dataset.")

    connection = sqlite3.connect(f"file:{paths['database']}?mode=ro", uri=True)

    try:
        quoted = '"' + table_name.replace('"', '""') + '"'
        frame = pd.read_sql_query(f"SELECT * FROM {quoted}", connection)

    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        logger.error("Could not read %s.%s: %s", dataset_id, table_name, exc)
        raise HTTPException(status_code=500, detail="The table could not be read.")

    finally:
        connection.close()

    file_name = f"{table_name}_cleaned.{file_format}"
    logger.info("Download %s (%d rows)", file_name, len(frame))

    return StreamingResponse(
        to_file_bytes(frame, file_format), media_type=MEDIA_TYPES[file_format],
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'})
