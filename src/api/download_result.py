
# ============================================================
# FILE: src/api/download_result.py
# PURPOSE: GET /datasets/{dataset_id}/chats/{chat_id}/turns/{position}/download
#
# Download the full result of one answered question as CSV or Excel.
# Chats only store the first 20 rows, so the turn's SQL is run again
# through the same SQL check and read-only executor as the question,
# with a larger row cap. A schema turn ("what columns are there?")
# downloads its column table, rebuilt from the profile (no SQL).
# Refused, failed and blocked turns have no result: 409.
# ============================================================

import io
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.helpers.chat_memory import load_turns
from src.helpers.pipeline import load_dataset
from src.helpers.schema_answer import schema_answer
from src.helpers.sql_executor import execute_sql
from src.helpers.sql_guardrails import check_sql

from src.api.common import require_chat, require_ready_dataset


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["chats"])

DOWNLOAD_MAX_ROWS = 100_000

MEDIA_TYPES = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


# ============================================================
# 1. HELPERS
# ============================================================

def find_turn(paths, chat_id, position):
    """The stored turn, or 404."""

    require_chat(paths, chat_id)

    for turn in load_turns(paths["chats"], chat_id):
        if turn["position"] == position:
            return turn

    raise HTTPException(status_code=404,
                        detail=f"Chat {chat_id} has no turn {position}.")


def to_file_bytes(frame, file_format):
    buffer = io.BytesIO()

    if file_format == "csv":
        buffer.write(frame.to_csv(index=False).encode("utf-8-sig"))
    else:
        frame.to_excel(buffer, index=False, sheet_name="result", engine="openpyxl")

    buffer.seek(0)

    return buffer


# ============================================================
# 2. ROUTE
# ============================================================

@router.get("/{dataset_id}/chats/{chat_id}/turns/{position}/download",
            summary="Download the full result of an answered question")
def download_result(
    dataset_id: str,
    chat_id: int,
    position: int,
    file_format: str = Query("csv", alias="format", pattern="^(csv|xlsx)$",
                             description="csv or xlsx"),
):
    """
    Runs the turn's SQL again (read-only) and returns every row; a schema
    turn returns its column table.
    """

    paths, _ = require_ready_dataset(dataset_id)
    turn = find_turn(paths, chat_id, position)
    dataset = load_dataset(paths)
    allowed = dataset["allowed_tables"]

    if turn["status"] == "schema":
        frame = schema_answer(dataset["profile"], allowed)["frame"]
        result = {"frame": frame, "row_count": len(frame), "truncated": False}

    elif turn["status"] not in {"answered", "empty"} or not turn["sql"]:
        raise HTTPException(status_code=409,
                            detail=f"Turn {position} has no result ({turn['status']}).")

    else:
        # Same checks as a question: static check, then the locked-down executor.
        sql_check = check_sql(turn["sql"], allowed)

        if not sql_check["ok"]:
            raise HTTPException(status_code=409, detail=sql_check["reason"])

        result = execute_sql(paths["database"], turn["sql"], allowed,
                             max_rows=DOWNLOAD_MAX_ROWS)

        if not result["ok"]:
            raise HTTPException(status_code=500, detail=result["error"])

    file_name = f"{dataset_id}_chat{chat_id}_turn{position}.{file_format}"
    headers = {"Content-Disposition": f'attachment; filename="{file_name}"'}

    if result["truncated"]:
        headers["X-Result-Truncated"] = f"first {DOWNLOAD_MAX_ROWS} rows"

    logger.info("Download %s (%d rows)", file_name, result["row_count"])

    return StreamingResponse(to_file_bytes(result["frame"], file_format),
                             media_type=MEDIA_TYPES[file_format], headers=headers)
