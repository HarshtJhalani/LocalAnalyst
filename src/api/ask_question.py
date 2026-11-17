
# ============================================================
# FILE: src/api/ask_question.py
# PURPOSE: POST /datasets/{dataset_id}/ask
#
# Ask a question in plain English. Without chat_id a new chat starts;
# with chat_id the question can build on earlier ones ("only paid
# orders", "how much did it make?").
#
#   guardrails -> follow-up rewrite -> SQL -> SQL check -> read-only
#   execution with self-correction -> stored turn -> chart, checked
#   summary, caveats
#
# Questions about the structure ("what columns are there?", "describe
# the data") get status "schema": the column table from the profile, a
# summary sentence and suggested questions, with no SQL and no model
# call for SQL.
#
# Messages that are not data questions get status "clarify" with a reply
# in "message" and no SQL: vague ("what", "ok") or feedback without a
# change ("that's wrong"). "Make it a pie chart" redraws the previous
# answer: its stored SQL is re-run read-only and drawn as asked. A
# clarify reply to the first message of a new chat starts no chat
# (chat_id and position are null).
#
# Status codes: 200 answered / empty / schema / clarify / refused / failed (see "status"),
# 400 blocked by the question guardrail, 404 unknown dataset or chat,
# 409 dataset not ready, 503 model unreachable or too slow.
# ============================================================

import logging

from fastapi import APIRouter, Body, HTTPException, Request

from src.helpers.pipeline import run_question

from src.api.common import (
    figures_payload,
    frame_to_payload,
    get_model_settings,
    require_ready_dataset,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["questions"])


# ============================================================
# 1. HELPERS
# ============================================================

def build_response(dataset_id, result, include_figures):
    """Pipeline result -> JSON-safe response."""

    response = {
        "dataset_id": dataset_id,
        "chat_id": result.get("chat_id"),
        "position": result.get("position"),
        "status": result["status"],
        "question": result["question"],
        "standalone_question": result.get("standalone_question"),
        "uses_history": result.get("uses_history", False),
        "plan": result.get("plan"),
        "sql": result.get("sql"),
        "message": result.get("message"),
        "result": {**frame_to_payload(result.get("frame")),
                   "row_count": result.get("row_count", 0),
                   "truncated": result.get("truncated", False)},
        "chart": result["chart"],
        "summary": result["summary"]["summary"],
        "summary_source": result["summary"]["source"],
        "caveats": result["caveats"],
        "suggestions": result.get("suggestions") or [],
        "attempts": [{key: attempt.get(key)
                      for key in ("attempt", "outcome", "sql", "error", "seconds")}
                     for attempt in result["attempts"]],
        "seconds": result["seconds"],
    }

    if include_figures:
        response["figures"] = figures_payload(result["figures"])

    return response


def model_failed(result):
    """True when the last attempt failed because the model did not answer."""

    attempts = result.get("attempts") or []

    return (result["status"] == "failed" and attempts
            and attempts[-1].get("outcome") == "model_error")


# ============================================================
# 2. ROUTE
# ============================================================

@router.post("/{dataset_id}/ask", summary="Ask a question about a dataset")
def ask_question(
    dataset_id: str,
    request: Request,
    question: str = Body(..., description="The question in plain English.",
                         examples=["What are total sales by region?"]),
    chat_id: int | None = Body(None, description="Continue this chat; omit to start a new one."),
    include_figures: bool = Body(True, description="Include the Plotly charts (light and dark)."),
):
    """
    Returns the SQL that was run, the result rows, the chosen chart, a
    plain-language summary whose numbers were checked against the result,
    caveats from the data cleaning and every attempt the model made.
    Structure questions ("what columns are there?") return status
    "schema" with one row per column and "suggestions": questions to try.
    """

    paths, _ = require_ready_dataset(dataset_id)
    settings = get_model_settings(request)

    try:
        result = run_question(settings, paths, chat_id, question)

    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if result["status"] == "blocked":
        raise HTTPException(status_code=400, detail={
            "status": "blocked", "message": result["message"],
            "chat_id": result.get("chat_id"), "position": result.get("position"),
        })

    if model_failed(result):
        raise HTTPException(status_code=503, detail={
            "status": "failed", "message": result["message"],
            "chat_id": result.get("chat_id"), "position": result.get("position"),
        })

    logger.info("Dataset %s chat %s: %s in %.1fs", dataset_id, result.get("chat_id"),
                result["status"], result["seconds"])

    return build_response(dataset_id, result, include_figures)
