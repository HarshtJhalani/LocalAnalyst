
# ============================================================
# FILE: src/api/chat_history.py
# PURPOSE: GET /datasets/{dataset_id}/chats/{chat_id}
#
# One chat with every turn in order: the question as typed, how it was
# interpreted, the SQL, the status and the first rows of the result
# (enough to redraw the chat without rerunning anything).
# ============================================================

import logging

from fastapi import APIRouter, Query

from src.helpers.chat_memory import load_turns

from src.api.common import figures_payload, frame_to_payload, get_existing_dataset, require_chat


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["chats"])


@router.get("/{dataset_id}/chats/{chat_id}", summary="One chat with all its turns")
def chat_history(
    dataset_id: str,
    chat_id: int,
    include_figures: bool = Query(True, description="Include the stored Plotly charts."),
):
    """
    Turns in order, each with a preview of up to 20 result rows and, for
    answers stored since summaries were kept, the summary, caveats and
    chart. Older turns have these as null / empty.
    """

    paths, _ = get_existing_dataset(dataset_id)
    chat = require_chat(paths, chat_id)

    turns = []

    for turn in load_turns(paths["chats"], chat_id):
        preview = turn.pop("preview")
        figures = turn.pop("figures")

        if include_figures and figures:
            turn["figures"] = figures_payload(figures)

        turns.append({**turn, "preview": frame_to_payload(preview)})

    return {**chat, "turns": turns}
