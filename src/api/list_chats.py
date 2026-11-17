
# ============================================================
# FILE: src/api/list_chats.py
# PURPOSE: GET /datasets/{dataset_id}/chats
#
# Every chat about one dataset, most recently used first, with its
# title and number of turns.
# ============================================================

import logging

from fastapi import APIRouter

from src.helpers.chat_memory import list_chats as list_stored_chats

from src.api.common import get_existing_dataset


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["chats"])


@router.get("/{dataset_id}/chats", summary="List the chats about a dataset")
def list_chats(dataset_id: str):
    """Chats, most recently used first."""

    paths, _ = get_existing_dataset(dataset_id)

    if not paths["chats"].is_file():
        return []

    return list_stored_chats(paths["chats"])
