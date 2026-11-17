
# ============================================================
# FILE: src/api/delete_chat.py
# PURPOSE: DELETE /datasets/{dataset_id}/chats/{chat_id}
#
# Permanently delete one chat and all its turns. This cannot be undone.
# ============================================================

import logging

from fastapi import APIRouter

from src.helpers.chat_memory import delete_chat as delete_stored_chat

from src.api.common import get_existing_dataset, require_chat


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["chats"])


@router.delete("/{dataset_id}/chats/{chat_id}", summary="Permanently delete a chat")
def delete_chat(dataset_id: str, chat_id: int):
    """Removes the chat and its turns. This cannot be undone."""

    paths, _ = get_existing_dataset(dataset_id)

    require_chat(paths, chat_id)
    delete_stored_chat(paths["chats"], chat_id)
    logger.info("Deleted chat %s of dataset %s", chat_id, dataset_id)

    return {"deleted_chat": chat_id, "dataset_id": dataset_id}
