
# ============================================================
# FILE: src/api/delete_chats.py
# PURPOSE: DELETE /datasets/{dataset_id}/chats?chat_ids=1&chat_ids=2
#          DELETE /datasets/{dataset_id}/chats?all=true
#
# Permanently delete several chats (or all of them) in one call.
# Every id is checked first: if one is unknown, nothing is deleted.
# ============================================================

import logging

from fastapi import APIRouter, HTTPException, Query

from src.helpers.chat_memory import delete_chat, list_chats

from src.api.common import get_existing_dataset


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["chats"])


@router.delete("/{dataset_id}/chats", summary="Permanently delete several chats")
def delete_chats(
    dataset_id: str,
    chat_ids: list[int] = Query(default=[], description="Chats to delete (repeat the parameter)."),
    delete_all: bool = Query(default=False, alias="all",
                             description="Delete every chat of the dataset."),
):
    """Removes the chosen chats and their turns. This cannot be undone."""

    paths, _ = get_existing_dataset(dataset_id)

    # Step 1: the chats that exist (a dataset without chats.db has none).
    existing = ([chat["chat_id"] for chat in list_chats(paths["chats"])]
                if paths["chats"].is_file() else [])

    # Step 2: what to delete.
    if delete_all:
        targets = existing

    else:
        targets = list(dict.fromkeys(chat_ids))          # drop repeats, keep order

        if not targets:
            raise HTTPException(status_code=400,
                                detail="Give at least one chat_ids value, or all=true.")

        unknown = [chat_id for chat_id in targets if chat_id not in existing]

        if unknown:
            raise HTTPException(status_code=404,
                                detail=f"Chat(s) {unknown} do not exist. Nothing was deleted.")

    # Step 3: delete.
    for chat_id in targets:
        delete_chat(paths["chats"], chat_id)

    logger.info("Deleted chats %s of dataset %s", targets, dataset_id)

    return {"deleted_chats": targets, "count": len(targets), "dataset_id": dataset_id}
