
# ============================================================
# FILE: src/api/delete_dataset.py
# PURPOSE: DELETE /datasets/{dataset_id}
#
# Permanently remove a dataset: the uploaded file, its database, its
# reports and every chat about it. This cannot be undone.
# ============================================================

import logging
import shutil

from fastapi import APIRouter, HTTPException

from src.api.common import get_existing_dataset


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.delete("/{dataset_id}", summary="Permanently delete a dataset and its chats")
def delete_dataset(dataset_id: str):
    """Removes datasets/<dataset_id>/ entirely. This cannot be undone."""

    paths, _ = get_existing_dataset(dataset_id)

    try:
        shutil.rmtree(paths["root"])

    except OSError as exc:
        logger.exception("Could not delete dataset %s", dataset_id)
        raise HTTPException(status_code=500, detail=f"Could not delete the dataset: {exc}")

    logger.info("Deleted dataset %s", dataset_id)

    return {"deleted": dataset_id}
