
# ============================================================
# FILE: src/api/dataset_details.py
# PURPOSE: GET /datasets/{dataset_id}
#
# Everything known about one dataset: status, tables, the cleaning
# report (what was changed and flagged), the schema text the model sees
# and the EDA profile (ranges, missing values, top categories,
# correlations).
# ============================================================

import logging

from fastapi import APIRouter, HTTPException, Query

from src.helpers.pipeline import load_dataset, read_json

from src.api.common import get_existing_dataset


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/{dataset_id}", summary="Status, cleaning report, schema and profile of a dataset")
def dataset_details(
    dataset_id: str,
    include_profile: bool = Query(
        True, description="Include the full EDA profile (larger response)."),
):
    """
    The status is always returned; the cleaning report, schema text and
    profile only once ingestion has completed.
    """

    paths, status = get_existing_dataset(dataset_id)
    details = {"dataset_id": dataset_id, "status": status}

    if status.get("status") != "completed":
        return details

    try:
        dataset = load_dataset(paths)
        details["cleaning_report"] = read_json(paths["cleaning_report"])

    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Could not read dataset %s: %s", dataset_id, exc)
        raise HTTPException(status_code=500, detail="Dataset files are unreadable.")

    details["schema_text"] = dataset["schema_text"]

    if include_profile:
        details["profile"] = dataset["profile"]

    return details
