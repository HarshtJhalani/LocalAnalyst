# ============================================================
# FILE: src/api/cleaning_review.py
# PURPOSE: GET  /datasets/{dataset_id}/review
#          POST /datasets/{dataset_id}/review
#
# The review step after upload. GET lists the judgment calls the
# cleaning left open (missing values, look-alike values, numbers stored
# as text, guessed dates, list columns, extreme values), each with
# options and a suggested pick. POST applies the user's choices: the
# original upload is cleaned again with them and the database rebuilt.
# No model call; answers already in chats are not re-run.
#
#   POST body, one of:
#     {"choices": {table: {column: {issue: option}}}}
#     {"use_suggestions": true}     every item set to its suggestion
#     {"skip": true}                keep the automatic cleaning as it is
# ============================================================

import logging

from fastapi import APIRouter, Body, HTTPException, Request

from src.helpers.cleaning_review import (
    apply_spellings,
    build_review,
    pick_spellings,
    suggest_synonyms,
    suggested_choices,
    synonym_items,
    validate_choices,
)
from src.helpers.pipeline import ingest_dataset, load_dataset, read_json, write_json, write_status

from src.api.common import get_existing_dataset


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


def stored_choices(paths):
    """The choices applied last time ({} before any)."""

    if not paths["cleaning_choices"].is_file():
        return {}

    return read_json(paths["cleaning_choices"])


def review_items(request, paths, report):
    """
    Review items with the correct spelling of look-alike values picked
    by the model once per dataset (spellings.json); without a model the
    rule's picks stay.
    """

    choices = stored_choices(paths)
    items = build_review(report, choices)
    settings = getattr(request.app.state, "model_settings", None)
    stored = paths["root"] / "spellings.json"
    picked = read_json(stored) if stored.is_file() else None

    if picked is None and settings is not None:
        picked = {f"{item['table']}|{item['column']}":
                  pick_spellings(settings, item["column"], item["groups"])
                  for item in items
                  if item["issue"] == "similar_categories" and item.get("groups")}
        write_json(stored, picked)

    items = apply_spellings(items, picked or {})

    # Values with the same meaning: suggested once per dataset (synonyms.json),
    # from the values as first cleaned (before any merge the user applies).
    stored = paths["root"] / "synonyms.json"
    synonyms = read_json(stored) if stored.is_file() else None

    if synonyms is None and settings is not None:
        synonyms = suggest_synonyms(settings, load_dataset(paths)["profile"])
        write_json(stored, synonyms)

    return items + synonym_items(synonyms or {}, choices)


def ready_dataset(dataset_id):
    """Paths, status and cleaning report of a completed dataset, or 409."""

    paths, status = get_existing_dataset(dataset_id)

    if status.get("status") != "completed":
        raise HTTPException(status_code=409, detail="This dataset is not ready yet.")

    try:
        report = read_json(paths["cleaning_report"])

    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Could not read cleaning report of %s: %s", dataset_id, exc)
        raise HTTPException(status_code=500, detail="Dataset files are unreadable.")

    return paths, status, report


@router.get("/{dataset_id}/review", summary="Cleaning decisions for the user to review")
def get_review(request: Request, dataset_id: str):
    """
    review_state: "pending" (not yet reviewed), "done" or "none" (nothing
    to decide). Each item has options, a suggested option with a reason,
    and the option chosen last time (null before any).
    """

    paths, status, report = ready_dataset(dataset_id)

    return {"dataset_id": dataset_id,
            "review_state": status.get("review", "none"),
            "items": review_items(request, paths, report)}


@router.post("/{dataset_id}/review", summary="Apply cleaning choices (re-cleans the file)")
def apply_review(
    request: Request,
    dataset_id: str,
    choices: dict | None = Body(None, description="{table: {column: {issue: option}}}"),
    use_suggestions: bool = Body(False, description="Apply every suggested option."),
    skip: bool = Body(False, description="Keep the automatic cleaning; mark as reviewed."),
):
    paths, status, report = ready_dataset(dataset_id)
    items = review_items(request, paths, report)

    # Step 1: skipping changes no data.
    if skip:
        write_status(paths, "completed", "ready", review="done")
        return {"dataset_id": dataset_id, "review_state": "done", "applied": {},
                "items": items}

    # Step 2: the choices to apply.
    if use_suggestions:
        choices = suggested_choices(items)

    if choices is None:
        raise HTTPException(status_code=400,
                            detail="Send choices, use_suggestions=true or skip=true.")

    try:
        choices = validate_choices(choices, items)

        # "0 counts as missing" comes from the chat; the review card has no
        # control for it, so it stays as it was.
        for table, columns in stored_choices(paths).items():
            for column, issues in columns.items():
                if issues.get("zeros_as_missing") is True:
                    choices.setdefault(table, {}).setdefault(column, {})["zeros_as_missing"] = True

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Step 3: clean the original upload again with them.
    upload_path = paths["input_dir"] / status["file_name"]

    try:
        ingest_dataset(paths, upload_path, status["file_name"], choices)

    except ValueError as exc:
        # e.g. the choices removed every row: put the previous cleaning back.
        ingest_dataset(paths, upload_path, status["file_name"], stored_choices(paths))
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        logger.exception("Re-cleaning failed for %s", dataset_id)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    return {"dataset_id": dataset_id, "review_state": "done", "applied": choices,
            "items": build_review(read_json(paths["cleaning_report"]), choices)}
