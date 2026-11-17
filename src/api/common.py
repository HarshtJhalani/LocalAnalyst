
# ============================================================
# FILE: src/api/common.py
# PURPOSE: Helpers shared by the route modules (no routes here)
#
#   save_upload()            stream an UploadFile to disk with a size cap
#   get_existing_dataset()   dataset_id -> (paths, status) or 404
#   require_ready_dataset()  same, but 409 unless ingestion completed
#   require_chat()           chat row of a dataset, or 404
#   get_model_settings()     model chosen at startup, or 503
#   frame_to_payload()       DataFrame -> JSON-safe columns + rows
#   figures_payload()        stored Plotly JSON strings -> objects
# ============================================================

import json
import logging

from fastapi import HTTPException

from config import DATASETS_DIR, get_dataset_paths
from src.helpers.chat_memory import get_chat
from src.helpers.pipeline import read_status


logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024


# ============================================================
# 1. SAVE THE UPLOADED FILE
# ============================================================

def save_upload(upload, destination, max_bytes):
    """
    Stream an uploaded file to disk.

    Raises ValueError when the file is larger than max_bytes; the
    partial file is removed.
    """

    written = 0

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as target:
            while True:
                chunk = upload.file.read(CHUNK_SIZE)

                if not chunk:
                    break

                written += len(chunk)

                if written > max_bytes:
                    raise ValueError(
                        f"File is larger than {max_bytes // (1024 * 1024)} MB.")

                target.write(chunk)

    except ValueError:
        destination.unlink(missing_ok=True)
        raise

    except OSError as exc:
        raise OSError(f"Could not save upload to {destination}: {exc}") from exc

    finally:
        try:
            upload.file.close()
        except Exception:
            pass

    logger.debug("Upload saved: %s (%d bytes)", destination, written)

    return written


# ============================================================
# 2. RESOLVE AN EXISTING DATASET
# ============================================================

def get_existing_dataset(dataset_id):
    """
    Resolve a dataset ID to its paths and status, or raise 404.

    Returns:
        (paths, status)
    """

    # Step 1: the ID must be well-formed (blocks path traversal).
    try:
        paths = get_dataset_paths(dataset_id, datasets_dir=DATASETS_DIR)

    except ValueError:
        logger.warning("Rejected malformed dataset ID: %r", dataset_id)
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Step 2: the workspace must exist and ingestion must have started.
    if not paths["root"].is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        status = read_status(paths)

    except (ValueError, OSError) as exc:
        logger.error("Unreadable status.json for %s: %s", dataset_id, exc)
        raise HTTPException(status_code=500, detail="Dataset status file is unreadable.")

    if status is None:
        raise HTTPException(status_code=404, detail="Dataset was never ingested.")

    return paths, status


def require_ready_dataset(dataset_id):
    """Resolve a dataset and raise 409 unless ingestion completed."""

    paths, status = get_existing_dataset(dataset_id)

    if status.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Dataset is {status.get('status')} ({status.get('stage')}); "
                   "it cannot be queried.",
        )

    return paths, status


def require_chat(paths, chat_id):
    """The chat's row, or 404 (also when the dataset has no chats yet)."""

    chat = get_chat(paths["chats"], chat_id) if paths["chats"].is_file() else None

    if chat is None:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} does not exist.")

    return chat


# ============================================================
# 3. MODEL SETTINGS
# ============================================================

def get_model_settings(request):
    """Model settings chosen at startup (see src/main/main.py)."""

    settings = getattr(request.app.state, "model_settings", None)

    if settings is None:
        raise HTTPException(status_code=503, detail="The model is not ready.")

    return settings


# ============================================================
# 4. RESULTS AS JSON
# ============================================================

def frame_to_payload(frame):
    """
    DataFrame -> {"columns": [...], "rows": [[...], ...]}.

    pandas' JSON writer turns NaN / NA into null and dates into ISO text.
    """

    if frame is None:
        return {"columns": [], "rows": []}

    data = json.loads(frame.to_json(orient="split", index=False, date_format="iso"))

    return {"columns": data["columns"], "rows": data["data"]}


def figures_payload(figures):
    """{"light": json_text | None, ...} -> {"light": dict | None, ...}"""

    return {mode: json.loads(figure) if figure else None
            for mode, figure in (figures or {}).items()}
