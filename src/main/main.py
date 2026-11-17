
# ============================================================
# FILE: src/main/main.py
# PURPOSE: FastAPI application for the Text-to-SQL Data Analyst
#
# Run from the project root:
#   uvicorn src.main.main:app --reload
#
# Interactive docs: http://127.0.0.1:8000/docs
#
# Routers (one file per route in src/api/):
#   POST   /datasets                                     upload_dataset.py
#   GET    /datasets                                     list_datasets.py
#   GET    /datasets/{id}                                dataset_details.py
#   DELETE /datasets/{id}                                delete_dataset.py
#   POST   /datasets/{id}/ask                            ask_question.py
#   GET    /datasets/{id}/chats                          list_chats.py
#   GET    /datasets/{id}/chats/{chat_id}                chat_history.py
#   DELETE /datasets/{id}/chats/{chat_id}                delete_chat.py
#   DELETE /datasets/{id}/chats?chat_ids=..|all=true     delete_chats.py
#   GET    /datasets/{id}/chats/{chat_id}/turns/{n}/download
#                                                        download_result.py
# ============================================================

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from config import APP_LOG, DATASETS_DIR, MODEL_PROFILE, OLLAMA_URL
from src.helpers.logging_config import setup_logging
from src.helpers.model_selector import check_ollama, select_model

from src.api.upload_dataset import router as upload_dataset_router
from src.api.list_datasets import router as list_datasets_router
from src.api.dataset_details import router as dataset_details_router
from src.api.delete_dataset import router as delete_dataset_router
from src.api.ask_question import router as ask_question_router
from src.api.list_chats import router as list_chats_router
from src.api.chat_history import router as chat_history_router
from src.api.delete_chat import router as delete_chat_router
from src.api.delete_chats import router as delete_chats_router
from src.api.download_result import router as download_result_router


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 1. STARTUP / SHUTDOWN
# ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    """
    Configure logging, prepare the datasets folder and pick the model.
    Fails fast if Ollama is not reachable or a model is missing.
    """

    # Step 1: logging to console + data/logs/app.log.
    setup_logging(log_file=APP_LOG)

    # Step 2: the datasets folder must be writable.
    try:
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        logger.critical("Cannot create datasets folder %s: %s", DATASETS_DIR, exc)
        raise

    # Step 3: Ollama must answer and the models must be pulled.
    try:
        check_ollama(mode=MODEL_PROFILE)
        app.state.model_settings = select_model(MODEL_PROFILE)

    except (ConnectionError, ValueError) as exc:
        logger.critical("Startup aborted: %s", exc)
        raise

    logger.info("Model: %s (%s profile)", app.state.model_settings["model"],
                app.state.model_settings["profile"])
    logger.info("Datasets: %s", DATASETS_DIR)
    logger.info("Text-to-SQL Data Analyst API ready.")

    yield

    logger.info("Text-to-SQL Data Analyst API shutting down.")


# ------------------------------------------------------------
# 2. APPLICATION
# ------------------------------------------------------------

app = FastAPI(
    title="Text-to-SQL Data Analyst",
    description=(
        "Upload a CSV, Excel, JSON or Parquet file; it is cleaned "
        "automatically and stored in SQLite. Then ask questions in plain "
        "English: a local Ollama model writes the SQL, guardrails check it, "
        "failed queries are corrected automatically, and each answer comes "
        "with the result table, a chart and a checked summary. Follow-up "
        "questions can build on earlier ones in the same chat."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# One router per route file.
app.include_router(upload_dataset_router)
app.include_router(list_datasets_router)
app.include_router(dataset_details_router)
app.include_router(delete_dataset_router)
app.include_router(ask_question_router)
app.include_router(list_chats_router)
app.include_router(chat_history_router)
app.include_router(delete_chat_router)
app.include_router(delete_chats_router)
app.include_router(download_result_router)


@app.get("/", tags=["health"], summary="Health check")
def health():
    """Liveness check with the model in use; does not call the model."""

    settings = getattr(app.state, "model_settings", None) or {}

    return {
        "status": "ok",
        "service": "Text-to-SQL Data Analyst",
        "model": settings.get("model"),
        "profile": settings.get("profile"),
        "ollama_url": OLLAMA_URL,
    }


# ------------------------------------------------------------
# 3. ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("src.main.main:app", host="127.0.0.1", port=8000, reload=True)
