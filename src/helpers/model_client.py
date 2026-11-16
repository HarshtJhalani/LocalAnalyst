
# ============================================================
# FILE: src/helpers/model_client.py
# PURPOSE: The one way every stage calls the model
#          (03_model_setup.ipynb, step 4)
#
# Ollama's format=<JSON schema> constrains the output to the schema,
# which is far more reliable for a 7B / 3B model than asking for JSON
# in the prompt. Temperature 0; output capped at MAX_OUTPUT_TOKENS.
#
#   get_client()   Ollama client for OLLAMA_URL
#   chat_json()    messages + schema -> (parsed answer, stats)
#
# Errors: ValueError for invalid / incomplete JSON (the retry loop
# handles it); httpx.HTTPError and ollama.ResponseError when the model
# times out or cannot be reached (callers turn them into a failure).
# ============================================================

import json
import logging
import time

import httpx
import ollama

from config import (
    CONNECT_TIMEOUT,
    KEEP_ALIVE,
    LLM_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    OLLAMA_URL,
)


logger = logging.getLogger(__name__)

# Errors meaning "the model did not answer", for callers to catch.
MODEL_ERRORS = (httpx.HTTPError, ollama.ResponseError)


def get_client(timeout=CONNECT_TIMEOUT):
    """Ollama client for OLLAMA_URL."""

    return ollama.Client(host=OLLAMA_URL, timeout=timeout)


def chat_json(settings, messages, schema):
    """
    Call the model and return its answer parsed against `schema`.

    settings: {"model", "num_ctx", "request_timeout"} and optionally
    "num_gpu" (0 forces CPU only).

    Returns:
        (dict answer, dict stats)
    """

    options = {"temperature": LLM_TEMPERATURE,
               "num_ctx": settings["num_ctx"],
               "num_predict": MAX_OUTPUT_TOKENS}

    if "num_gpu" in settings:
        options["num_gpu"] = settings["num_gpu"]

    # Step 1: call the model with the schema as the output format.
    client = get_client(settings["request_timeout"])
    started = time.perf_counter()
    response = client.chat(model=settings["model"], messages=messages,
                           format=schema, options=options, keep_alive=KEEP_ALIVE)
    content = response["message"]["content"]

    # Step 2: parse and check the required keys.
    try:
        answer = json.loads(content)

    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {content[:200]!r}") from exc

    missing = [key for key in schema.get("required", []) if key not in answer]

    if missing:
        raise ValueError(f"Model answer is missing keys {missing}: {answer}")

    stats = {"seconds": round(time.perf_counter() - started, 2),
             "prompt_tokens": response["prompt_eval_count"],
             "output_tokens": response["eval_count"]}

    return answer, stats
