
# ============================================================
# FILE: src/helpers/model_selector.py
# PURPOSE: Check Ollama and pick the model this machine runs well
#          (03_model_setup.ipynb, steps 2-3)
#
# Ollama decides GPU or CPU itself; the same model file runs on either.
# What we choose is the model size: load the small model and ask Ollama
# how much of it sits in GPU memory. No GPU -> "cpu" profile (3B).
# GPU -> try the 7B and keep it only if it fits entirely; a partly
# offloaded 7B is often slower than a 3B that fits.
#
#   check_ollama()   server reachable, profile models pulled
#   select_model()   MODEL_PROFILE auto | gpu | cpu -> settings dict
# ============================================================

import logging

from config import (
    GPU_FIT_SHARE,
    KEEP_ALIVE,
    MODEL_LOAD_TIMEOUT,
    MODEL_PROFILE,
    MODEL_PROFILES,
    OLLAMA_URL,
)
from src.helpers.model_client import get_client


logger = logging.getLogger(__name__)


# ============================================================
# 1. CONNECTING TO OLLAMA
# ============================================================

def check_ollama(client=None, mode=MODEL_PROFILE):
    """
    Make sure the server answers and the needed models are installed:
    only the forced profile's model for mode "gpu" / "cpu", both for
    "auto" (detection loads each of them).

    Returns:
        list of installed model names
    """

    client = client or get_client()

    # Step 1: the server must be reachable.
    try:
        installed = [model["model"] for model in client.list()["models"]]

    except Exception as exc:
        raise ConnectionError(
            f"Ollama is not reachable at {OLLAMA_URL}. Start it with "
            f"'ollama serve' (or the Ollama app). Details: {exc}"
        ) from exc

    # Step 2: the models this mode uses must be pulled.
    profiles = [mode] if mode in MODEL_PROFILES else list(MODEL_PROFILES)
    missing = [MODEL_PROFILES[profile]["model"] for profile in profiles
               if MODEL_PROFILES[profile]["model"] not in installed]

    if missing:
        raise ValueError(
            "Missing Ollama model(s): " + ", ".join(missing) + ". Pull with: "
            + " ; ".join(f"ollama pull {name}" for name in missing)
        )

    logger.info("Ollama reachable at %s, %d model(s) installed.",
                OLLAMA_URL, len(installed))

    return installed


# ============================================================
# 2. DETECTING GPU OR CPU
# ============================================================

def load_model(client, model, num_ctx, num_gpu=None):
    """Load a model into memory without generating anything."""

    options = {"num_ctx": num_ctx}

    if num_gpu is not None:
        options["num_gpu"] = num_gpu

    client.generate(model=model, prompt="", options=options, keep_alive=KEEP_ALIVE)


def unload_model(client, model):
    """Free the memory a model uses."""

    client.generate(model=model, prompt="", keep_alive=0)


def gpu_share(client, model):
    """
    Share of a loaded model that sits in GPU memory (0.0 - 1.0).
    None when the model is not loaded.
    """

    for loaded in client.ps()["models"]:
        if loaded["model"] == model and loaded["size"]:
            return loaded["size_vram"] / loaded["size"]

    return None


def detect_profile(client):
    """
    Pick "gpu" or "cpu" from what Ollama actually loaded. The models are
    loaded with the num_ctx they will be used with, since context size
    changes the memory they need.

    Returns:
        (profile name, details dict)
    """

    small, large = MODEL_PROFILES["cpu"], MODEL_PROFILES["gpu"]

    # Step 1: load the small model; no GPU memory used -> CPU machine.
    load_model(client, small["model"], small["num_ctx"])
    small_share = gpu_share(client, small["model"]) or 0.0
    details = {"small_model_gpu_share": round(small_share, 3)}

    if small_share == 0:
        details["reason"] = "No GPU memory used by the small model."
        return "cpu", details

    # Step 2: free the small model, then try the large one.
    unload_model(client, small["model"])
    load_model(client, large["model"], large["num_ctx"])
    large_share = gpu_share(client, large["model"]) or 0.0
    details["large_model_gpu_share"] = round(large_share, 3)

    if large_share >= GPU_FIT_SHARE:
        details["reason"] = f"{large['model']} fits entirely in GPU memory."
        return "gpu", details

    # Step 3: the large model only partly fits -> use the small one.
    unload_model(client, large["model"])
    load_model(client, small["model"], small["num_ctx"])
    details["reason"] = (f"{large['model']} only {large_share:.0%} in GPU memory; "
                         "the small model is faster.")

    return "cpu", details


def select_model(mode=MODEL_PROFILE, client=None):
    """
    Model settings for this machine.

    mode "gpu" / "cpu" skips detection; "auto" detects.

    Returns:
        {"profile", "model", "num_ctx", "max_retries", "request_timeout",
         "detection"}
    """

    if mode not in {"auto", "gpu", "cpu"}:
        raise ValueError(f"MODEL_PROFILE must be auto, gpu or cpu, not {mode!r}")

    # Detection loads models, which takes much longer than a connection
    # check, so it gets the long timeout.
    client = client or get_client(MODEL_LOAD_TIMEOUT)

    if mode == "auto":
        profile, details = detect_profile(client)
    else:
        profile, details = mode, {"reason": f"Forced by MODEL_PROFILE={mode}."}

    settings = {"profile": profile, **MODEL_PROFILES[profile], "detection": details}

    logger.info("Model profile: %s -> %s (%s)",
                profile, settings["model"], details["reason"])

    return settings
