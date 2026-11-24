# ============================================================
# FILE: src/evaluation/run_stress.py
# PURPOSE: Stress set: question wordings small models get wrong
#
# Each question in data/input_data/eval/stress_questions.json has the
# expected SQL (or several acceptable readings). Both files are cleaned
# fresh with the review's suggested choices, then every question goes
# through answer_question() (the same SQL step as the app) for each
# model named on the command line, and is graded like the evaluation:
# same rows, column names and order ignored, numbers within rounding.
#
#   python -m src.evaluation.run_stress                        default 7B
#   python -m src.evaluation.run_stress qwen2.5:7b-instruct qwen2.5:3b-instruct
#
# Writes stress_results.csv and stress_summary.json to
# data/output_data/evaluation/stress/.
# ============================================================

import argparse
import json
import logging
import shutil
import time

import pandas as pd

from config import BASE_DIR, EVAL_DIR, EVALUATION_OUTPUT_DIR, MODEL_PROFILES, get_dataset_paths
from src.evaluation.result_matching import results_match
from src.evaluation.run_evaluation import expected_frames, load_json, warm_up
from src.helpers.cleaning_review import build_review, suggested_choices
from src.helpers.logging_config import setup_logging
from src.helpers.model_selector import check_ollama
from src.helpers.pipeline import answer_question, ingest_dataset, read_json



logger = logging.getLogger(__name__)

STRESS_FILE = EVAL_DIR / "stress_questions.json"
OUTPUT_DIR = EVALUATION_OUTPUT_DIR / "stress"
WORKSPACE_DIR = OUTPUT_DIR / "workspace"


def prepare(files):
    """Clean each file fresh, with the review's suggested choices applied."""

    shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
    workspaces = {}

    for number, (key, relative) in enumerate(files.items()):
        source = BASE_DIR / relative
        paths = get_dataset_paths(f"DS_20260101_00000{number}_STRS", datasets_dir=WORKSPACE_DIR)

        for folder in ("root", "input_dir", "logs_dir"):
            paths[folder].mkdir(parents=True, exist_ok=True)

        upload = paths["input_dir"] / source.name
        shutil.copyfile(source, upload)
        ingest_dataset(paths, upload, source.name)
        items = build_review(read_json(paths["cleaning_report"]))
        ingest_dataset(paths, upload, source.name, suggested_choices(items))
        workspaces[key] = paths

    return workspaces


def grade(settings, paths, item):
    """One question -> one result row."""

    started = time.perf_counter()
    outcome = answer_question(settings, paths, item["question"])
    seconds = round(time.perf_counter() - started, 2)
    frames = expected_frames(paths, item["expected"])

    if any(frame is None for frame in frames):
        raise ValueError(f"Expected SQL does not run for: {item['question']}")
    correct = (outcome["status"] in {"answered", "empty"} and outcome["frame"] is not None
               and any(results_match(outcome["frame"], frame) for frame in frames))

    return {"kind": item["kind"], "question": item["question"], "correct": correct,
            "status": outcome["status"], "seconds": seconds, "sql": outcome["sql"],
            "attempts": " > ".join(a.get("outcome", "?") for a in outcome["attempts"]),
            "got": (outcome["frame"].head(3).values.tolist()
                    if outcome["frame"] is not None else outcome.get("message")),
            "want": frames[0].head(3).values.tolist()}


def run(models):
    stress = load_json(STRESS_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_ollama(mode="auto")
    workspaces = prepare(stress["files"])
    rows = []

    for model in models:
        settings = {**MODEL_PROFILES["gpu"], "model": model}
        warm_up(settings)

        for item in stress["questions"]:
            row = grade(settings, workspaces[item["file"]], item)
            rows.append({"model": model, **row})
            logger.info("%-5s %-28s %s", "ok" if row["correct"] else "WRONG",
                        model, item["question"])

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "stress_results.csv", index=False)

    summary = {model: {"correct": f"{int(part['correct'].sum())}/{len(part)}",
                       "avg_seconds": round(part["seconds"].mean(), 2),
                       "by_kind": {kind: f"{int(group['correct'].sum())}/{len(group)}"
                                   for kind, group in part.groupby("kind")}}
               for model, part in results.groupby("model", sort=False)}
    (OUTPUT_DIR / "stress_summary.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Run the stress question set.")
    parser.add_argument("models", nargs="*", default=[MODEL_PROFILES["gpu"]["model"]])
    setup_logging()
    results, summary = run(parser.parse_args().models)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
