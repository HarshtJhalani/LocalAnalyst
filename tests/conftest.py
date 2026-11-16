
# ============================================================
# FILE: tests/conftest.py
# PURPOSE: Shared fixtures: sample datasets ingested into a temporary
#          workspace once per test session (no model needed)
# ============================================================

import shutil

import pytest

from config import INPUT_DIR, get_dataset_paths
from src.helpers.pipeline import ingest_dataset


SAMPLES = {
    "messy_sales": "messy_sales.csv",
    "messy_employees": "messy_employees.xlsx",
    "customers": "customers.json",
    "products": "products.parquet",
}


def ingest_sample(datasets_dir, dataset_id, file_name):
    paths = get_dataset_paths(dataset_id, datasets_dir=datasets_dir)

    for key in ("root", "input_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    upload = paths["input_dir"] / file_name
    shutil.copyfile(INPUT_DIR / file_name, upload)
    ingest_dataset(paths, upload, file_name)

    return paths


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    """dataset_id -> paths for every ingested sample file."""

    root = tmp_path_factory.mktemp("datasets")

    return {dataset_id: ingest_sample(root, dataset_id, file_name)
            for dataset_id, file_name in SAMPLES.items()}


@pytest.fixture()
def sales(workspace):
    return workspace["messy_sales"]
