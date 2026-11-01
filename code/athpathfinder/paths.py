"""Resolve PATHWAYML_* locations relative to this publication package."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_DIR.parent
REPO_ROOT = CODE_DIR.parent


def data_root() -> Path:
    env = os.environ.get("PATHWAYML_DATA_ROOT")
    if env:
        return Path(env)
    return REPO_ROOT / "data" / "input_data"


def processed_dir() -> Path:
    env = os.environ.get("PATHWAYML_PROCESSED_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "data" / "processed_data"


def grouped_output_dir() -> Path:
    env = os.environ.get("PATHWAYML_GROUPED_OUTPUT_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "data" / "grouped_analysis_results"


def output_dir() -> Path:
    env = os.environ.get("PATHWAYML_OUTPUT_DIR")
    if env:
        return Path(env)
    return CODE_DIR / "corrected_results"


def fits_dir() -> Path:
    env = os.environ.get("PATHWAYML_FITS_DIR")
    if env:
        return Path(env)
    return CODE_DIR / "fits"


DATA_ROOT = data_root()
DATA = DATA_ROOT / "data"
CURATED = (
    DATA_ROOT
    / "lofo/training_samples/model_13_comparison_curated/ratio_360_M"
    / "training_dataset_ratio_360_M_with_features.csv"
)
HIST = DATA / "historical/KEGG_2011_vs_current"
PROCESSED = processed_dir()
GROUPED_OUTPUT = grouped_output_dir()
OUT = output_dir()
FITS = fits_dir()
BIOC_URL = (
    "https://bioconductor.org/packages/2.10/data/annotation/src/contrib/"
    "org.At.tair.db_2.7.1.tar.gz"
)


def refresh() -> None:
    """Re-read env vars after the publication runner has set them."""
    global DATA_ROOT, DATA, CURATED, HIST, PROCESSED, GROUPED_OUTPUT, OUT, FITS
    DATA_ROOT = data_root()
    DATA = DATA_ROOT / "data"
    CURATED = (
        DATA_ROOT
        / "lofo/training_samples/model_13_comparison_curated/ratio_360_M"
        / "training_dataset_ratio_360_M_with_features.csv"
    )
    HIST = DATA / "historical/KEGG_2011_vs_current"
    PROCESSED = processed_dir()
    GROUPED_OUTPUT = grouped_output_dir()
    OUT = output_dir()
    FITS = fits_dir()
