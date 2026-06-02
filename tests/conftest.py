"""Shared fixtures for price_forecast tests."""

from __future__ import annotations

import importlib
import os
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Resolve the repo root (one level up from tests/)
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Import the main forecasting module once per session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mod():
    """Import the latest annotated forecast module."""
    import sys

    # Ensure repo root is on sys.path so the module can be found
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module("rt_forecast_b_route_v2_6_9_annotated")


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_csv_path():
    """Path to the main historical dataset CSV."""
    p = DATA_DIR / "价格预测数据集.csv"
    if not p.exists():
        pytest.skip(f"Sample data not found: {p}")
    return str(p)


@pytest.fixture(scope="session")
def sample_input_csv_path():
    """Path to the single-day forecast input CSV."""
    p = DATA_DIR / "价格预测输入数据.csv"
    if not p.exists():
        pytest.skip(f"Input data not found: {p}")
    return str(p)


@pytest.fixture(scope="session")
def sample_df(mod, sample_csv_path):
    """Load the historical dataset (session-scoped, read once)."""
    return mod.read_csv_robust(sample_csv_path)
