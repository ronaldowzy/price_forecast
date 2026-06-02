"""Schema validation for sample data files."""

from __future__ import annotations

import pandas as pd
import pytest


# Columns expected in the historical dataset (Chinese headers)
EXPECTED_COLS_HIST = [
    "日期",
    "时刻",
    "是否节假日",
    "是否周末休息日",
    "直调负荷(预测)",
    "联络线受电负荷(预测)",
    "风电总加(预测)",
    "光伏总加(预测)",
    "非市场化核电总加(预测)",
    "自备机组总加(预测)",
    "地方电厂发电总加(预测)",
    "直调负荷(实际)",
    "联络线受电负荷(实际)",
    "风电总加(实际)",
    "光伏总加(实际)",
    "抽蓄(实际)",
    "地方电厂发电总加(实际)",
    "非市场化核电总加(实际)",
    "自备机组总加(实际)",
    "日前价格",
    "实时价格",
]


class TestHistoricalDataSchema:
    """Validate the main historical dataset structure."""

    def test_file_loads(self, sample_df):
        assert isinstance(sample_df, pd.DataFrame)
        assert len(sample_df) > 0

    def test_has_expected_columns(self, sample_df):
        missing = [c for c in EXPECTED_COLS_HIST if c not in sample_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_date_column_format(self, sample_df):
        """日期 should be parseable as dates."""
        dates = pd.to_datetime(sample_df["日期"], errors="coerce")
        assert dates.notna().sum() > 0, "No valid dates found in 日期 column"

    def test_time_slot_values(self, sample_df):
        """时刻 should contain 15-min interval labels."""
        valid_slots = {"00:15", "00:30", "00:45", "01:00", "24:00"}
        unique_times = set(sample_df["时刻"].astype(str).unique())
        # At least some standard slots should be present
        overlap = unique_times & valid_slots
        assert len(overlap) > 0, f"No standard time slots found. Got: {list(unique_times)[:5]}"

    def test_row_count_multiple_of_96(self, sample_df):
        """Each day has 96 slots; total rows should be ~N*96."""
        n = len(sample_df)
        # Allow a small remainder for incomplete trailing days
        assert n >= 96, f"Expected at least 96 rows, got {n}"
        assert n % 96 <= 5, f"Row count {n} is not close to a multiple of 96"

    def test_numeric_columns_not_all_null(self, sample_df):
        """Key numeric columns should have data."""
        for col in ["直调负荷(预测)", "风电总加(预测)", "光伏总加(预测)", "日前价格"]:
            if col in sample_df.columns:
                non_null = sample_df[col].notna().sum()
                assert non_null > 0, f"Column '{col}' is entirely null"


class TestInputDataSchema:
    """Validate the single-day forecast input data."""

    def test_input_loads(self, mod, sample_input_csv_path):
        df = mod.read_csv_robust(sample_input_csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 96, f"Expected 96 rows for one day, got {len(df)}"

    def test_input_has_forecast_columns(self, mod, sample_input_csv_path):
        df = mod.read_csv_robust(sample_input_csv_path)
        for col in ["直调负荷(预测)", "风电总加(预测)", "光伏总加(预测)"]:
            assert col in df.columns, f"Missing forecast column: {col}"
