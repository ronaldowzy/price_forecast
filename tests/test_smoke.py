"""Smoke tests for key pipeline functions using real sample data.

These tests exercise the feature-engineering and data-processing functions
without running full model training (which would be too slow for CI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ======================================================================
# Data reading
# ======================================================================
class TestReadCsvRobust:
    def test_reads_sample(self, mod, sample_csv_path):
        df = mod.read_csv_robust(sample_csv_path)
        assert len(df) > 0
        assert "日期" in df.columns

    def test_reads_utf8(self, mod, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        df = mod.read_csv_robust(str(p))
        assert list(df.columns) == ["a", "b"]

    def test_reads_gbk(self, mod, tmp_path):
        p = tmp_path / "test.csv"
        p.write_bytes("日期,值\n2024-01-01,100\n".encode("gbk"))
        df = mod.read_csv_robust(str(p))
        assert "日期" in df.columns

    def test_raises_on_bad_path(self, mod):
        with pytest.raises(RuntimeError, match="Cannot read CSV"):
            mod.read_csv_robust("/nonexistent/path.csv")


class TestReadTable:
    def test_csv(self, mod, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("x,y\n1,2\n")
        df = mod.read_table(str(p))
        assert len(df) == 1

    def test_unsupported_ext(self, mod, tmp_path):
        p = tmp_path / "test.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported"):
            mod.read_table(str(p))


# ======================================================================
# add_slot_calendar
# ======================================================================
class TestAddSlotCalendar:
    def test_adds_columns(self, mod, sample_df):
        """add_slot_calendar should add slot_id, date_dt, is_holiday, etc."""
        # add_slot_calendar returns a copy — capture the return value
        df = mod.add_slot_calendar(sample_df.head(200))
        for col in ["slot_id", "date_dt", "is_holiday", "is_weekend", "month", "season"]:
            assert col in df.columns, f"Missing column after add_slot_calendar: {col}"

    def test_slot_id_range(self, mod, sample_df):
        df = mod.add_slot_calendar(sample_df.head(200))
        valid_ids = df["slot_id"].dropna()
        assert valid_ids.between(1, 96).all(), "slot_id should be in [1, 96]"

    def test_season_range(self, mod, sample_df):
        df = mod.add_slot_calendar(sample_df.head(200))
        valid_seasons = df["season"].dropna()
        assert valid_seasons.isin([1, 2, 3, 4]).all(), "season should be in [1, 2, 3, 4]"


# ======================================================================
# add_netload
# ======================================================================
class TestAddNetload:
    def test_creates_netload_columns(self, mod, sample_df):
        # add_netload returns a copy; for history data it creates NetLoad_fc and NetLoad_act
        df = mod.add_netload(sample_df.head(200), is_history=True)
        assert "NetLoad_fc" in df.columns, "NetLoad_fc column not created"

    def test_netload_is_numeric(self, mod, sample_df):
        df = mod.add_netload(sample_df.head(200), is_history=True)
        assert pd.api.types.is_numeric_dtype(df["NetLoad_fc"]), "NetLoad_fc should be numeric"


# ======================================================================
# build_history_feature_frame (lightweight — just first 960 rows)
# ======================================================================
class TestBuildHistoryFeatureFrame:
    def test_returns_dataframe(self, mod, sample_csv_path):
        """Verify the full feature pipeline runs without error on real data."""
        # Use the real file but only a subset won't work since it reads the
        # whole file — just verify it completes and returns a DataFrame.
        # This is the heaviest test; skip in ultra-fast CI if needed.
        df = mod.build_history_feature_frame(sample_csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_key_features(self, mod, sample_csv_path):
        df = mod.build_history_feature_frame(sample_csv_path)
        expected_features = ["slot_id", "NetLoad_fc", "month", "season"]
        for col in expected_features:
            assert col in df.columns, f"Missing feature column: {col}"


# ======================================================================
# parse_args
# ======================================================================
class TestParseArgs:
    def test_train_args(self, mod, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "train", "--history", "h.csv", "--model_dir", "m"],
        )
        args = mod.parse_args()
        assert args.cmd == "train"
        assert args.history == "h.csv"
        assert args.model_dir == "m"

    def test_predict_args(self, mod, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "predict", "--history", "h.csv", "--model_dir", "m",
             "--forecast", "f.csv", "--date", "2026-01-01", "--out", "o.xlsx"],
        )
        args = mod.parse_args()
        assert args.cmd == "predict"
        assert args.date == "2026-01-01"

    def test_backtest_args(self, mod, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "backtest", "--history", "h.csv", "--model_dir", "m",
             "--dates", "2025-01-01,2025-02-01", "--out", "o.xlsx"],
        )
        args = mod.parse_args()
        assert args.cmd == "backtest"
        assert "2025-01-01" in args.dates
