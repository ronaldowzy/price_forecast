"""Unit tests for pure utility functions (no I/O, no side effects)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ======================================================================
# normalize_time_str
# ======================================================================
class TestNormalizeTimeStr:
    def test_zero_padded(self, mod):
        assert mod.normalize_time_str("00:15") == "00:15"

    def test_single_digit_hour(self, mod):
        assert mod.normalize_time_str("8:30") == "08:30"

    def test_with_seconds(self, mod):
        assert mod.normalize_time_str("12:00:00") == "12:00"

    def test_midnight_24(self, mod):
        assert mod.normalize_time_str("24:00") == "24:00"

    def test_whitespace(self, mod):
        assert mod.normalize_time_str("  09:45  ") == "09:45"


# ======================================================================
# slot_id_from_time
# ======================================================================
class TestSlotIdFromTime:
    def test_first_slot(self, mod):
        assert mod.slot_id_from_time("00:15") == 1

    def test_last_slot(self, mod):
        assert mod.slot_id_from_time("24:00") == 96

    def test_mid_slots(self, mod):
        assert mod.slot_id_from_time("01:00") == 4
        assert mod.slot_id_from_time("12:00") == 48
        assert mod.slot_id_from_time("23:45") == 95

    def test_invalid_returns_none(self, mod):
        assert mod.slot_id_from_time("abc") is None

    def test_boundary(self, mod):
        assert mod.slot_id_from_time("00:00") == 0


# ======================================================================
# map_yesno
# ======================================================================
class TestMapYesno:
    @pytest.mark.parametrize("val", ["是", "Y", "y", "yes", "Yes", "1", "TRUE", "True", "true"])
    def test_truthy(self, mod, val):
        assert mod.map_yesno(val) == 1

    @pytest.mark.parametrize("val", ["否", "N", "n", "no", "No", "0", "FALSE", "False", "false"])
    def test_falsy(self, mod, val):
        assert mod.map_yesno(val) == 0

    def test_nan(self, mod):
        assert mod.map_yesno(None) is None
        assert mod.map_yesno(np.nan) is None

    def test_numeric_string(self, mod):
        assert mod.map_yesno("2") == 2


# ======================================================================
# season_from_month
# ======================================================================
class TestSeasonFromMonth:
    @pytest.mark.parametrize("m", [12, 1, 2])
    def test_winter(self, mod, m):
        assert mod.season_from_month(m) == 1

    @pytest.mark.parametrize("m", [3, 4, 5])
    def test_spring(self, mod, m):
        assert mod.season_from_month(m) == 2

    @pytest.mark.parametrize("m", [6, 7, 8])
    def test_summer(self, mod, m):
        assert mod.season_from_month(m) == 3

    @pytest.mark.parametrize("m", [9, 10, 11])
    def test_autumn(self, mod, m):
        assert mod.season_from_month(m) == 4


# ======================================================================
# require_cols
# ======================================================================
class TestRequireCols:
    def test_no_error_when_present(self, mod):
        df = pd.DataFrame({"a": [1], "b": [2]})
        mod.require_cols(df, ["a", "b"])  # should not raise

    def test_raises_on_missing(self, mod):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(KeyError, match="Missing columns"):
            mod.require_cols(df, ["a", "b", "c"])

    def test_includes_where_context(self, mod):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(KeyError, match="test_context"):
            mod.require_cols(df, ["a", "b"], where="test_context")


# ======================================================================
# clip_nonneg
# ======================================================================
class TestClipNonneg:
    def test_clips_negatives(self, mod):
        df = pd.DataFrame({"x": [-5, 0, 3, -1.5]})
        mod.clip_nonneg(df, ["x"])
        assert (df["x"] >= 0).all()
        assert df["x"].tolist() == [0.0, 0.0, 3.0, 0.0]

    def test_ignores_missing_col(self, mod):
        df = pd.DataFrame({"x": [1]})
        mod.clip_nonneg(df, ["nonexistent"])  # should not raise


# ======================================================================
# _enforce_monotonic
# ======================================================================
class TestEnforceMonotonic:
    def test_already_monotone(self, mod):
        q10 = np.array([10.0, 20.0])
        q50 = np.array([15.0, 25.0])
        q90 = np.array([20.0, 30.0])
        r10, r50, r90 = mod._enforce_monotonic(q10, q50, q90)
        np.testing.assert_array_equal(r10, q10)
        np.testing.assert_array_equal(r50, q50)
        np.testing.assert_array_equal(r90, q90)

    def test_fixes_violations(self, mod):
        q10 = np.array([100.0])
        q50 = np.array([50.0])   # q50 < q10 → violation
        q90 = np.array([30.0])   # q90 < q50 → violation
        r10, r50, r90 = mod._enforce_monotonic(q10, q50, q90)
        assert r10[0] == 100.0
        assert r50[0] == 100.0  # clamped up
        assert r90[0] == 100.0  # clamped up


# ======================================================================
# _renorm_probs
# ======================================================================
class TestRenormProbs:
    def test_sum_to_one(self, mod):
        p_neg = np.array([0.1, 0.3])
        p_spike = np.array([0.2, 0.5])
        p_norm, p_s, p_n = mod._renorm_probs(p_neg, p_spike)
        total = p_norm + p_s + p_n
        np.testing.assert_allclose(total, 1.0, atol=1e-10)

    def test_clips_negative(self, mod):
        p_neg = np.array([-0.1])
        p_spike = np.array([0.2])
        p_norm, p_s, p_n = mod._renorm_probs(p_neg, p_spike)
        assert p_n[0] >= 0
        assert p_s[0] >= 0
        assert p_norm[0] >= 0

    def test_overshoot_scaled(self, mod):
        """When p_neg + p_spike > 1, they should be scaled down."""
        p_neg = np.array([0.6])
        p_spike = np.array([0.7])
        p_norm, p_s, p_n = mod._renorm_probs(p_neg, p_spike)
        assert p_n[0] + p_s[0] <= 1.0 + 1e-10
        assert p_norm[0] >= 0
