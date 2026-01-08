# rt_forecast_b_route_v2_2.py
# Route-B RT price forecasting (15-min, 96 points/day) for Shandong market.
#
# V2.2 upgrades (vs v2/v2.1):
# 1) Add NEGATIVE price regime (common in PV/Wind high periods):
#      - P_neg = P(RT < NEG_THRESHOLD_RT)
#      - Negative magnitude quantile regressors trained on y_neg = -RT for RT < NEG_THRESHOLD_RT
# 2) Keep SPIKE regime:
#      - P_spike = P(RT > SPIKE_THRESHOLD_RT)
#      - Spike excess quantile regressors trained on log(RT - SPIKE_THRESHOLD_RT + 1)
# 3) Probability calibration (time-series OOF) for P_neg & P_spike (IsotonicRegression if available)
# 4) Provide BOTH mixture outputs and gate outputs (more trading-friendly):
#      - Mixture:
#          RT_mix_p50 = pN*RT_norm_p50 + pS*RT_spike_p50 + pNeg*RT_neg_p50
#          RT_mix_upper90 = pN*RT_norm_p90 + pS*RT_spike_p90 + pNeg*RT_neg_p50
#          RT_mix_lower10 = pN*RT_norm_p10 + pS*RT_spike_p50 + pNeg*RT_neg_p90
#      - Gate (hard switch):
#          RT_gate_p50 (priority: negative -> spike -> normal)
#          RT_gate_risk90 (upper risk; negative uses norm upper by default, but outputs neg scenario columns too)
#
# Notes:
# - Time slots are "interval end" labels: 00:15 ... 24:00 (NO 00:00).
# - Missing rows in history are dropped during training as needed.
# - Wind/PV are clipped to >= 0 (negative values treated as noise).
# - NetLoad includes tie-line: NetLoad = Load - Wind - PV - Tie.
#
# Dependencies:
#   pip install pandas numpy scikit-learn joblib openpyxl
#
# Usage (CLI):
#   Train:
#     python rt_forecast_b_route_v2_2.py train --history "价格预测数据集.csv" --model_dir "models_v2_2" --th_spike 800 --th_neg 0
#   Predict:
#     python rt_forecast_b_route_v2_2.py predict --history "价格预测数据集.csv" --model_dir "models_v2_2" --forecast "Dplus1_forecast.xlsx" --date "2026-01-01" --out "RT_pred_2026-01-01.xlsx"
#   Backtest:
#     python rt_forecast_b_route_v2_2.py backtest --history "价格预测数据集.csv" --model_dir "models_v2_2" --dates "2025-11-08,2025-01-26" --out "eval_backtest_v2_2.xlsx"
#
# If you prefer editing parameters inside the script, use run_without_args().

from __future__ import annotations

import argparse
import os
import re
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# Optional calibration (older sklearn might not have it; we'll fallback gracefully)
try:
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover
    IsotonicRegression = None  # type: ignore


# =========================
# Config (thresholds)
# =========================

SPIKE_THRESHOLD_RT_DEFAULT = 800.0  # spike: RT > 800
NEG_THRESHOLD_RT_DEFAULT = 0.0      # negative regime: RT < 0

# Gate thresholds (recommended starting points; tune via backtest)
P_GATE_SPIKE_DEFAULT = 0.15
P_GATE_NEG_DEFAULT = 0.30


P_FULL_NEG_DEFAULT = 0.50  # ramp gate: full negative weight when p_neg >= p_full_neg
ARM_DAY_NEG_DEFAULT = 0.40  # day-level arming threshold for negative gating (set <=0 to disable)
ARM_DAY_NEG_PV_TH_DEFAULT = 11500.0   # PV_fc daily max threshold to allow negative-day arming
ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT = 7000.0  # NetLoad_fc daily min threshold to allow negative-day arming
ARM_DAY_NEG_USE_OR_DEFAULT = True    # True: (PV OR NetLoad) ; False: (PV AND NetLoad)


# Spike budget trigger (v2.5): select a small number of high-risk slots per day
SPIKE_BUDGET_ENABLE_DEFAULT = True
SPIKE_BUDGET_PTS_DEFAULT = 6          # max number of 15-min slots to flag per day
SPIKE_BUDGET_SEEDS_DEFAULT = 2        # number of seed slots before expansion
SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT = 1 # expand each seed by +/- radius (in 15-min slots)
SPIKE_BUDGET_MIN_GAP_DEFAULT = 4      # minimum separation between seeds (in 15-min slots)
SPIKE_APPLY_TO_P50_DEFAULT = False    # if True, spike budget affects RT_gate_p50; if False, only affects risk bounds & scenario
# =========================
# Column names
# =========================

COL_DATE = "日期"
COL_TIME = "时刻"
COL_HOL = "是否节假日"
COL_WKND = "是否周末休息日"

COL_LOAD_FC = "直调负荷(预测)"
COL_TIE_FC = "联络线受电负荷(预测)"
COL_WIND_FC = "风电总加(预测)"
COL_PV_FC = "光伏总加(预测)"

COL_LOAD_ACT = "直调负荷(实际)"
COL_TIE_ACT = "联络线受电负荷(实际)"
COL_WIND_ACT = "风电总加(实际)"
COL_PV_ACT = "光伏总加(实际)"

COL_RT = "实时价格"


# =========================
# Feature sets (unchanged from v2 to keep continuity)
# =========================

FEATURE_B1 = [
    "slot_id", "is_holiday", "is_weekend", "month", "season",
    "NetLoad_fc", "Ramp_fc_1", "Ramp_fc_4",
    "PV_fc", "dPV_fc_1", "Wind_fc", "Tie_fc",
    "Err_1", "Err_mean_7", "Err_absmean_7", "Err_p10_14", "Err_p90_14",
]

FEATURE_B2 = [
    "slot_id", "is_holiday", "is_weekend", "month", "season",
    "NetLoad_fc", "Ramp_fc_1", "Ramp_fc_4",
    "PV_fc", "dPV_fc_1", "Wind_fc", "Tie_fc",
    "RT_lag_1d", "RT_lag_7d", "RT_vol_1d",
    "NetLoad_act_p10", "NetLoad_act_p50", "NetLoad_act_p90",
]


# =========================
# Utilities
# =========================

def read_csv_robust(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot read CSV: {path}. Last error: {last_err}")


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return read_csv_robust(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {ext}. Use CSV or XLSX.")


def normalize_time_str(s: object) -> str:
    s = str(s).strip()
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if not m:
        return s
    hh = int(m.group(1))
    mm = int(m.group(2))
    return f"{hh:02d}:{mm:02d}"


def slot_id_from_time(s: object) -> Optional[int]:
    s = normalize_time_str(s)
    m = re.match(r"^(\d{2}):(\d{2})$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh == 24 and mm == 0:
        return 96
    return hh * 4 + (mm // 15)  # 00:15 -> 1 ... 23:45 -> 95


def map_yesno(x: object) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in ["是", "Y", "y", "yes", "Yes", "1", "TRUE", "True", "true"]:
        return 1
    if s in ["否", "N", "n", "no", "No", "0", "FALSE", "False", "false"]:
        return 0
    try:
        return int(float(s))
    except Exception:
        return None


def season_from_month(m: int) -> int:
    # 1=Winter,2=Spring,3=Summer,4=Autumn
    if m in (12, 1, 2):
        return 1
    if m in (3, 4, 5):
        return 2
    if m in (6, 7, 8):
        return 3
    return 4


def require_cols(df: pd.DataFrame, cols: List[str], where: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"Missing columns {missing}"
        if where:
            msg += f" in {where}"
        msg += f". Available columns: {list(df.columns)}"
        raise KeyError(msg)


def clip_nonneg(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0)


def to_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def add_slot_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    require_cols(df, [COL_DATE, COL_TIME, COL_HOL, COL_WKND], where="input")

    df["date_dt"] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df["time_str"] = df[COL_TIME].map(normalize_time_str)
    df["slot_id"] = df["time_str"].map(slot_id_from_time)

    df["is_holiday"] = df[COL_HOL].map(map_yesno)
    df["is_weekend"] = df[COL_WKND].map(map_yesno)

    df["month"] = df["date_dt"].dt.month
    df["season"] = df["month"].map(season_from_month)

    df["slot_id"] = pd.to_numeric(df["slot_id"], errors="coerce")
    return df


def add_netload(df: pd.DataFrame, is_history: bool) -> pd.DataFrame:
    df = df.copy()
    require_cols(df, [COL_LOAD_FC, COL_TIE_FC, COL_WIND_FC, COL_PV_FC], where="forecast fields")

    # Treat negative PV/Wind readings as noise; keep non-negative.
    clip_nonneg(df, [COL_WIND_FC, COL_PV_FC])
    to_numeric(df, [COL_LOAD_FC, COL_TIE_FC])

    df["Load_fc"] = df[COL_LOAD_FC]
    df["Tie_fc"] = df[COL_TIE_FC]
    df["Wind_fc"] = pd.to_numeric(df[COL_WIND_FC], errors="coerce")
    df["PV_fc"] = pd.to_numeric(df[COL_PV_FC], errors="coerce")

    df["NetLoad_fc"] = df["Load_fc"] - df["Wind_fc"] - df["PV_fc"] - df["Tie_fc"]

    if is_history:
        require_cols(df, [COL_LOAD_ACT, COL_TIE_ACT, COL_WIND_ACT, COL_PV_ACT], where="actual fields")
        clip_nonneg(df, [COL_WIND_ACT, COL_PV_ACT])
        to_numeric(df, [COL_LOAD_ACT, COL_TIE_ACT])

        df["Load_act"] = df[COL_LOAD_ACT]
        df["Tie_act"] = df[COL_TIE_ACT]
        df["Wind_act"] = pd.to_numeric(df[COL_WIND_ACT], errors="coerce")
        df["PV_act"] = pd.to_numeric(df[COL_PV_ACT], errors="coerce")

        df["NetLoad_act"] = df["Load_act"] - df["Wind_act"] - df["PV_act"] - df["Tie_act"]
        df["dNetLoad"] = df["NetLoad_act"] - df["NetLoad_fc"]

    return df


def add_within_day_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["date_dt", "slot_id"]).reset_index(drop=True)
    g = df.groupby("date_dt", group_keys=False)

    # keep all 96 slots
    df["Ramp_fc_1"] = g["NetLoad_fc"].diff(1).fillna(0.0)
    df["Ramp_fc_4"] = g["NetLoad_fc"].diff(4).fillna(0.0)
    df["dPV_fc_1"] = g["PV_fc"].diff(1).fillna(0.0)

    return df


def add_error_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Error profile per slot_id across days, using STRICTLY PAST dNetLoad:
      Err_1 = dNetLoad(d-1,t)
      Err_mean_7, Err_absmean_7 from last 7
      Err_p10_14, Err_p90_14 from last 14
    """
    df = df.copy().sort_values(["date_dt", "slot_id"]).reset_index(drop=True)
    require_cols(df, ["dNetLoad", "slot_id", "date_dt"], where="error-profile base")

    def _calc(group: pd.DataFrame) -> pd.DataFrame:
        s = group["dNetLoad"]
        s1 = s.shift(1)  # strictly past
        out = pd.DataFrame(index=group.index)
        out["Err_1"] = s1
        out["Err_mean_7"] = s1.rolling(7, min_periods=5).mean()
        out["Err_absmean_7"] = s1.abs().rolling(7, min_periods=5).mean()
        out["Err_p10_14"] = s1.rolling(14, min_periods=10).quantile(0.1)
        out["Err_p90_14"] = s1.rolling(14, min_periods=10).quantile(0.9)
        return out

    stats = df.groupby("slot_id", group_keys=False).apply(_calc)
    for c in stats.columns:
        df[c] = stats[c]
    return df


def add_rt_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["date_dt", "slot_id"]).reset_index(drop=True)
    require_cols(df, [COL_RT, "slot_id", "date_dt"], where="RT lags base")
    df[COL_RT] = pd.to_numeric(df[COL_RT], errors="coerce")

    g = df.groupby("slot_id", group_keys=False)
    df["RT_lag_1d"] = g[COL_RT].shift(1)
    df["RT_lag_7d"] = g[COL_RT].shift(7)

    daily_std = df.groupby("date_dt")[COL_RT].std().sort_index()
    daily_std_shift = daily_std.shift(1).rename("RT_vol_1d")
    df = df.merge(daily_std_shift, left_on="date_dt", right_index=True, how="left")

    return df


def sanity_check_tie_sign(df_hist: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    tmp = df_hist.dropna(subset=["NetLoad_act", "NetLoad_fc", COL_RT]).copy()
    if len(tmp) < 100:
        return {"corr_act_rt": np.nan, "corr_fc_rt": np.nan}
    out["corr_act_rt"] = float(tmp["NetLoad_act"].corr(tmp[COL_RT]))
    out["corr_fc_rt"] = float(tmp["NetLoad_fc"].corr(tmp[COL_RT]))
    return out


# =========================
# Models
# =========================

def _fit_quantile_gbr(X: np.ndarray, y: np.ndarray, alpha: float, seed: int = 42) -> GradientBoostingRegressor:
    m = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=450,
        learning_rate=0.05,
        subsample=0.7,
        max_depth=3,
        min_samples_leaf=25,
        random_state=seed,
    )
    m.fit(X, y)
    return m


def train_b1_models(df: pd.DataFrame) -> Dict[str, object]:
    X = df[FEATURE_B1].astype(float).values
    y = df["dNetLoad"].astype(float).values
    return {
        "p10": _fit_quantile_gbr(X, y, 0.10),
        "p50": _fit_quantile_gbr(X, y, 0.50),
        "p90": _fit_quantile_gbr(X, y, 0.90),
    }


def b1_oof_predictions(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    OOF predictions by time-split days for B1 dNetLoad quantiles to avoid leakage into B2.
    """
    df = df.copy()
    days = np.array(sorted(df["date_dt"].unique()))
    if len(days) < 10:
        X = df[FEATURE_B1].astype(float).values
        y = df["dNetLoad"].astype(float).values
        m10 = _fit_quantile_gbr(X, y, 0.10)
        m50 = _fit_quantile_gbr(X, y, 0.50)
        m90 = _fit_quantile_gbr(X, y, 0.90)
        df["dNet_p10_oof"] = m10.predict(X)
        df["dNet_p50_oof"] = m50.predict(X)
        df["dNet_p90_oof"] = m90.predict(X)
        return df

    n_splits = min(n_splits, max(2, len(days) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    pred10 = np.full(len(df), np.nan)
    pred50 = np.full(len(df), np.nan)
    pred90 = np.full(len(df), np.nan)

    for train_idx, test_idx in tscv.split(days):
        train_days = set(days[train_idx])
        test_days = set(days[test_idx])

        train_mask = df["date_dt"].isin(train_days).values
        test_mask = df["date_dt"].isin(test_days).values

        Xtr = df.loc[train_mask, FEATURE_B1].astype(float).values
        ytr = df.loc[train_mask, "dNetLoad"].astype(float).values
        Xte = df.loc[test_mask, FEATURE_B1].astype(float).values

        m10 = _fit_quantile_gbr(Xtr, ytr, 0.10)
        m50 = _fit_quantile_gbr(Xtr, ytr, 0.50)
        m90 = _fit_quantile_gbr(Xtr, ytr, 0.90)

        pred10[test_mask] = m10.predict(Xte)
        pred50[test_mask] = m50.predict(Xte)
        pred90[test_mask] = m90.predict(Xte)

    df["dNet_p10_oof"] = pred10
    df["dNet_p50_oof"] = pred50
    df["dNet_p90_oof"] = pred90
    return df


def _fit_classifier(X: np.ndarray, y: np.ndarray, seed: int = 42):
    pos = int(y.sum())
    if pos < 10:
        clf = DummyClassifier(strategy="prior")
        clf.fit(X, y)
        return clf, pos
    clf = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=400,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf, pos


def _clf_predict_proba(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    # fallback
    return np.full(X.shape[0], np.nan)


def _oof_proba_time_split(df: pd.DataFrame, y_col: str, n_splits: int = 5) -> np.ndarray:
    """
    Time-series OOF probabilities for a binary classifier.
    """
    days = np.array(sorted(df["date_dt"].unique()))
    if len(days) < 10:
        X = df[FEATURE_B2].astype(float).values
        y = df[y_col].astype(int).values
        clf, _ = _fit_classifier(X, y)
        return _clf_predict_proba(clf, X)

    n_splits = min(n_splits, max(2, len(days) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    p_oof = np.full(len(df), np.nan)

    for train_idx, test_idx in tscv.split(days):
        train_days = set(days[train_idx])
        test_days = set(days[test_idx])

        train_mask = df["date_dt"].isin(train_days).values
        test_mask = df["date_dt"].isin(test_days).values

        Xtr = df.loc[train_mask, FEATURE_B2].astype(float).values
        ytr = df.loc[train_mask, y_col].astype(int).values
        Xte = df.loc[test_mask, FEATURE_B2].astype(float).values

        clf, _ = _fit_classifier(Xtr, ytr)
        p_oof[test_mask] = _clf_predict_proba(clf, Xte)

    return p_oof


def _fit_isotonic_calibrator(p: np.ndarray, y: np.ndarray):
    """
    Fit isotonic calibration mapping p_raw -> p_cal. If IsotonicRegression unavailable, returns None.
    """
    if IsotonicRegression is None:
        return None
    mask = np.isfinite(p) & np.isfinite(y)
    if mask.sum() < 200:
        return None
    # If p is constant or y is single class, calibration is meaningless
    if np.nanstd(p[mask]) < 1e-6:
        return None
    if len(np.unique(y[mask].astype(int))) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p[mask], y[mask].astype(float))
    return iso


def _apply_calibrator(cal, p_raw: np.ndarray) -> np.ndarray:
    if cal is None:
        return p_raw
    p = p_raw.copy()
    mask = np.isfinite(p)
    p[mask] = cal.predict(p[mask])
    return p


def train_b2_regime_models(
    df_b2: pd.DataFrame,
    th_spike: float,
    th_neg: float,
) -> Dict[str, object]:
    """
    Train B2 regime models on RT:
      - Two binary classifiers (spike, neg) with isotonic calibration using time OOF
      - Normal quantile regressors on th_neg <= RT <= th_spike
      - Spike excess quantile regressors on log(RT-th_spike+1) for RT > th_spike
      - Neg magnitude quantile regressors on y=-RT for RT < th_neg
    """
    X = df_b2[FEATURE_B2].astype(float).values
    y = df_b2[COL_RT].astype(float).values

    y_spike = (y > th_spike).astype(int)
    y_neg = (y < th_neg).astype(int)

    # --- OOF probs for calibration ---
    df_tmp = df_b2.copy()
    df_tmp["y_spike"] = y_spike
    df_tmp["y_neg"] = y_neg

    p_spike_oof = _oof_proba_time_split(df_tmp, "y_spike", n_splits=5)
    p_neg_oof = _oof_proba_time_split(df_tmp, "y_neg", n_splits=5)

    cal_spike = _fit_isotonic_calibrator(p_spike_oof, y_spike)
    cal_neg = _fit_isotonic_calibrator(p_neg_oof, y_neg)

    # --- final classifiers on all data ---
    clf_spike, pos_spike = _fit_classifier(X, y_spike)
    clf_neg, pos_neg = _fit_classifier(X, y_neg)

    # --- normal quantile regressors ---
    idx_norm = (y >= th_neg) & (y <= th_spike)
    Xn, yn = X[idx_norm], y[idx_norm]
    norm_models = {
        "p10": _fit_quantile_gbr(Xn, yn, 0.10),
        "p50": _fit_quantile_gbr(Xn, yn, 0.50),
        "p90": _fit_quantile_gbr(Xn, yn, 0.90),
    }

    # --- spike excess models ---
    idx_sp = y > th_spike
    if idx_sp.sum() < 30:
        spike_models = None
        spike_count = int(idx_sp.sum())
    else:
        Xs = X[idx_sp]
        ys = y[idx_sp]
        y_ex = np.log(np.maximum(ys - th_spike, 0.0) + 1.0)
        spike_models = {
            "p50": _fit_quantile_gbr(Xs, y_ex, 0.50),
            "p90": _fit_quantile_gbr(Xs, y_ex, 0.90),
        }
        spike_count = int(idx_sp.sum())

    # --- negative magnitude models ---
    idx_ng = y < th_neg
    if idx_ng.sum() < 30:
        neg_models = None
        neg_count = int(idx_ng.sum())
    else:
        Xg = X[idx_ng]
        yg = y[idx_ng]
        y_mag = np.maximum(-yg, 0.0)  # magnitude
        neg_models = {
            "p50": _fit_quantile_gbr(Xg, y_mag, 0.50),
            "p90": _fit_quantile_gbr(Xg, y_mag, 0.90),
        }
        neg_count = int(idx_ng.sum())

    return {
        "th_spike": float(th_spike),
        "th_neg": float(th_neg),

        "clf_spike": clf_spike,
        "clf_neg": clf_neg,
        "pos_spike": int(pos_spike),
        "pos_neg": int(pos_neg),

        "cal_spike": cal_spike,
        "cal_neg": cal_neg,

        "norm_models": norm_models,
        "spike_models": spike_models,
        "neg_models": neg_models,
        "spike_count": int(spike_count),
        "neg_count": int(neg_count),
    }


# =========================
# Feature frame builders
# =========================

def build_history_feature_frame(history_path: str) -> pd.DataFrame:
    df = read_table(history_path)
    df = add_slot_calendar(df)
    df = add_netload(df, is_history=True)
    to_numeric(df, [COL_RT])
    df = add_within_day_features(df)
    df = add_error_profile_features(df)
    df = add_rt_lag_features(df)
    return df


def select_train_rows(df_feat: pd.DataFrame) -> pd.DataFrame:
    df_feat = df_feat.copy()
    base_b2 = [c for c in FEATURE_B2 if not c.startswith("NetLoad_act_")]
    needed = set(FEATURE_B1 + base_b2 + ["dNetLoad", COL_RT])

    missing = [c for c in needed if c not in df_feat.columns]
    if missing:
        raise KeyError(f"Feature frame missing required columns: {missing}")

    df = df_feat.dropna(subset=list(needed)).copy()
    for c in (FEATURE_B1 + base_b2 + ["dNetLoad", COL_RT]):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(needed)).copy()
    return df


# =========================
# Train / Save
# =========================

def train_and_save(history_path: str, model_dir: str, th_spike: float, th_neg: float) -> None:
    os.makedirs(model_dir, exist_ok=True)

    df_feat = build_history_feature_frame(history_path)
    corr_info = sanity_check_tie_sign(df_feat)

    df_train = select_train_rows(df_feat)

    # --- B1 OOF to create B2 training features (no leakage) ---
    df_oof = b1_oof_predictions(df_train, n_splits=5)

    df_b2 = df_oof.dropna(subset=["dNet_p10_oof", "dNet_p50_oof", "dNet_p90_oof"]).copy()
    df_b2["NetLoad_act_p10"] = df_b2["NetLoad_fc"] + df_b2["dNet_p10_oof"]
    df_b2["NetLoad_act_p50"] = df_b2["NetLoad_fc"] + df_b2["dNet_p50_oof"]
    df_b2["NetLoad_act_p90"] = df_b2["NetLoad_fc"] + df_b2["dNet_p90_oof"]

    # Train final models
    b1_models = train_b1_models(df_train)
    b2_pack = train_b2_regime_models(df_b2, th_spike=th_spike, th_neg=th_neg)

    bundle = {
        "version": "v2.2",
        "th_spike": float(th_spike),
        "th_neg": float(th_neg),
        "feature_b1": FEATURE_B1,
        "feature_b2": FEATURE_B2,
        "b1_models": b1_models,

        "b2_clf_spike": b2_pack["clf_spike"],
        "b2_clf_neg": b2_pack["clf_neg"],
        "b2_cal_spike": b2_pack["cal_spike"],
        "b2_cal_neg": b2_pack["cal_neg"],

        "b2_norm_models": b2_pack["norm_models"],
        "b2_spike_models": b2_pack["spike_models"],
        "b2_neg_models": b2_pack["neg_models"],

        "b2_pos_spike": b2_pack["pos_spike"],
        "b2_pos_neg": b2_pack["pos_neg"],
        "b2_spike_count": b2_pack["spike_count"],
        "b2_neg_count": b2_pack["neg_count"],

        "corr_info": corr_info,
        "history_path": os.path.abspath(history_path),
    }

    model_path = os.path.join(model_dir, "rt_routeB_v2_2.joblib")
    joblib.dump(bundle, model_path)

    meta = {
        "model_path": os.path.abspath(model_path),
        "version": "v2.2",
        "th_spike": float(th_spike),
        "th_neg": float(th_neg),
        "feature_b1": FEATURE_B1,
        "feature_b2": FEATURE_B2,
        "b2_pos_spike": int(b2_pack["pos_spike"]),
        "b2_pos_neg": int(b2_pack["pos_neg"]),
        "b2_spike_count": int(b2_pack["spike_count"]),
        "b2_neg_count": int(b2_pack["neg_count"]),
        "has_cal_spike": bool(b2_pack["cal_spike"] is not None),
        "has_cal_neg": bool(b2_pack["cal_neg"] is not None),
        "corr_info": corr_info,
    }
    meta_path = os.path.join(model_dir, "rt_routeB_v2_2_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved model bundle: {os.path.abspath(model_path)}")
    print(f"[OK] Saved metadata:     {os.path.abspath(meta_path)}")
    print(f"[INFO] History rows used: {len(df_train):,}  days={df_train['date_dt'].nunique():,}")
    print(f"[INFO] Spike: RT>{th_spike}  pos={b2_pack['pos_spike']}  reg_rows={b2_pack['spike_count']}  cal={meta['has_cal_spike']}")
    print(f"[INFO] Neg:   RT<{th_neg}   pos={b2_pack['pos_neg']}   reg_rows={b2_pack['neg_count']}   cal={meta['has_cal_neg']}")
    print(f"[INFO] Sanity corr: {corr_info}")


# =========================
# Prediction helpers
# =========================

def compute_error_profile_for_date(df_hist: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    hist = df_hist[df_hist["date_dt"] < target_date].copy()
    hist = hist.dropna(subset=["slot_id", "dNetLoad"]).copy()
    hist = hist.sort_values(["date_dt", "slot_id"])

    out = []
    for sid, g in hist.groupby("slot_id"):
        s = g["dNetLoad"].astype(float).values
        err_1 = s[-1] if len(s) >= 1 else np.nan
        last7 = s[-7:] if len(s) >= 7 else s
        last14 = s[-14:] if len(s) >= 14 else s

        row = {
            "slot_id": sid,
            "Err_1": float(err_1) if len(s) >= 1 else np.nan,
            "Err_mean_7": float(np.nanmean(last7)) if len(last7) >= 5 else np.nan,
            "Err_absmean_7": float(np.nanmean(np.abs(last7))) if len(last7) >= 5 else np.nan,
            "Err_p10_14": float(np.nanquantile(last14, 0.10)) if len(last14) >= 10 else np.nan,
            "Err_p90_14": float(np.nanquantile(last14, 0.90)) if len(last14) >= 10 else np.nan,
        }
        out.append(row)

    return pd.DataFrame(out)


def compute_rt_state_for_date(df_hist: pd.DataFrame, target_date: pd.Timestamp) -> Tuple[pd.DataFrame, float]:
    hist = df_hist[df_hist["date_dt"] < target_date].copy()
    hist = hist.dropna(subset=["slot_id", COL_RT]).copy()
    hist = hist.sort_values(["date_dt", "slot_id"])
    hist[COL_RT] = pd.to_numeric(hist[COL_RT], errors="coerce")

    rt_mean7 = []
    for sid, g in hist.groupby("slot_id"):
        s = g[COL_RT].astype(float).values
        last7 = s[-7:] if len(s) >= 7 else s
        rt_mean7.append({"slot_id": sid, "RT_mean_7": float(np.nanmean(last7)) if len(last7) >= 3 else np.nan})
    rt_mean7 = pd.DataFrame(rt_mean7)
    # Robust: ensure slot_id exists even when history is empty (e.g., very early dates)
    if rt_mean7.empty or ("slot_id" not in rt_mean7.columns):
        rt_mean7 = pd.DataFrame({"slot_id": list(range(1, 97)), "RT_mean_7": [np.nan]*96})
    else:
        # Ensure full 96 slots present
        rt_mean7 = rt_mean7.merge(pd.DataFrame({"slot_id": list(range(1, 97))}), on="slot_id", how="right")

    d1 = target_date - pd.Timedelta(days=1)
    d7 = target_date - pd.Timedelta(days=7)

    rt_d1 = hist[hist["date_dt"] == d1][["slot_id", COL_RT]].rename(columns={COL_RT: "RT_lag_1d"})
    rt_d7 = hist[hist["date_dt"] == d7][["slot_id", COL_RT]].rename(columns={COL_RT: "RT_lag_7d"})

    state = rt_mean7.merge(rt_d1, on="slot_id", how="left").merge(rt_d7, on="slot_id", how="left")
    state["RT_lag_1d"] = state["RT_lag_1d"].fillna(state["RT_mean_7"])
    state["RT_lag_7d"] = state["RT_lag_7d"].fillna(state["RT_mean_7"])

    daily_std = hist.groupby("date_dt")[COL_RT].std().sort_index()
    vol_1d = float(daily_std.get(d1, np.nan))
    if np.isnan(vol_1d):
        last7days = daily_std[daily_std.index < target_date].tail(7).values
        vol_1d = float(np.nanmean(last7days)) if len(last7days) >= 3 else np.nan

    return state[["slot_id", "RT_lag_1d", "RT_lag_7d"]], vol_1d


def _enforce_monotonic(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.vstack([p10, p50, p90]).T
    a.sort(axis=1)
    return a[:, 0], a[:, 1], a[:, 2]


def _renorm_probs(p_neg: np.ndarray, p_spike: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create mutually-consistent regime weights:
      p_norm = max(0, 1 - p_neg - p_spike), then normalize to sum=1.
    """
    p_neg = np.clip(p_neg, 0.0, 1.0)
    p_spike = np.clip(p_spike, 0.0, 1.0)
    p_norm = np.maximum(0.0, 1.0 - p_neg - p_spike)
    s = p_neg + p_spike + p_norm
    s = np.where(s <= 0, 1.0, s)
    return p_norm / s, p_spike / s, p_neg / s



def select_spike_budget_trigger(
    p_spike: np.ndarray,
    netload_fc: np.ndarray,
    pv_fc: np.ndarray,
    budget_pts: int = SPIKE_BUDGET_PTS_DEFAULT,
    seeds: int = SPIKE_BUDGET_SEEDS_DEFAULT,
    block_radius: int = SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT,
    min_gap: int = SPIKE_BUDGET_MIN_GAP_DEFAULT,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Budget-based spike trigger (v2.5).

    Picks a small number of high-risk 15-min slots per day using a greedy seed selection, expands each seed
    into a contiguous block (radius), and caps total selected points by budget_pts.

    Returns:
      selected: bool array (len=n_slots)
      intervals: list of (start_slot_id, end_slot_id) in 1..96 inclusive
    """
    n = len(p_spike)
    if n == 0:
        return np.zeros(0, dtype=bool), []

    budget_pts = int(max(0, budget_pts))
    seeds = int(max(0, seeds))
    block_radius = int(max(0, block_radius))
    min_gap = int(max(0, min_gap))

    if budget_pts <= 0 or seeds <= 0:
        return np.zeros(n, dtype=bool), []

    nl = np.asarray(netload_fc, dtype=float)
    pv = np.asarray(pv_fc, dtype=float)
    ramp_nl = np.abs(np.diff(nl, prepend=nl[0]))
    ramp_pv = np.abs(np.diff(pv, prepend=pv[0]))

    base = np.asarray(p_spike, dtype=float).copy()

    def _rank01(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
        return r

    # main prob + small rank-based tie breakers (help when p_spike is quantized)
    score = base + 1e-4 * _rank01(ramp_nl) + 1e-4 * _rank01(ramp_pv)

    idx_sorted = np.argsort(score)[::-1]
    seed_idx: List[int] = []
    for i in idx_sorted:
        if len(seed_idx) >= seeds:
            break
        if all(abs(int(i) - int(j)) > min_gap for j in seed_idx):
            seed_idx.append(int(i))

    if not seed_idx:
        return np.zeros(n, dtype=bool), []

    cand: set[int] = set()
    for s in seed_idx:
        lo = max(0, s - block_radius)
        hi = min(n - 1, s + block_radius)
        for j in range(lo, hi + 1):
            cand.add(int(j))

    cand = sorted(cand)
    if len(cand) > budget_pts:
        cand = sorted(cand, key=lambda k: score[k], reverse=True)[:budget_pts]
        cand = sorted(cand)

    selected = np.zeros(n, dtype=bool)
    selected[cand] = True

    intervals: List[Tuple[int, int]] = []
    if cand:
        start = prev = cand[0]
        for x in cand[1:]:
            if x == prev + 1:
                prev = x
                continue
            intervals.append((start + 1, prev + 1))
            start = prev = x
        intervals.append((start + 1, prev + 1))

    return selected, intervals


def predict_one_day(
    history_path: str,
    model_dir: str,
    forecast_path: str,
    date_str: str,
    out_path: str,
    p_gate_spike: float = P_GATE_SPIKE_DEFAULT,
    p_gate_neg: float = P_GATE_NEG_DEFAULT,
    p_full_neg: float = P_FULL_NEG_DEFAULT,
    arm_day_neg: float = ARM_DAY_NEG_DEFAULT,
    arm_day_neg_pv_th: float = ARM_DAY_NEG_PV_TH_DEFAULT,
    arm_day_neg_netload_min_th: float = ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT,
    arm_day_neg_use_or: bool = ARM_DAY_NEG_USE_OR_DEFAULT,

    # v2.5 spike budget trigger
    spike_budget_enable: bool = SPIKE_BUDGET_ENABLE_DEFAULT,
    spike_budget_pts: int = SPIKE_BUDGET_PTS_DEFAULT,
    spike_budget_seeds: int = SPIKE_BUDGET_SEEDS_DEFAULT,
    spike_budget_block_radius: int = SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT,
    spike_budget_min_gap: int = SPIKE_BUDGET_MIN_GAP_DEFAULT,
    spike_apply_to_p50: bool = SPIKE_APPLY_TO_P50_DEFAULT,
) -> None:
    bundle_path = os.path.join(model_dir, "rt_routeB_v2_2.joblib")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}. Run train first.")

    bundle = joblib.load(bundle_path)
    th_spike = float(bundle["th_spike"])
    th_neg = float(bundle["th_neg"])

    b1_models = bundle["b1_models"]

    clf_spike = bundle["b2_clf_spike"]
    clf_neg = bundle["b2_clf_neg"]
    cal_spike = bundle.get("b2_cal_spike", None)
    cal_neg = bundle.get("b2_cal_neg", None)

    norm_models = bundle["b2_norm_models"]
    spike_models = bundle.get("b2_spike_models", None)
    neg_models = bundle.get("b2_neg_models", None)

    target_date = pd.to_datetime(date_str)

    # Load history for profiles/lags (strictly before target_date)
    df_hist = build_history_feature_frame(history_path)
    df_hist = df_hist[df_hist["date_dt"] < target_date].copy()

    # Load forecast input (must contain 96 rows for the target date)
    df_fc = read_table(forecast_path)
    df_fc = add_slot_calendar(df_fc)
    df_fc = df_fc[df_fc["date_dt"] == target_date].copy()

    if df_fc.empty:
        raise ValueError(f"No rows for target date {date_str} found in forecast file.")
    if df_fc["slot_id"].nunique() != 96:
        raise ValueError(f"Forecast must contain 96 unique slots for {date_str}, got {df_fc['slot_id'].nunique()}.")

    df_fc = add_netload(df_fc, is_history=False)
    df_fc = add_within_day_features(df_fc)

    # Day-level context for negative-price arming guardrails
    pv_fc_max_day = float(np.nanmax(pd.to_numeric(df_fc["PV_fc"], errors="coerce").values))
    netload_fc_min_day = float(np.nanmin(pd.to_numeric(df_fc["NetLoad_fc"], errors="coerce").values))


    # error profile and RT state from history
    err = compute_error_profile_for_date(df_hist, target_date)
    rt_state, rt_vol_1d = compute_rt_state_for_date(df_hist, target_date)

    df_fc = df_fc.merge(err, on="slot_id", how="left").merge(rt_state, on="slot_id", how="left")
    df_fc["RT_vol_1d"] = rt_vol_1d

    # fill missing profile stats with neutral 0.0 (avoid losing slots)
    err_cols = ["Err_1", "Err_mean_7", "Err_absmean_7", "Err_p10_14", "Err_p90_14"]
    for c in err_cols:
        if c in df_fc.columns:
            df_fc[c] = pd.to_numeric(df_fc[c], errors="coerce").fillna(0.0)

    for c in ["RT_lag_1d", "RT_lag_7d", "RT_vol_1d"]:
        if c in df_fc.columns:
            df_fc[c] = pd.to_numeric(df_fc[c], errors="coerce")

    # --- B1 predict dNetLoad quantiles ---
    df_fc = df_fc.dropna(subset=FEATURE_B1).copy()
    X1 = df_fc[FEATURE_B1].astype(float).values
    d10 = b1_models["p10"].predict(X1)
    d50 = b1_models["p50"].predict(X1)
    d90 = b1_models["p90"].predict(X1)

    df_fc["NetLoad_act_p10"] = df_fc["NetLoad_fc"] + d10
    df_fc["NetLoad_act_p50"] = df_fc["NetLoad_fc"] + d50
    df_fc["NetLoad_act_p90"] = df_fc["NetLoad_fc"] + d90

    # --- B2 predict regimes ---
    df_fc = df_fc.dropna(subset=FEATURE_B2).copy()
    X2 = df_fc[FEATURE_B2].astype(float).values

    p_spike_raw = _clf_predict_proba(clf_spike, X2)
    p_neg_raw = _clf_predict_proba(clf_neg, X2)

    p_spike = _apply_calibrator(cal_spike, p_spike_raw)
    p_neg = _apply_calibrator(cal_neg, p_neg_raw)

    p_norm, p_spike, p_neg = _renorm_probs(p_neg, p_spike)

    # normal scenario
    rtN10 = norm_models["p10"].predict(X2)
    rtN50 = norm_models["p50"].predict(X2)
    rtN90 = norm_models["p90"].predict(X2)
    rtN10, rtN50, rtN90 = _enforce_monotonic(rtN10, rtN50, rtN90)

    # spike scenario
    if spike_models is None:
        rtS50 = np.full(len(X2), np.nan)
        rtS90 = np.full(len(X2), np.nan)
    else:
        ex50 = spike_models["p50"].predict(X2)
        ex90 = spike_models["p90"].predict(X2)
        rtS50 = (th_spike - 1.0) + np.exp(ex50)
        rtS90 = (th_spike - 1.0) + np.exp(ex90)
        rtS50, rtS90 = np.minimum(rtS50, rtS90), np.maximum(rtS50, rtS90)

    # negative scenario
    if neg_models is None:
        rtG50 = np.full(len(X2), np.nan)
        rtG90 = np.full(len(X2), np.nan)
    else:
        mag50 = neg_models["p50"].predict(X2)
        mag90 = neg_models["p90"].predict(X2)
        # p90 magnitude -> more negative
        rtG50 = -np.maximum(mag50, 0.0)
        rtG90 = -np.maximum(mag90, 0.0)
        # ensure rtG90 <= rtG50 (more negative)
        rtG90, rtG50 = np.minimum(rtG90, rtG50), np.maximum(rtG90, rtG50)

    # mixture outputs
    rt_mix_p50 = p_norm * rtN50 + p_spike * rtS50 + p_neg * rtG50
    rt_mix_upper90 = p_norm * rtN90 + p_spike * rtS90 + p_neg * rtG50  # neg doesn't push upper
    rt_mix_lower10 = p_norm * rtN10 + p_spike * rtS50 + p_neg * rtG90  # neg tail affects downside

    # Day-level arming for negative gate: reduce false triggering on "non-negative" days.
    # Condition (when arm_day_neg>0):
    #   (max_t p_neg(t) >= arm_day_neg) AND (PV_fc_max_day >= arm_day_neg_pv_th  OR/AND  NetLoad_fc_min_day <= arm_day_neg_netload_min_th)
    pv_ok = bool(np.isfinite(pv_fc_max_day) and (pv_fc_max_day >= float(arm_day_neg_pv_th)))
    nl_ok = bool(np.isfinite(netload_fc_min_day) and (netload_fc_min_day <= float(arm_day_neg_netload_min_th)))
    neg_guard_ok = (pv_ok or nl_ok) if bool(arm_day_neg_use_or) else (pv_ok and nl_ok)

    neg_prob_ok = True
    armed_neg = True
    if arm_day_neg is not None and float(arm_day_neg) > 0:
        neg_prob_ok = bool(np.nanmax(p_neg) >= float(arm_day_neg))
        armed_neg = bool(neg_prob_ok and neg_guard_ok)

    p_neg_eff = p_neg if armed_neg else np.zeros_like(p_neg)

    # gate outputs (priority: neg -> spike -> normal)
    take_neg = p_neg_eff >= p_gate_neg

    # v2.5 spike budget trigger: select a small number of high-risk slots per day.
    spike_selected: np.ndarray
    spike_intervals: List[Tuple[int, int]] = []
    spike_budget_used = False
    if bool(spike_budget_enable) and int(spike_budget_pts) > 0 and int(spike_budget_seeds) > 0:
        spike_selected, spike_intervals = select_spike_budget_trigger(
            p_spike=p_spike,
            netload_fc=df_fc["NetLoad_fc"].values,
            pv_fc=df_fc["PV_fc"].values,
            budget_pts=int(spike_budget_pts),
            seeds=int(spike_budget_seeds),
            block_radius=int(spike_budget_block_radius),
            min_gap=int(spike_budget_min_gap),
        )
        spike_budget_used = True
    else:
        # fallback to legacy absolute-probability threshold (often too strict if p_spike is not calibrated)
        spike_selected = (p_spike >= p_gate_spike)

    take_spike = (~take_neg) & spike_selected
    take_spike_p50 = take_spike if bool(spike_apply_to_p50) else np.zeros_like(take_spike, dtype=bool)

    if p_full_neg is None or p_full_neg <= p_gate_neg:
        # hard gate
        rt_gate_p50 = np.where(take_neg, rtG50, np.where(take_spike_p50, rtS50, rtN50))
        rt_gate_upper90 = np.where(take_neg, rtN90, np.where(take_spike, rtS90, rtN90))  # upper risk ignores neg
        rt_gate_lower10 = np.where(take_neg, rtG90, np.where(take_spike, rtS50, rtN10))
    else:
        # ramp gate: blend between normal and negative as p_neg rises from p_gate_neg to p_full_neg
        denom = max(p_full_neg - p_gate_neg, 1e-12)
        w_neg = np.clip((p_neg_eff - p_gate_neg) / denom, 0.0, 1.0)

        # base outputs: spike if not neg, else normal
        base50 = np.where(take_spike_p50, rtS50, rtN50)
        base50 = np.where(take_neg, rtN50, base50)
        rt_gate_p50 = np.where(take_neg, base50 + w_neg * (rtG50 - base50), base50)

        baseU = np.where(take_spike, rtS90, rtN90)
        rt_gate_upper90 = np.where(take_neg, rtN90, baseU)  # upper risk still ignores neg

        baseL = np.where(take_spike, rtS50, rtN10)
        baseL = np.where(take_neg, rtN10, baseL)
        rt_gate_lower10 = np.where(take_neg, baseL + w_neg * (rtG90 - baseL), baseL)
    # v2.5: explicit spike scenario (budget-triggered)
    rt_spike_budget_p50 = np.where(take_spike, rtS50, rt_gate_p50)
    rt_spike_budget_upper90 = np.where(take_spike, rtS90, rt_gate_upper90)

    df_out = pd.DataFrame({
        "日期": df_fc[COL_DATE].values,
        "时刻": df_fc["time_str"].values,
        "slot_id": df_fc["slot_id"].astype(int).values,

        "p_norm": p_norm,
        "p_spike": p_spike,
        "p_neg": p_neg,
        "spike_budget_used": np.full_like(p_neg, int(spike_budget_used), dtype=int),
        "spike_selected": take_spike.astype(int),
        "spike_budget_pts": np.full_like(p_neg, int(spike_budget_pts), dtype=int),
        "spike_budget_seeds": np.full_like(p_neg, int(spike_budget_seeds), dtype=int),
        "spike_budget_block_radius": np.full_like(p_neg, int(spike_budget_block_radius), dtype=int),
        "spike_budget_min_gap": np.full_like(p_neg, int(spike_budget_min_gap), dtype=int),
        "spike_apply_to_p50": np.full_like(p_neg, int(bool(spike_apply_to_p50)), dtype=int),
        "neg_armed": np.full_like(p_neg, int(armed_neg), dtype=int),
        "neg_prob_ok": np.full_like(p_neg, int(neg_prob_ok), dtype=int),
        "neg_guard_ok": np.full_like(p_neg, int(neg_guard_ok), dtype=int),
        "neg_guard_pv_ok": np.full_like(p_neg, int(pv_ok), dtype=int),
        "neg_guard_netload_ok": np.full_like(p_neg, int(nl_ok), dtype=int),
        "PV_fc_max_day": np.full_like(p_neg, pv_fc_max_day, dtype=float),
        "NetLoad_fc_min_day": np.full_like(p_neg, netload_fc_min_day, dtype=float),
        "arm_day_neg_pv_th": np.full_like(p_neg, float(arm_day_neg_pv_th), dtype=float),
        "arm_day_neg_netload_min_th": np.full_like(p_neg, float(arm_day_neg_netload_min_th), dtype=float),
        "arm_day_neg_use_or": np.full_like(p_neg, int(bool(arm_day_neg_use_or)), dtype=int),
        f"P(RT>{int(th_spike)})": p_spike,
        f"P(RT<{th_neg:g})": p_neg,

        "RT_norm_p10": rtN10,
        "RT_norm_p50": rtN50,
        "RT_norm_p90": rtN90,

        "RT_spike_p50": rtS50,
        "RT_spike_p90": rtS90,

        "RT_neg_p50": rtG50,
        "RT_neg_p90": rtG90,

        "RT_mix_p50": rt_mix_p50,
        "RT_mix_upper90": rt_mix_upper90,
        "RT_mix_lower10": rt_mix_lower10,

        "RT_gate_p50": rt_gate_p50,
        "RT_gate_upper90": rt_gate_upper90,
        "RT_gate_lower10": rt_gate_lower10,
        "RT_spike_budget_p50": rt_spike_budget_p50,
        "RT_spike_budget_upper90": rt_spike_budget_upper90,

        # debug columns
        "NetLoad_fc": df_fc["NetLoad_fc"].values,
        "NetLoad_act_p50": df_fc["NetLoad_act_p50"].values,
        "PV_fc": df_fc["PV_fc"].values,
        "Wind_fc": df_fc["Wind_fc"].values,
        "Tie_fc": df_fc["Tie_fc"].values,
        "RT_lag_1d": df_fc["RT_lag_1d"].values,
        "RT_vol_1d": df_fc["RT_vol_1d"].values,
    }).sort_values("slot_id")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="pred", index=False)

    print(f"[OK] Saved prediction: {os.path.abspath(out_path)}")


# =========================
# Backtest (evaluation)
# =========================

def _safe_auc(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    return (
        float(roc_auc_score(y, p)),
        float(average_precision_score(y, p)),
        float(brier_score_loss(y, p)),
    )


def backtest_dates(
    history_path: str,
    model_dir: str,
    dates: List[str],
    out_path: str,
    p_gate_spike: float = P_GATE_SPIKE_DEFAULT,
    p_gate_neg: float = P_GATE_NEG_DEFAULT,
    p_full_neg: float = P_FULL_NEG_DEFAULT,
    arm_day_neg: float = ARM_DAY_NEG_DEFAULT,
    arm_day_neg_pv_th: float = ARM_DAY_NEG_PV_TH_DEFAULT,
    arm_day_neg_netload_min_th: float = ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT,
    arm_day_neg_use_or: bool = ARM_DAY_NEG_USE_OR_DEFAULT,

    # v2.5 spike budget trigger
    spike_budget_enable: bool = SPIKE_BUDGET_ENABLE_DEFAULT,
    spike_budget_pts: int = SPIKE_BUDGET_PTS_DEFAULT,
    spike_budget_seeds: int = SPIKE_BUDGET_SEEDS_DEFAULT,
    spike_budget_block_radius: int = SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT,
    spike_budget_min_gap: int = SPIKE_BUDGET_MIN_GAP_DEFAULT,
    spike_apply_to_p50: bool = SPIKE_APPLY_TO_P50_DEFAULT,
) -> None:
    """
    Backtest using forecast columns from history itself:
      For each date D, use D's forecast columns + flags as forecast input and predict D.
    """
    hist = read_table(history_path)
    hist = add_slot_calendar(hist)
    require_cols(hist, [COL_LOAD_FC, COL_TIE_FC, COL_WIND_FC, COL_PV_FC, COL_RT], where="history backtest base")
    hist[COL_RT] = pd.to_numeric(hist[COL_RT], errors="coerce")
    hist["时刻"] = hist["time_str"]

    # Load thresholds from trained model (align backtest labels with training thresholds)
    th_spike = SPIKE_THRESHOLD_RT_DEFAULT
    th_neg = NEG_THRESHOLD_RT_DEFAULT
    bundle_path = os.path.join(model_dir, "rt_routeB_v2_2.joblib")
    if os.path.exists(bundle_path):
        try:
            _bundle = joblib.load(bundle_path)
            th_spike = float(_bundle.get("th_spike", th_spike))
            th_neg = float(_bundle.get("th_neg", th_neg))
        except Exception:
            pass


    per_day_rows = []
    details = {}

    for d in dates:
        dts = pd.to_datetime(d)
        df_day = hist[hist["date_dt"] == dts].copy()
        if df_day.empty:
            print(f"[WARN] date not found in history: {d}")
            continue

        df_fc = df_day[[COL_DATE, "时刻", COL_HOL, COL_WKND, COL_LOAD_FC, COL_TIE_FC, COL_WIND_FC, COL_PV_FC]].copy()

        tmp_fc_path = os.path.join(os.path.dirname(out_path) or ".", f"__tmp_forecast_{d}.xlsx")
        with pd.ExcelWriter(tmp_fc_path, engine="openpyxl") as w:
            df_fc.to_excel(w, sheet_name="forecast", index=False)

        tmp_pred_path = os.path.join(os.path.dirname(out_path) or ".", f"__tmp_pred_{d}.xlsx")
        predict_one_day(
            history_path,
            model_dir,
            tmp_fc_path,
            d,
            tmp_pred_path,
            p_gate_spike=p_gate_spike,
            p_gate_neg=p_gate_neg,
            p_full_neg=p_full_neg,
            arm_day_neg=arm_day_neg,
            arm_day_neg_pv_th=arm_day_neg_pv_th,
            arm_day_neg_netload_min_th=arm_day_neg_netload_min_th,
            arm_day_neg_use_or=arm_day_neg_use_or,
            spike_budget_enable=spike_budget_enable,
            spike_budget_pts=spike_budget_pts,
            spike_budget_seeds=spike_budget_seeds,
            spike_budget_block_radius=spike_budget_block_radius,
            spike_budget_min_gap=spike_budget_min_gap,
            spike_apply_to_p50=spike_apply_to_p50,
        )

        df_pred = pd.read_excel(tmp_pred_path, sheet_name="pred")
        df_pred["时刻"] = df_pred["时刻"].map(normalize_time_str)

        df_act = df_day[["时刻", COL_RT]].rename(columns={COL_RT: "RT_true"})
        df = df_pred.merge(df_act, on="时刻", how="left").dropna(subset=["RT_true"])

        y = df["RT_true"].astype(float).values
        yhat_mix = pd.to_numeric(df["RT_mix_p50"], errors="coerce").values
        yhat_gate = pd.to_numeric(df["RT_gate_p50"], errors="coerce").values
        yhat_spike_budget = pd.to_numeric(df.get("RT_spike_budget_p50"), errors="coerce").values

        mae_mix = float(mean_absolute_error(y, yhat_mix))
        rmse_mix = float(np.sqrt(mean_squared_error(y, yhat_mix)))
        mae_gate = float(mean_absolute_error(y, yhat_gate))
        rmse_gate = float(np.sqrt(mean_squared_error(y, yhat_gate)))

        if "RT_spike_budget_p50" in df.columns:
            mae_spike_budget = float(mean_absolute_error(y, yhat_spike_budget))
            rmse_spike_budget = float(np.sqrt(mean_squared_error(y, yhat_spike_budget)))
        else:
            mae_spike_budget = np.nan
            rmse_spike_budget = np.nan

        spike_selected_n = int(pd.to_numeric(df.get("spike_selected"), errors="coerce").fillna(0).astype(int).sum()) if "spike_selected" in df.columns else 0

        # event metrics
        p_spike = pd.to_numeric(df["p_spike"], errors="coerce").values
        p_neg = pd.to_numeric(df["p_neg"], errors="coerce").values

        is_spike = (y > th_spike).astype(int)
        is_neg = (y < th_neg).astype(int)

        auc_sp, ap_sp, brier_sp = _safe_auc(is_spike, p_spike)
        auc_ng, ap_ng, brier_ng = _safe_auc(is_neg, p_neg)

        cov_norm = float(((y >= df["RT_norm_p10"]) & (y <= df["RT_norm_p90"])).mean())

        per_day_rows.append({
            "date": d,
            "RT_true_min": float(np.nanmin(y)),
            "RT_true_max": float(np.nanmax(y)),
            "neg_points_true": int(is_neg.sum()),
            "spike_points_true": int(is_spike.sum()),

            "MAE_mix_p50": mae_mix,
            "RMSE_mix_p50": rmse_mix,
            "MAE_gate_p50": mae_gate,
            "RMSE_gate_p50": rmse_gate,
            "MAE_spikeBudget_p50": mae_spike_budget,
            "RMSE_spikeBudget_p50": rmse_spike_budget,
            "spike_selected_n": spike_selected_n,

            "Coverage_norm_p10_p90": cov_norm,

            "AUC_spike": auc_sp,
            "AP_spike": ap_sp,
            "Brier_spike": brier_sp,

            "AUC_neg": auc_ng,
            "AP_neg": ap_ng,
            "Brier_neg": brier_ng,

            "p_spike_max": float(np.nanmax(p_spike)),
            "p_neg_max": float(np.nanmax(p_neg)),
        })

        details[d] = df

        try:
            os.remove(tmp_fc_path)
            os.remove(tmp_pred_path)
        except Exception:
            pass

    df_sum = pd.DataFrame(per_day_rows).sort_values("date")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_sum.to_excel(w, sheet_name="per_day", index=False)
        for d, df in details.items():
            sheet = d.replace("-", "")[:31]
            df.to_excel(w, sheet_name=sheet, index=False)

        # --- run_info: capture parameters & date coverage to avoid confusion when reusing filenames ---
        try:
            run_info = {
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "history_path": os.path.abspath(history_path),
                "model_dir": os.path.abspath(model_dir),
                "bundle_path": os.path.abspath(os.path.join(model_dir, "rt_routeB_v2_2.joblib")),
                "th_spike": float(th_spike),
                "th_neg": float(th_neg),
                "p_gate_spike": float(p_gate_spike),
                "p_gate_neg": float(p_gate_neg),
                "p_full_neg": float(p_full_neg),
                "spike_budget_enable": int(bool(spike_budget_enable)),
                "spike_budget_pts": int(spike_budget_pts),
                "spike_budget_seeds": int(spike_budget_seeds),
                "spike_budget_block_radius": int(spike_budget_block_radius),
                "spike_budget_min_gap": int(spike_budget_min_gap),
                "spike_apply_to_p50": int(bool(spike_apply_to_p50)),
                "arm_day_neg": float(arm_day_neg) if arm_day_neg is not None else np.nan,
                "arm_day_neg_pv_th": float(arm_day_neg_pv_th),
                "arm_day_neg_netload_min_th": float(arm_day_neg_netload_min_th),
                "arm_day_neg_use_or": int(bool(arm_day_neg_use_or)),
                "dates_requested": ",".join([str(x) for x in dates]),
                "dates_requested_n": int(len(dates)),
                "dates_processed": ",".join(list(details.keys())),
                "dates_processed_n": int(len(details)),
            }
            missing = [d for d in dates if d not in details]
            run_info["dates_missing"] = ",".join(missing)
            run_info["dates_missing_n"] = int(len(missing))
            df_run = pd.DataFrame(list(run_info.items()), columns=["key", "value"])
            df_run.to_excel(w, sheet_name="run_info", index=False)
        except Exception:
            pass

    print(f"[OK] Saved backtest report: {os.path.abspath(out_path)}")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train Route-B v2.2 models and save to model_dir")
    p_train.add_argument("--history", required=True, help="History CSV/XLSX with full columns")
    p_train.add_argument("--model_dir", required=True, help="Directory to save models")
    p_train.add_argument("--th_spike", type=float, default=SPIKE_THRESHOLD_RT_DEFAULT, help="Spike threshold (default 800)")
    p_train.add_argument("--th_neg", type=float, default=NEG_THRESHOLD_RT_DEFAULT, help="Negative threshold (default 0)")

    p_pred = sub.add_parser("predict", help="Predict RT for one target date (D+1)")
    p_pred.add_argument("--history", required=True, help="History CSV/XLSX with full columns")
    p_pred.add_argument("--model_dir", required=True, help="Directory containing saved models")
    p_pred.add_argument("--forecast", required=True, help="Forecast CSV/XLSX for D+1 (96 rows for target date)")
    p_pred.add_argument("--date", required=True, help="Target date, e.g. 2026-01-01")
    p_pred.add_argument("--out", required=True, help="Output Excel path")
    p_pred.add_argument("--p_gate_spike", type=float, default=P_GATE_SPIKE_DEFAULT, help="Gate threshold for spike")
    p_pred.add_argument("--p_gate_neg", type=float, default=P_GATE_NEG_DEFAULT, help="Gate threshold for negative")
    p_pred.add_argument("--p_full_neg", type=float, default=P_FULL_NEG_DEFAULT, help="Ramp gate: full negative weight threshold (<= p_gate_neg disables ramp)")
    p_pred.add_argument("--arm_day_neg", type=float, default=ARM_DAY_NEG_DEFAULT, help="Day-level arming threshold for negative gate (<=0 disables)")
    p_pred.add_argument("--arm_day_neg_pv_th", type=float, default=ARM_DAY_NEG_PV_TH_DEFAULT, help="PV_fc daily max threshold for negative-day arming guard")
    p_pred.add_argument("--arm_day_neg_netload_min_th", type=float, default=ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT, help="NetLoad_fc daily min threshold for negative-day arming guard")
    p_pred.add_argument("--arm_day_neg_use_or", type=int, default=1, help="Use OR guard (1) or AND guard (0) for (PV vs NetLoad)")

    # v2.5 spike budget trigger
    p_pred.add_argument("--spike_budget_enable", type=int, default=int(SPIKE_BUDGET_ENABLE_DEFAULT), help="Enable spike budget trigger (1/0)")
    p_pred.add_argument("--spike_budget_pts", type=int, default=SPIKE_BUDGET_PTS_DEFAULT, help="Max number of 15-min slots flagged as spike per day")
    p_pred.add_argument("--spike_budget_seeds", type=int, default=SPIKE_BUDGET_SEEDS_DEFAULT, help="Number of spike seed slots before expansion")
    p_pred.add_argument("--spike_budget_block_radius", type=int, default=SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT, help="Expand each seed by +/- radius (15-min slots)")
    p_pred.add_argument("--spike_budget_min_gap", type=int, default=SPIKE_BUDGET_MIN_GAP_DEFAULT, help="Minimum separation between spike seeds (15-min slots)")
    p_pred.add_argument("--spike_apply_to_p50", type=int, default=int(SPIKE_APPLY_TO_P50_DEFAULT), help="If 1, spike budget affects RT_gate_p50; if 0, only affects risk bounds & scenario columns")

    p_bt = sub.add_parser("backtest", help="Backtest a list of dates using forecast columns from history")
    p_bt.add_argument("--history", required=True, help="History CSV/XLSX with full columns")
    p_bt.add_argument("--model_dir", required=True, help="Directory containing saved models")
    p_bt.add_argument("--dates", required=True, help="Comma-separated dates, e.g. 2025-11-08,2025-01-26")
    p_bt.add_argument("--out", required=True, help="Output Excel report path")
    p_bt.add_argument("--p_gate_spike", type=float, default=P_GATE_SPIKE_DEFAULT, help="Gate threshold for spike")
    p_bt.add_argument("--p_gate_neg", type=float, default=P_GATE_NEG_DEFAULT, help="Gate threshold for negative")
    p_bt.add_argument("--arm_day_neg", type=float, default=ARM_DAY_NEG_DEFAULT, help="Day-level arming threshold for negative gate (<=0 disables)")
    p_bt.add_argument("--arm_day_neg_pv_th", type=float, default=ARM_DAY_NEG_PV_TH_DEFAULT, help="PV_fc daily max threshold for negative-day arming guard")
    p_bt.add_argument("--arm_day_neg_netload_min_th", type=float, default=ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT, help="NetLoad_fc daily min threshold for negative-day arming guard")
    p_bt.add_argument("--arm_day_neg_use_or", type=int, default=1, help="Use OR guard (1) or AND guard (0) for (PV vs NetLoad)")

    # v2.5 spike budget trigger
    p_bt.add_argument("--spike_budget_enable", type=int, default=int(SPIKE_BUDGET_ENABLE_DEFAULT), help="Enable spike budget trigger (1/0)")
    p_bt.add_argument("--spike_budget_pts", type=int, default=SPIKE_BUDGET_PTS_DEFAULT, help="Max number of 15-min slots flagged as spike per day")
    p_bt.add_argument("--spike_budget_seeds", type=int, default=SPIKE_BUDGET_SEEDS_DEFAULT, help="Number of spike seed slots before expansion")
    p_bt.add_argument("--spike_budget_block_radius", type=int, default=SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT, help="Expand each seed by +/- radius (15-min slots)")
    p_bt.add_argument("--spike_budget_min_gap", type=int, default=SPIKE_BUDGET_MIN_GAP_DEFAULT, help="Minimum separation between spike seeds (15-min slots)")
    p_bt.add_argument("--spike_apply_to_p50", type=int, default=int(SPIKE_APPLY_TO_P50_DEFAULT), help="If 1, spike budget affects RT_gate_p50; if 0, only affects risk bounds & scenario columns")

    return p.parse_args()


# =========================
# Optional: run without args (edit here)
# =========================

def run_without_args() -> None:
    CONFIG = {
        "cmd": "backtest",  # "train" | "predict" | "backtest"
        "history": r"data/价格预测数据集.csv",
        "model_dir": r"models_v2_2",
        "th_spike": 800.0,
        "th_neg": 0.0,

        # gate thresholds
        "p_gate_spike": 0.15,
        "p_gate_neg": 0.30,  # tune here or pass --p_gate_neg
        "p_full_neg": 0.50,  # ramp gate full neg prob (<=p_gate_neg => hard gate)

        # v2.5 spike budget trigger
        "spike_budget_enable": 1,
        "spike_budget_pts": 6,
        "spike_budget_seeds": 2,
        "spike_budget_block_radius": 1,
        "spike_budget_min_gap": 4,
        "spike_apply_to_p50": 0,  # 0=稳健(不改p50); 1=让尖峰预算影响RT_gate_p50

        "arm_day_neg": 0.40,
        # predict
        "forecast": r"forecast_input.xlsx",
        "date": "2025-11-08",
        "out": r"outPut/forcast_2025-11-08_v2_2.xlsx",

        # backtest
        "dates": "2025-05-18, 2025-02-15, 2025-03-25, 2025-03-24, 2025-03-22, 2025-03-21, 2025-03-20, 2025-03-19, 2025-03-18, 2025-05-24,2025-04-21, 2025-10-01, 2025-05-01, 2024-03-28, 2024-05-11, 2024-01-15, 2024-05-03, 2024-05-02, 2024-02-14,2025-02-01, 2025-01-31, 2025-02-02, 2024-05-17, 2024-05-18, 2024-05-22, 2024-05-21, 2024-04-02, 2024-04-09, 2024-05-06",
        "report_out": r"outPut/eval_backtest_v2_2.xlsx",
    }

    if CONFIG["cmd"] == "train":
        train_and_save(CONFIG["history"], CONFIG["model_dir"], th_spike=float(CONFIG["th_spike"]), th_neg=float(CONFIG["th_neg"]))
    elif CONFIG["cmd"] == "predict":
        predict_one_day(
            CONFIG["history"], CONFIG["model_dir"], CONFIG["forecast"], CONFIG["date"], CONFIG["out"],
            p_gate_spike=float(CONFIG["p_gate_spike"]), p_gate_neg=float(CONFIG["p_gate_neg"]),
            arm_day_neg=float(CONFIG.get("arm_day_neg", ARM_DAY_NEG_DEFAULT)),
            arm_day_neg_pv_th=float(CONFIG.get("arm_day_neg_pv_th", ARM_DAY_NEG_PV_TH_DEFAULT)),
            arm_day_neg_netload_min_th=float(CONFIG.get("arm_day_neg_netload_min_th", ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT)),
            arm_day_neg_use_or=bool(int(CONFIG.get("arm_day_neg_use_or", 1))),
            spike_budget_enable=bool(int(CONFIG.get("spike_budget_enable", int(SPIKE_BUDGET_ENABLE_DEFAULT)))),
            spike_budget_pts=int(CONFIG.get("spike_budget_pts", SPIKE_BUDGET_PTS_DEFAULT)),
            spike_budget_seeds=int(CONFIG.get("spike_budget_seeds", SPIKE_BUDGET_SEEDS_DEFAULT)),
            spike_budget_block_radius=int(CONFIG.get("spike_budget_block_radius", SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT)),
            spike_budget_min_gap=int(CONFIG.get("spike_budget_min_gap", SPIKE_BUDGET_MIN_GAP_DEFAULT)),
            spike_apply_to_p50=bool(int(CONFIG.get("spike_apply_to_p50", int(SPIKE_APPLY_TO_P50_DEFAULT)))),
        )
    elif CONFIG["cmd"] == "backtest":
        dates = [x.strip() for x in str(CONFIG["dates"]).split(",") if x.strip()]
        backtest_dates(
            CONFIG["history"], CONFIG["model_dir"], dates, CONFIG["report_out"],
            p_gate_spike=float(CONFIG["p_gate_spike"]),
            p_gate_neg=float(CONFIG["p_gate_neg"]),
            p_full_neg=float(CONFIG.get("p_full_neg", P_FULL_NEG_DEFAULT)),
            arm_day_neg=float(CONFIG.get("arm_day_neg", ARM_DAY_NEG_DEFAULT)),
            arm_day_neg_pv_th=float(CONFIG.get("arm_day_neg_pv_th", ARM_DAY_NEG_PV_TH_DEFAULT)),
            arm_day_neg_netload_min_th=float(CONFIG.get("arm_day_neg_netload_min_th", ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT)),
            arm_day_neg_use_or=bool(int(CONFIG.get("arm_day_neg_use_or", int(ARM_DAY_NEG_USE_OR_DEFAULT)))),
            spike_budget_enable=bool(int(CONFIG.get("spike_budget_enable", int(SPIKE_BUDGET_ENABLE_DEFAULT)))),
            spike_budget_pts=int(CONFIG.get("spike_budget_pts", SPIKE_BUDGET_PTS_DEFAULT)),
            spike_budget_seeds=int(CONFIG.get("spike_budget_seeds", SPIKE_BUDGET_SEEDS_DEFAULT)),
            spike_budget_block_radius=int(CONFIG.get("spike_budget_block_radius", SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT)),
            spike_budget_min_gap=int(CONFIG.get("spike_budget_min_gap", SPIKE_BUDGET_MIN_GAP_DEFAULT)),
            spike_apply_to_p50=bool(int(CONFIG.get("spike_apply_to_p50", int(SPIKE_APPLY_TO_P50_DEFAULT)))),
        )
    else:
        raise ValueError(f"Unknown cmd: {CONFIG['cmd']}")


def main() -> None:
    import sys
    if len(sys.argv) == 1:
        run_without_args()
        return

    args = parse_args()
    if args.cmd == "train":
        train_and_save(args.history, args.model_dir, th_spike=float(args.th_spike), th_neg=float(args.th_neg))
    elif args.cmd == "predict":
        predict_one_day(
            args.history, args.model_dir, args.forecast, args.date, args.out,
            p_gate_spike=float(args.p_gate_spike), p_gate_neg=float(args.p_gate_neg),
            p_full_neg=float(args.p_full_neg),
            arm_day_neg=float(args.arm_day_neg),
            arm_day_neg_pv_th=float(args.arm_day_neg_pv_th),
            arm_day_neg_netload_min_th=float(args.arm_day_neg_netload_min_th),
            arm_day_neg_use_or=bool(int(args.arm_day_neg_use_or)),
            spike_budget_enable=bool(int(args.spike_budget_enable)),
            spike_budget_pts=int(args.spike_budget_pts),
            spike_budget_seeds=int(args.spike_budget_seeds),
            spike_budget_block_radius=int(args.spike_budget_block_radius),
            spike_budget_min_gap=int(args.spike_budget_min_gap),
            spike_apply_to_p50=bool(int(args.spike_apply_to_p50)),
        )
    elif args.cmd == "backtest":
        dates = [x.strip() for x in str(args.dates).split(",") if x.strip()]
        backtest_dates(
            args.history, args.model_dir, dates, args.out,
            p_gate_spike=float(args.p_gate_spike), p_gate_neg=float(args.p_gate_neg),
            p_full_neg=float(args.p_full_neg),
            arm_day_neg=float(args.arm_day_neg),
            arm_day_neg_pv_th=float(args.arm_day_neg_pv_th),
            arm_day_neg_netload_min_th=float(args.arm_day_neg_netload_min_th),
            arm_day_neg_use_or=bool(int(args.arm_day_neg_use_or)),
            spike_budget_enable=bool(int(args.spike_budget_enable)),
            spike_budget_pts=int(args.spike_budget_pts),
            spike_budget_seeds=int(args.spike_budget_seeds),
            spike_budget_block_radius=int(args.spike_budget_block_radius),
            spike_budget_min_gap=int(args.spike_budget_min_gap),
            spike_apply_to_p50=bool(int(args.spike_apply_to_p50)),
        )
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()