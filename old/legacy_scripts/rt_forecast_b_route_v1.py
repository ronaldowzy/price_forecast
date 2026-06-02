
# rt_forecast_b_route_v1.py
# Two-stage (Route B) RT price forecasting (15-min, 96 points/day) for Shandong market.
#
# V1 assumptions / decisions (fixed by design):
# - Time slots are "interval end" labels: 00:15 ... 24:00 (NO 00:00).
# - Drop rows with missing required fields.
# - Wind/PV values (forecast/actual) are clipped to >= 0.
# - Extended NetLoad includes tie-line "联络线受电负荷": NetLoad = Load - Wind - PV - Tie.
# - B1 predicts ΔNetLoad quantiles (p10/p50/p90).
# - B2 predicts RT quantiles (p10/p50/p90) and P(RT>800). TH=800.
#
# Dependencies:
#   pip install pandas numpy scikit-learn joblib openpyxl matplotlib
#
# Usage:
#   1) Train:
#      python rt_forecast_b_route_v1.py train --history "价格预测数据集.csv" --model_dir "models_v1"
#
#   2) Predict for D+1:
#      python rt_forecast_b_route_v1.py predict --history "价格预测数据集.csv" --model_dir "models_v1" \
#          --forecast "Dplus1_forecast.xlsx" --date "2026-01-01" --out "RT_pred_2026-01-01.xlsx"
#
# Forecast input file (CSV/XLSX) must contain exactly 96 rows for the target date with columns:
#   日期, 时刻, 是否节假日, 是否周末休息日,
#   直调负荷(预测), 联络线受电负荷(预测), 风电总加(预测), 光伏总加(预测)
#
# Note:
# - If you later add "self forecast" columns, keep this script as-is and extend input handling separately.

from __future__ import annotations

import argparse
import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit


# =========================
# Config
# =========================

THRESHOLD_RT = 800.0  # P(RT > 800)

# Column names in your dataset
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

# Feature names (fixed by design)
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
    # 00:15 -> 1, 00:30 -> 2, ..., 23:45 -> 95
    return hh * 4 + (mm // 15)


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

    # enforce slot_id is int
    df["slot_id"] = pd.to_numeric(df["slot_id"], errors="coerce")

    return df


def add_netload(df: pd.DataFrame, is_history: bool) -> pd.DataFrame:
    df = df.copy()
    # Always need forecast side
    require_cols(df, [COL_LOAD_FC, COL_TIE_FC, COL_WIND_FC, COL_PV_FC], where="forecast fields")
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

    df["Ramp_fc_1"] = g["NetLoad_fc"].diff(1)
    df["Ramp_fc_4"] = g["NetLoad_fc"].diff(4)
    df["dPV_fc_1"] = g["PV_fc"].diff(1)

    return df


def add_error_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Error profile is computed per slot_id across days and MUST use ONLY past values:
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
    """
    Quick check (non-binding): corr(NetLoad_act, RT) should typically be positive.
    """
    out = {}
    tmp = df_hist.dropna(subset=["NetLoad_act", "NetLoad_fc", COL_RT]).copy()
    if len(tmp) < 100:
        return {"corr_act_rt": np.nan, "corr_fc_rt": np.nan}
    out["corr_act_rt"] = float(tmp["NetLoad_act"].corr(tmp[COL_RT]))
    out["corr_fc_rt"] = float(tmp["NetLoad_fc"].corr(tmp[COL_RT]))
    return out


def _fit_quantile_gbr(X: np.ndarray, y: np.ndarray, alpha: float, seed: int = 42) -> GradientBoostingRegressor:
    # Conservative hyper-parameters for stability. You can tune later.
    m = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=350,
        learning_rate=0.05,
        subsample=0.7,
        max_depth=3,
        min_samples_leaf=30,
        random_state=seed,
    )
    m.fit(X, y)
    return m


def train_b1_models(df: pd.DataFrame) -> Dict[str, object]:
    """
    Train B1 quantile models for dNetLoad.
    Returns dict with models and metadata.
    """
    X = df[FEATURE_B1].astype(float).values
    y = df["dNetLoad"].astype(float).values

    models = {
        "p10": _fit_quantile_gbr(X, y, 0.10),
        "p50": _fit_quantile_gbr(X, y, 0.50),
        "p90": _fit_quantile_gbr(X, y, 0.90),
    }
    return models


def b1_oof_predictions(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    Produce out-of-fold predictions for B1 (to train B2 without leakage).
    Split is done on unique dates with forward-chaining TimeSeriesSplit.
    """
    df = df.copy()
    days = np.array(sorted(df["date_dt"].unique()))
    if len(days) < 10:
        # Too few days to do meaningful OOF; fall back to in-sample preds
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


def train_b2_models(df_b2: pd.DataFrame) -> Dict[str, object]:
    """
    Train B2 quantile models for RT, plus a classifier for P(RT>THRESHOLD_RT).
    """
    X = df_b2[FEATURE_B2].astype(float).values
    y = df_b2[COL_RT].astype(float).values

    models_rt = {
        "p10": _fit_quantile_gbr(X, y, 0.10),
        "p50": _fit_quantile_gbr(X, y, 0.50),
        "p90": _fit_quantile_gbr(X, y, 0.90),
    }

    y_cls = (y > THRESHOLD_RT).astype(int)
    pos = int(y_cls.sum())

    if pos < 5:
        clf = DummyClassifier(strategy="prior")
        clf.fit(X, y_cls)
    else:
        clf = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )
        clf.fit(X, y_cls)

    return {"rt_models": models_rt, "clf": clf, "pos_count": pos}


def build_history_feature_frame(history_path: str) -> pd.DataFrame:
    """
    Build a full feature frame from the history file.
    """
    df = read_table(history_path)
    df = add_slot_calendar(df)
    df = add_netload(df, is_history=True)

    # needed numeric fields
    to_numeric(df, [COL_RT])

    df = add_within_day_features(df)
    df = add_error_profile_features(df)
    df = add_rt_lag_features(df)

    return df


def select_train_rows(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    For training:
      - B1 needs FEATURE_B1 + dNetLoad
      - B2 (base) needs all of FEATURE_B2 except the NetLoad_act_pXX (these are generated by B1 OOF preds),
        plus the RT label.
    Therefore, we only enforce the existence / non-null of:
      FEATURE_B1 + dNetLoad + RT + BASE_B2
    """
    df_feat = df_feat.copy()

    base_b2 = [c for c in FEATURE_B2 if not c.startswith("NetLoad_act_")]
    needed = set(FEATURE_B1 + base_b2 + ["dNetLoad", COL_RT])

    missing = [c for c in needed if c not in df_feat.columns]
    if missing:
        raise KeyError(f"Feature frame missing required columns: {missing}")

    df = df_feat.dropna(subset=list(needed)).copy()

    # enforce numeric
    for c in (FEATURE_B1 + base_b2 + ["dNetLoad", COL_RT]):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(needed)).copy()

    return df


def train_and_save(history_path: str, model_dir: str) -> None:
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
    b2_pack = train_b2_models(df_b2)

    bundle = {
        "threshold_rt": THRESHOLD_RT,
        "feature_b1": FEATURE_B1,
        "feature_b2": FEATURE_B2,
        "b1_models": b1_models,
        "b2_rt_models": b2_pack["rt_models"],
        "b2_clf": b2_pack["clf"],
        "b2_pos_count": b2_pack["pos_count"],
        "corr_info": corr_info,
        "history_path": os.path.abspath(history_path),
    }

    model_path = os.path.join(model_dir, "rt_routeB_v1.joblib")
    joblib.dump(bundle, model_path)

    meta_path = os.path.join(model_dir, "rt_routeB_v1_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": os.path.abspath(model_path),
                "threshold_rt": THRESHOLD_RT,
                "feature_b1": FEATURE_B1,
                "feature_b2": FEATURE_B2,
                "b2_pos_count": b2_pack["pos_count"],
                "corr_info": corr_info,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] Saved model bundle: {os.path.abspath(model_path)}")
    print(f"[OK] Saved metadata:     {os.path.abspath(meta_path)}")
    print(f"[INFO] History rows used for training: {len(df_train):,}")
    print(f"[INFO] Unique days used for training:  {df_train['date_dt'].nunique():,}")
    print(f"[INFO] B2 positives (RT>{THRESHOLD_RT}): {b2_pack['pos_count']}")
    print(f"[INFO] Sanity corr: {corr_info}")


def compute_error_profile_for_date(df_hist: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute Err_1 / Err_mean_7 / Err_absmean_7 / Err_p10_14 / Err_p90_14 for each slot_id
    using history strictly before target_date.
    """
    hist = df_hist[df_hist["date_dt"] < target_date].copy()
    hist = hist.dropna(subset=["slot_id", "dNetLoad"]).copy()
    hist = hist.sort_values(["date_dt", "slot_id"])

    out = []
    for sid, g in hist.groupby("slot_id"):
        s = g["dNetLoad"].astype(float).values
        # last values
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
    """
    For each slot_id:
      RT_lag_1d = RT(date-1, slot)
      RT_lag_7d = RT(date-7, slot)
    plus RT_vol_1d = std(RT) of date-1.
    Fallbacks:
      - If a lag is missing, fill with same-slot mean of last 7 days before target_date.
      - If RT_vol_1d missing, fill with mean daily std of last 7 days before target_date.
    """
    hist = df_hist[df_hist["date_dt"] < target_date].copy()
    hist = hist.dropna(subset=["slot_id", COL_RT]).copy()
    hist = hist.sort_values(["date_dt", "slot_id"])
    hist[COL_RT] = pd.to_numeric(hist[COL_RT], errors="coerce")

    # Build per-slot rolling mean (last 7 days)
    rt_mean7 = []
    for sid, g in hist.groupby("slot_id"):
        s = g[COL_RT].astype(float).values
        last7 = s[-7:] if len(s) >= 7 else s
        rt_mean7.append({"slot_id": sid, "RT_mean_7": float(np.nanmean(last7)) if len(last7) >= 3 else np.nan})
    rt_mean7 = pd.DataFrame(rt_mean7)

    # lag 1d / 7d
    d1 = target_date - pd.Timedelta(days=1)
    d7 = target_date - pd.Timedelta(days=7)

    rt_d1 = hist[hist["date_dt"] == d1][["slot_id", COL_RT]].rename(columns={COL_RT: "RT_lag_1d"})
    rt_d7 = hist[hist["date_dt"] == d7][["slot_id", COL_RT]].rename(columns={COL_RT: "RT_lag_7d"})

    state = rt_mean7.merge(rt_d1, on="slot_id", how="left").merge(rt_d7, on="slot_id", how="left")

    # fill missing lags with RT_mean_7
    state["RT_lag_1d"] = state["RT_lag_1d"].fillna(state["RT_mean_7"])
    state["RT_lag_7d"] = state["RT_lag_7d"].fillna(state["RT_mean_7"])

    # daily vol lag 1d
    daily_std = hist.groupby("date_dt")[COL_RT].std().sort_index()
    vol_1d = float(daily_std.get(d1, np.nan))
    if np.isnan(vol_1d):
        last7days = daily_std[daily_std.index < target_date].tail(7).values
        vol_1d = float(np.nanmean(last7days)) if len(last7days) >= 3 else np.nan

    return state[["slot_id", "RT_lag_1d", "RT_lag_7d"]], vol_1d


def predict_one_day(history_path: str, model_dir: str, forecast_path: str, date_str: str, out_path: str) -> None:
    bundle_path = os.path.join(model_dir, "rt_routeB_v1.joblib")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}. Run train first.")

    bundle = joblib.load(bundle_path)
    b1_models = bundle["b1_models"]
    b2_models = bundle["b2_rt_models"]
    clf = bundle["b2_clf"]

    target_date = pd.to_datetime(date_str)

    # Load history and build required fields for computing profiles/lags
    df_hist = build_history_feature_frame(history_path)
    df_hist = df_hist[df_hist["date_dt"] < target_date].copy()

    # Load forecast input (must contain only the target date, 96 rows)
    df_fc = read_table(forecast_path)
    df_fc = add_slot_calendar(df_fc)
    df_fc = df_fc[df_fc["date_dt"] == target_date].copy()

    if df_fc.empty:
        raise ValueError(f"No rows for target date {date_str} found in forecast file.")
    if df_fc["slot_id"].nunique() != 96:
        raise ValueError(f"Forecast file must contain 96 unique slot_id rows for {date_str}, got {df_fc['slot_id'].nunique()}.")

    df_fc = add_netload(df_fc, is_history=False)
    df_fc = add_within_day_features(df_fc)

    # error profile for target date (from history)
    err = compute_error_profile_for_date(df_hist, target_date)
    # RT state for target date (from history)
    rt_state, rt_vol_1d = compute_rt_state_for_date(df_hist, target_date)

    df_fc = df_fc.merge(err, on="slot_id", how="left").merge(rt_state, on="slot_id", how="left")
    df_fc["RT_vol_1d"] = rt_vol_1d

    # drop rows with missing required features (if history too short)
    df_fc = df_fc.dropna(subset=FEATURE_B1).copy()

    X1 = df_fc[FEATURE_B1].astype(float).values
    d10 = b1_models["p10"].predict(X1)
    d50 = b1_models["p50"].predict(X1)
    d90 = b1_models["p90"].predict(X1)

    df_fc["dNet_p10"] = d10
    df_fc["dNet_p50"] = d50
    df_fc["dNet_p90"] = d90

    df_fc["NetLoad_act_p10"] = df_fc["NetLoad_fc"] + df_fc["dNet_p10"]
    df_fc["NetLoad_act_p50"] = df_fc["NetLoad_fc"] + df_fc["dNet_p50"]
    df_fc["NetLoad_act_p90"] = df_fc["NetLoad_fc"] + df_fc["dNet_p90"]

    # B2 features
    df_fc = df_fc.dropna(subset=FEATURE_B2).copy()
    X2 = df_fc[FEATURE_B2].astype(float).values

    rt_p10 = b2_models["p10"].predict(X2)
    rt_p50 = b2_models["p50"].predict(X2)
    rt_p90 = b2_models["p90"].predict(X2)

    # probability
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X2)[:, 1]
    else:
        # DummyClassifier always has predict_proba; keep safe
        prob = np.full(len(X2), np.nan)

    df_out = pd.DataFrame({
        "日期": df_fc[COL_DATE].values,
        "时刻": df_fc["time_str"].values,
        "slot_id": df_fc["slot_id"].astype(int).values,
        "RT_p10": rt_p10,
        "RT_p50": rt_p50,
        "RT_p90": rt_p90,
        f"P(RT>{int(THRESHOLD_RT)})": prob,
        "NetLoad_fc": df_fc["NetLoad_fc"].values,
        "NetLoad_act_p50": df_fc["NetLoad_act_p50"].values,
        "Err_mean_7": df_fc["Err_mean_7"].values,
        "RT_lag_1d": df_fc["RT_lag_1d"].values,
    }).sort_values("slot_id")

    # Write Excel
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="pred", index=False)

    print(f"[OK] Saved prediction: {os.path.abspath(out_path)}")


def main():
    # parser = argparse.ArgumentParser()
    # sub = parser.add_subparsers(dest="cmd", required=True)
    #
    # p_train = sub.add_parser("train", help="Train Route-B models and save to model_dir")
    # p_train.add_argument("--history", required=True, help="History CSV/XLSX with full columns")
    # p_train.add_argument("--model_dir", required=True, help="Directory to save models")
    #
    # p_pred = sub.add_parser("predict", help="Predict RT for one target date (D+1)")
    # p_pred.add_argument("--history", required=True, help="History CSV/XLSX with full columns")
    # p_pred.add_argument("--model_dir", required=True, help="Directory containing saved models")
    # p_pred.add_argument("--forecast", required=True, help="Forecast CSV/XLSX for D+1 (96 rows for target date)")
    # p_pred.add_argument("--date", required=True, help="Target date, e.g. 2026-01-01")
    # p_pred.add_argument("--out", required=True, help="Output Excel path")

    # args = parser.parse_args()
    cmd = "predict"
    history = "data/价格预测数据集.csv"
    model_dir = "model"
    forecast = "forecast_input.xlsx"
    date = "2025-12-31"
    out = "outPut/forcast.xlsx"



    if cmd == "train":
        train_and_save(history, model_dir)
    elif cmd == "predict":
        predict_one_day(history, model_dir, forecast, date, out)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
