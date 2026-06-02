# rt_forecast_b_route_v2_6_4.py
# Route-B RT price forecasting (15-min, 96 points/day) for Shandong market.
#
# V2.6 upgrades (vs v2.5.x):
# 0) Add pumped-storage (抽蓄) prediction & effective net-load features using inferred sign convention.
#
# Historical V2.2 upgrades (kept):
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
#     python rt_forecast_b_route_v2_6.py train --history "价格预测数据集.csv" --model_dir "models_v2_6" --th_spike 800 --th_neg 0
#   Predict:
#     python rt_forecast_b_route_v2_6.py predict --history "价格预测数据集.csv" --model_dir "models_v2_6" --forecast "Dplus1_forecast.xlsx" --date "2026-01-01" --out "RT_pred_2026-01-01.xlsx"
#   Backtest:
#     python rt_forecast_b_route_v2_6.py backtest --history "价格预测数据集.csv" --model_dir "models_v2_6" --dates "2025-11-08,2025-01-26" --out "eval_backtest_v2_6.xlsx"
#
# If you prefer editing parameters inside the script, use run_without_args().

from __future__ import annotations

import argparse
import os
import re
import json
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# Excel output helper (bilingual header)
try:
    import openpyxl  # type: ignore
    from openpyxl.chart import LineChart, Reference  # type: ignore
except Exception:  # pragma: no cover
    openpyxl = None  # type: ignore
    LineChart = None  # type: ignore
    Reference = None  # type: ignore

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
ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT = 7000.0
ARM_DAY_NEG_L10_DEFAULT = None  # optional: day-level arming threshold for negative risk band (lower10) only; None=>use ARM_DAY_NEG_DEFAULT
P_GATE_NEG_L10_DEFAULT = None      # optional: p_neg gate threshold for lower10; None=>use p_gate_neg
P_FULL_NEG_L10_DEFAULT = None      # optional: full negative gate threshold for lower10; None=>use p_full_neg
  # NetLoad_fc daily min threshold to allow negative-day arming
ARM_DAY_NEG_USE_OR_DEFAULT = True    # True: (PV OR NetLoad) ; False: (PV AND NetLoad)


# Spike budget trigger (v2.5): select a small number of high-risk slots per day
SPIKE_BUDGET_ENABLE_DEFAULT = True
SPIKE_BUDGET_PTS_DEFAULT = 6          # max number of 15-min slots to flag per day
SPIKE_BUDGET_SEEDS_DEFAULT = 2        # number of seed slots before expansion
SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT = 1 # expand each seed by +/- radius (in 15-min slots)
SPIKE_BUDGET_MIN_GAP_DEFAULT = 4      # minimum separation between seeds (in 15-min slots)
SPIKE_APPLY_TO_P50_DEFAULT = False    # if True, spike budget affects RT_gate_p50; if False, only affects risk bounds & scenario
SPIKE_SCORE_MODE_DEFAULT = "p_spike"  # 'p_spike' | 'u90' | 'hybrid' (spike budget scoring)


# =========================
# Excel output (bilingual headers)
# =========================

# 说明：第一行保留原始（英文/原名）列名；第二行为中文解释列名。
# 若未命中映射，则默认使用原列名作为中文标题（不会中断程序）。

CN_COLMAP_PRED: Dict[str, str] = {
    "日期": "日期",
    "时刻": "时段(15分钟, 00:15…24:00)",
    "slot_id": "时段序号(1=00:15,…,96=24:00)",

    "p_norm": "常规(非尖峰/非负价)概率",
    "p_spike": "尖峰概率(校准后)",
    "p_neg": "负价概率(校准后)",

    "spike_budget_used": "是否启用尖峰预算触发(1/0)",
    "spike_selected": "尖峰预算选中(1/0)",
    "spike_budget_pts": "尖峰预算：每日最多选中点数",
    "spike_budget_seeds": "尖峰预算：种子点数量",
    "spike_budget_block_radius": "尖峰预算：每个种子扩展半径(点)",
    "spike_budget_min_gap": "尖峰预算：种子最小间隔(点)",
    "spike_apply_to_p50": "尖峰预算是否作用于P50(1=是,0=否)",

    "neg_armed": "负价门控：当日是否武装(1/0)",
    "neg_prob_ok": "负价门控：概率条件是否满足(1/0)",
    "neg_guard_ok": "负价门控：护栏条件是否满足(1/0)",
    "neg_guard_pv_ok": "负价护栏：PV条件是否满足(1/0)",
    "neg_guard_netload_ok": "负价护栏：净负荷条件是否满足(1/0)",
    "PV_fc_max_day": "当日PV预测最大值",
    "NetLoad_fc_min_day": "当日净负荷预测最小值",
    "arm_day_neg_pv_th": "负价护栏：PV阈值",
    "arm_day_neg_netload_min_th": "负价护栏：净负荷阈值",
    "arm_day_neg_use_or": "负价护栏：OR(1) / AND(0)",

    "RT_norm_p10": "常规情景RT预测P10",
    "RT_norm_p50": "常规情景RT预测P50",
    "RT_norm_p90": "常规情景RT预测P90",

    "RT_spike_p50": "尖峰情景RT预测P50(尖峰阈值以上)",
    "RT_spike_p90": "尖峰情景RT预测P90(尖峰阈值以上)",

    "RT_neg_p50": "负价情景RT预测P50(负价幅度)",
    "RT_neg_p90": "负价情景RT预测P90(更负/更极端)",

    "RT_mix_p50": "混合输出：P50(三情景加权)",
    "RT_mix_upper90": "混合输出：上行风险P90",
    "RT_mix_lower10": "混合输出：下行风险P10",

    "RT_gate_p50": "门控输出：P50(优先负价→尖峰→常规)",
    "RT_gate_upper90": "门控输出：上行风险P90",
    "RT_gate_lower10": "门控输出：下行风险P10",

    "RT_spike_budget_p50": "尖峰预算输出：P50(仅对预算选中点替换为尖峰情景)",
    "RT_spike_budget_upper90": "尖峰预算输出：上行风险P90(预算选中点替换)",

    "NetLoad_fc": "净负荷(预测)=直调负荷-风-光-受电",
    "NetLoad_act_p50": "净负荷(推断实际)P50",
    "NetLoad_eff_fc": "有效净负荷(预测)=净负荷+抽蓄折算负荷",
    "Ramp_eff_fc_1": "有效净负荷爬坡1步(预测)",
    "Ramp_eff_fc_4": "有效净负荷爬坡4步(预测)",
    "NetLoad_eff_act_p10": "有效净负荷(推断)P10",
    "NetLoad_eff_act_p50": "有效净负荷(推断)P50",
    "NetLoad_eff_act_p90": "有效净负荷(推断)P90",
    "PS_pred_p10": "抽蓄预测P10(MW)",
    "PS_pred_p50": "抽蓄预测P50(MW)",
    "PS_pred_p90": "抽蓄预测P90(MW)",
    "PS_pred_as_load": "抽蓄预测折算为负荷(MW)",
    "PV_fc": "光伏(预测)",
    "Wind_fc": "风电(预测)",
    "Tie_fc": "联络线受电负荷(预测)",
    "RT_lag_1d": "RT滞后：上一完整日同slot",
    "RT_lag_7d": "RT滞后：7日前同slot",
    "RT_vol_1d": "上一完整日RT日内波动(标准差)",
}


CN_COLMAP_PER_DAY: Dict[str, str] = {
    "date": "日期",
    "RT_true_min": "当日RT最小值",
    "RT_true_max": "当日RT最大值",
    "neg_points_true": "当日负价点数(RT<0)",
    "spike_points_true": "当日尖峰点数(RT>尖峰阈值)",
    "MAE_mix_p50": "MAE(混合P50)",
    "RMSE_mix_p50": "RMSE(混合P50)",
    "MAE_gate_p50": "MAE(门控P50)",
    "RMSE_gate_p50": "RMSE(门控P50)",
    "MAE_spikeBudget_p50": "MAE(尖峰预算P50)",
    "RMSE_spikeBudget_p50": "RMSE(尖峰预算P50)",
    "spike_selected_n": "尖峰预算选中点数",
    "Coverage_norm_p10_p90": "覆盖率(常规P10~P90)",
    "AUC_spike": "AUC(尖峰分类)",
    "AP_spike": "AP(尖峰分类)",
    "Brier_spike": "Brier(尖峰分类)",
    "AUC_neg": "AUC(负价分类)",
    "AP_neg": "AP(负价分类)",
    "Brier_neg": "Brier(负价分类)",
    "p_spike_max": "当日p_spike最大值",
    "p_neg_max": "当日p_neg最大值",
}

CN_COLMAP_RUN_INFO: Dict[str, str] = {
    "key": "参数项",
    "value": "参数值",
}


def _cn_header_for_col(col: str) -> str:
    """Return Chinese header for a column, with light pattern support for dynamic columns."""
    if col in CN_COLMAP_PRED:
        return CN_COLMAP_PRED[col]
    if col in CN_COLMAP_PER_DAY:
        return CN_COLMAP_PER_DAY[col]
    if col in CN_COLMAP_RUN_INFO:
        return CN_COLMAP_RUN_INFO[col]
    # dynamic probability columns like: P(RT>800) / P(RT<0)
    if isinstance(col, str) and col.startswith("P(RT>"):
        return "尖峰概率(阈值口径)"  # 阈值已在英文列名中体现
    if isinstance(col, str) and col.startswith("P(RT<"):
        return "负价概率(阈值口径)"
    if col == "RT_true":
        return "实时价格(真实)"
    return str(col)


def write_df_with_bilingual_header(
    writer: pd.ExcelWriter,
    sheet_name: str,
    df: pd.DataFrame,
    freeze_panes: str = "A3",
) -> None:
    """Write DataFrame with two header rows: (1) original column names, (2) Chinese titles."""
    # Write data (no header) starting from row 3
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=2)
    ws = writer.sheets[sheet_name]
    for j, col in enumerate(df.columns, start=1):
        ws.cell(row=1, column=j, value=col)
        ws.cell(row=2, column=j, value=_cn_header_for_col(str(col)))
    try:
        ws.freeze_panes = freeze_panes
    except Exception:
        pass
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

# Additional load components that should be included in net load calculation
COL_NON_MARKET_NUCLEAR_FC = "非市场化核电总加(预测)"
COL_SELF_SUPPLY_UNITS_FC = "自备机组总加(预测)"
COL_LOCAL_POWER_PLANT_FC = "地方电厂发电总加(预测)"

COL_LOAD_ACT = "直调负荷(实际)"
COL_TIE_ACT = "联络线受电负荷(实际)"
COL_WIND_ACT = "风电总加(实际)"
COL_PV_ACT = "光伏总加(实际)"

# Additional actual load components
COL_NON_MARKET_NUCLEAR_ACT = "非市场化核电总加(实际)"
COL_SELF_SUPPLY_UNITS_ACT = "自备机组总加(实际)"
COL_LOCAL_POWER_PLANT_ACT = "地方电厂发电总加(实际)"

COL_PS_ACT = "抽蓄(实际)"  # 抽水蓄能(实际), MW, 正负含义以数据为准

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

FEATURE_PS = [
    "slot_id", "is_holiday", "is_weekend", "month", "season",
    "NetLoad_fc", "Ramp_fc_1", "Ramp_fc_4",
    "PV_fc", "dPV_fc_1", "Wind_fc", "Tie_fc",
    "Err_1", "Err_mean_7", "Err_absmean_7", "Err_p10_14", "Err_p90_14",
    "RT_lag_1d", "RT_lag_7d", "RT_vol_1d",
]

FEATURE_B2 = [
    "slot_id", "is_holiday", "is_weekend", "month", "season",
    "NetLoad_fc", "NetLoad_eff_fc",
    "Ramp_fc_1", "Ramp_fc_4", "Ramp_eff_fc_1", "Ramp_eff_fc_4",
    "PV_fc", "dPV_fc_1", "Wind_fc", "Tie_fc",
    "PS_pred_as_load",
    "RT_lag_1d", "RT_lag_7d", "RT_vol_1d",
    "NetLoad_eff_act_p10", "NetLoad_eff_act_p50", "NetLoad_eff_act_p90",
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
    
    # Add additional load components with fallback to 0 if not present
    # Add additional generation / supply components (column missing or NaN -> 0.0)
    for _src, _dst in [
        (COL_NON_MARKET_NUCLEAR_FC, "NonMarketNuclear_fc"),
        (COL_SELF_SUPPLY_UNITS_FC, "SelfSupplyUnits_fc"),
        (COL_LOCAL_POWER_PLANT_FC, "LocalPowerPlant_fc"),
    ]:
        if _src in df.columns:
            df[_dst] = pd.to_numeric(df[_src], errors="coerce").fillna(0.0)
        else:
            df[_dst] = 0.0


    # Updated net load calculation including additional components
    df["NetLoad_fc"] = (df["Load_fc"] 
                        - df["Wind_fc"] 
                        - df["PV_fc"] 
                        - df["Tie_fc"]
                        - df["NonMarketNuclear_fc"]
                        - df["SelfSupplyUnits_fc"]
                        - df["LocalPowerPlant_fc"])

    if is_history:
        require_cols(df, [COL_LOAD_ACT, COL_TIE_ACT, COL_WIND_ACT, COL_PV_ACT], where="actual fields")
        clip_nonneg(df, [COL_WIND_ACT, COL_PV_ACT])
        to_numeric(df, [COL_LOAD_ACT, COL_TIE_ACT])

        df["Load_act"] = df[COL_LOAD_ACT]
        df["Tie_act"] = df[COL_TIE_ACT]
        df["Wind_act"] = pd.to_numeric(df[COL_WIND_ACT], errors="coerce")
        df["PV_act"] = pd.to_numeric(df[COL_PV_ACT], errors="coerce")
        
        # Add additional actual load components with fallback to 0 if not present
        # Add additional generation / supply components (column missing or NaN -> 0.0)
        for _src, _dst in [
            (COL_NON_MARKET_NUCLEAR_ACT, "NonMarketNuclear_act"),
            (COL_SELF_SUPPLY_UNITS_ACT, "SelfSupplyUnits_act"),
            (COL_LOCAL_POWER_PLANT_ACT, "LocalPowerPlant_act"),
        ]:
            if _src in df.columns:
                df[_dst] = pd.to_numeric(df[_src], errors="coerce").fillna(0.0)
            else:
                df[_dst] = 0.0


        # Updated actual net load calculation including additional components
        df["NetLoad_act"] = (df["Load_act"]
                             - df["Wind_act"] 
                             - df["PV_act"] 
                             - df["Tie_act"]
                             - df["NonMarketNuclear_act"]
                             - df["SelfSupplyUnits_act"]
                             - df["LocalPowerPlant_act"])
        df["dNetLoad"] = df["NetLoad_act"] - df["NetLoad_fc"]

    return df


def add_pumped_storage_actual(df: pd.DataFrame) -> pd.DataFrame:
    """Add pumped-storage actual column.

    Column convention in the dataset:
      - COL_PS_ACT = 抽蓄(实际), typical range about [-2500, 2500] MW.
    IMPORTANT: The sign convention may vary across sources. We infer a robust
    'as_load' direction during training (ps_sign_as_load) and persist it in the model bundle.
    """
    df = df.copy()
    require_cols(df, [COL_PS_ACT], where="pumped-storage actual")
    df["PS_act"] = pd.to_numeric(df[COL_PS_ACT], errors="coerce")
    # Keep outliers bounded (avoid training blow-ups). Adjust if needed.
    df["PS_act"] = df["PS_act"].clip(-5000.0, 5000.0)
    return df


def add_eff_netload_from_ps(df: pd.DataFrame, ps_as_load_col: str, out_prefix: str = "") -> pd.DataFrame:
    """Compute effective net-load features after incorporating pumped-storage as 'additional load'.

    NetLoad_eff = NetLoad + PS_as_load
      - If PS_as_load>0 means pumping (extra demand), NetLoad_eff increases.
      - If PS_as_load<0 means generating (extra supply), NetLoad_eff decreases.

    This function does NOT decide the sign; caller should provide PS_as_load already.
    """
    df = df.copy()
    if "NetLoad_fc" not in df.columns:
        raise KeyError("NetLoad_fc missing before computing NetLoad_eff.")
    df[f"{out_prefix}NetLoad_eff_fc"] = pd.to_numeric(df["NetLoad_fc"], errors="coerce") + pd.to_numeric(df[ps_as_load_col], errors="coerce")
    # ramps on effective net-load (within the day)
    df = df.sort_values(["date_dt", "slot_id"]).reset_index(drop=True)
    g = df.groupby("date_dt", group_keys=False)
    df[f"{out_prefix}Ramp_eff_fc_1"] = g[f"{out_prefix}NetLoad_eff_fc"].diff(1).fillna(0.0)
    df[f"{out_prefix}Ramp_eff_fc_4"] = g[f"{out_prefix}NetLoad_eff_fc"].diff(4).fillna(0.0)
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


def infer_ps_sign_as_load(df_train: pd.DataFrame) -> int:
    """Define sign convention for pumped-storage in this project.

    Dataset semantics (verified on the provided history set):
      - 抽蓄(实际) > 0 : pumped-storage DISCHARGE / generation injected to the grid (supply increases)
      - 抽蓄(实际) < 0 : pumped-storage PUMPING / charging (load increases)

    For price formation, it is more useful to express pumped-storage as an *equivalent load* (PS_as_load),
    where:
      - PS_as_load > 0 behaves like additional demand (tends to push RT up)
      - PS_as_load < 0 behaves like additional supply (tends to push RT down)

    Therefore we convert:
        PS_as_load = - 抽蓄(实际)

    This means ps_sign_as_load is FIXED to -1 (do NOT infer from corr(PS, RT), which is endogenous).
    """
    return -1



def ps_oof_predictions(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """OOF median prediction for pumped-storage (PS_act) by time-split days."""
    df = df.copy()
    require_cols(df, ["PS_act"], where="PS OOF target")
    days = np.array(sorted(df["date_dt"].unique()))
    if len(days) < 10:
        X = df[FEATURE_PS].astype(float).values
        y = df["PS_act"].astype(float).values
        m50 = _fit_quantile_gbr(X, y, 0.50)
        df["PS_p50_oof"] = m50.predict(X)
        return df

    pred50 = np.full(len(df), np.nan, dtype=float)
    tss = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(days) // 3)))

    for train_idx, test_idx in tss.split(days):
        train_days = days[train_idx]
        test_days = days[test_idx]

        train_mask = df["date_dt"].isin(train_days).values
        test_mask = df["date_dt"].isin(test_days).values

        Xtr = df.loc[train_mask, FEATURE_PS].astype(float).values
        ytr = df.loc[train_mask, "PS_act"].astype(float).values
        Xte = df.loc[test_mask, FEATURE_PS].astype(float).values

        m50 = _fit_quantile_gbr(Xtr, ytr, 0.50)
        pred50[test_mask] = m50.predict(Xte)

    df["PS_p50_oof"] = pred50
    return df


def train_ps_models(df_train: pd.DataFrame) -> Dict[str, object]:
    """Train pumped-storage quantile models on full training set."""
    X = df_train[FEATURE_PS].astype(float).values
    y = df_train["PS_act"].astype(float).values
    return {
        "p10": _fit_quantile_gbr(X, y, 0.10),
        "p50": _fit_quantile_gbr(X, y, 0.50),
        "p90": _fit_quantile_gbr(X, y, 0.90),
    }



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
    df = add_pumped_storage_actual(df)
    to_numeric(df, [COL_RT, "PS_act"])
    df = add_within_day_features(df)
    df = add_error_profile_features(df)
    df = add_rt_lag_features(df)
    return df


def select_train_rows(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Select rows usable for training (drop missing).

    v2.6 adds pumped-storage modeling, so training needs:
      - B1 features + dNetLoad + RT
      - PS features + PS_act
      - RT lag/state features that are available for D+1 prediction
    """
    df_feat = df_feat.copy()

    needed = set(FEATURE_B1 + FEATURE_PS + ["dNetLoad", COL_RT, "PS_act"])

    missing = [c for c in needed if c not in df_feat.columns]
    if missing:
        raise KeyError(f"Feature frame missing required columns: {missing}")

    df = df_feat.dropna(subset=list(needed)).copy()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(needed)).copy()
    return df


# =========================
# Train / Save
# =========================

def train_and_save(history_path: str, model_dir: str, th_spike: float, th_neg: float, ps_alpha: float = 0.6) -> None:
    """Train v2.6 models and save bundle.

    v2.6 additions:
      - Pumped-storage (PS) median model using only D+1-available signals
      - Effective net-load features: NetLoad_eff = NetLoad + PS_as_load
      - Use the most recent complete days to derive recent-state features (handled at prediction time)
    """
    os.makedirs(model_dir, exist_ok=True)

    df_feat = build_history_feature_frame(history_path)
    corr_info = sanity_check_tie_sign(df_feat)

    df_train = select_train_rows(df_feat)

    # Infer sign convention and persist it
    ps_sign_as_load = infer_ps_sign_as_load(df_train)

    # --- OOF predictions (time-split) to avoid leakage into B2 ---
    df_oof = b1_oof_predictions(df_train, n_splits=5)
    df_oof = ps_oof_predictions(df_oof, n_splits=5)

    df_b2 = df_oof.dropna(subset=["dNet_p10_oof", "dNet_p50_oof", "dNet_p90_oof", "PS_p50_oof"]).copy()
    df_b2["NetLoad_act_p10"] = df_b2["NetLoad_fc"] + df_b2["dNet_p10_oof"]
    df_b2["NetLoad_act_p50"] = df_b2["NetLoad_fc"] + df_b2["dNet_p50_oof"]
    df_b2["NetLoad_act_p90"] = df_b2["NetLoad_fc"] + df_b2["dNet_p90_oof"]

    # Pumped-storage predicted (OOF) as-load, used to build effective net-load features
    df_b2["PS_pred_as_load"] = float(ps_alpha) * float(ps_sign_as_load) * df_b2["PS_p50_oof"]
    df_b2 = add_eff_netload_from_ps(df_b2, ps_as_load_col="PS_pred_as_load", out_prefix="")

    # Effective actual net-load quantiles (approx: NetLoad_act_pXX + PS_pred_as_load)
    df_b2["NetLoad_eff_act_p10"] = df_b2["NetLoad_act_p10"] + df_b2["PS_pred_as_load"]
    df_b2["NetLoad_eff_act_p50"] = df_b2["NetLoad_act_p50"] + df_b2["PS_pred_as_load"]
    df_b2["NetLoad_eff_act_p90"] = df_b2["NetLoad_act_p90"] + df_b2["PS_pred_as_load"]

    # --- Train final models on full data ---
    b1_models = train_b1_models(df_train)
    ps_models = train_ps_models(df_train)
    b2_pack = train_b2_regime_models(df_b2, th_spike=th_spike, th_neg=th_neg)

    bundle = {
        "version": "v2.6.4",
        "th_spike": float(th_spike),
        "th_neg": float(th_neg),

        "feature_b1": FEATURE_B1,
        "feature_ps": FEATURE_PS,
        "feature_b2": FEATURE_B2,

        "b1_models": b1_models,
        "ps_models": ps_models,
        "ps_sign_as_load": int(ps_sign_as_load),
        "ps_alpha": float(ps_alpha),
        "ps_clip": [-5000.0, 5000.0],

        "b2_clf_spike": b2_pack["clf_spike"],
        "b2_clf_neg": b2_pack["clf_neg"],
        "b2_cal_spike": b2_pack["cal_spike"],
        "b2_cal_neg": b2_pack["cal_neg"],

        "b2_norm_models": b2_pack["norm_models"],
        "b2_spike_models": b2_pack["spike_models"],
        "b2_neg_models": b2_pack["neg_models"],

        "pos_spike": int(b2_pack["pos_spike"]),
        "pos_neg": int(b2_pack["pos_neg"]),
        "corr_info": corr_info,
        "history_path": os.path.abspath(history_path),
    }

    model_path = os.path.join(model_dir, "rt_routeB_v2_6.joblib")
    joblib.dump(bundle, model_path)

    meta = {
        "model_path": os.path.abspath(model_path),
        "version": "v2.6.4",
        "th_spike": float(th_spike),
        "th_neg": float(th_neg),
        "feature_b1": FEATURE_B1,
        "feature_ps": FEATURE_PS,
        "feature_b2": FEATURE_B2,
        "ps_sign_as_load": int(ps_sign_as_load),
        "ps_alpha": float(ps_alpha),
        "corr_info": corr_info,
        "history_path": os.path.abspath(history_path),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = os.path.join(model_dir, "rt_routeB_v2_6_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved model bundle: {model_path}")
    print(f"[OK] Saved meta json: {meta_path}")


def compute_error_profile_for_date(df_hist: pd.DataFrame, target_date: dt.date) -> pd.DataFrame:
    """Compute recent net-load forecast error profile by slot_id.

    We compute statistics from the most recent *complete* days to avoid leaking partial-day values.
    A "complete" day is defined as having >=96 unique slot_id with non-null dNetLoad.
    """
    hist = df_hist.copy()
    require_cols(hist, ["date_dt", "slot_id", "dNetLoad"], where="error profile")
    hist = hist[hist["date_dt"] < target_date].copy()
    hist["dNetLoad"] = pd.to_numeric(hist["dNetLoad"], errors="coerce")
    hist = hist.dropna(subset=["dNetLoad", "slot_id", "date_dt"]).copy()

    if hist.empty:
        raise ValueError(f"Not enough history before {target_date} to compute error profile.")

    day_cnt = hist.groupby("date_dt")["slot_id"].nunique().sort_index()
    complete_days = day_cnt[day_cnt >= 96].index.tolist()

    if len(complete_days) >= 1:
        hist_c = hist[hist["date_dt"].isin(complete_days)].copy()
    else:
        # fallback: keep original behavior
        hist_c = hist

    def _slot_stats(g: pd.DataFrame) -> pd.Series:
        s = g.sort_values("date_dt")["dNetLoad"].values.astype(float)
        if len(s) == 0:
            return pd.Series({"Err_1": 0.0, "Err_mean_7": 0.0, "Err_absmean_7": 0.0, "Err_p10_14": 0.0, "Err_p90_14": 0.0})
        last7 = s[-7:] if len(s) >= 7 else s
        last14 = s[-14:] if len(s) >= 14 else s
        return pd.Series({
            "Err_1": float(s[-1]),
            "Err_mean_7": float(np.mean(last7)),
            "Err_absmean_7": float(np.mean(np.abs(last7))),
            "Err_p10_14": float(np.percentile(last14, 10)),
            "Err_p90_14": float(np.percentile(last14, 90)),
        })

    prof = hist_c.groupby("slot_id").apply(_slot_stats).reset_index()
    return prof


def compute_rt_state_for_date(df_hist: pd.DataFrame, target_date: dt.date) -> Tuple[pd.DataFrame, float]:
    """Compute recent RT state features for a target date.

    IMPORTANT (daily prediction reality):
      - Sometimes the most recent calendar day is incomplete (data not fully available).
      - We therefore compute "recent 7 days" using the most recent *complete* days.
      - A "complete" day is defined as having >=96 unique slot_id with non-null RT.

    Returns:
      state_df with columns: slot_id, RT_mean_7, RT_d1, RT_d7, RT_vol_1d
      rt_vol_1d_daylevel: day-level avg abs(RT_d1-RT_d7)
    """
    hist = df_hist.copy()
    require_cols(hist, ["date_dt", "slot_id", COL_RT], where="RT state")
    hist = hist[hist["date_dt"] < target_date].copy()
    hist[COL_RT] = pd.to_numeric(hist[COL_RT], errors="coerce")
    hist = hist.dropna(subset=[COL_RT, "slot_id", "date_dt"]).copy()

    if hist.empty:
        raise ValueError(f"Not enough history before {target_date} to compute RT state.")

    # Identify complete days based on RT availability
    day_cnt = hist.dropna(subset=[COL_RT]).groupby("date_dt")["slot_id"].nunique().sort_index()
    complete_days = day_cnt[day_cnt >= 96].index.tolist()

    if len(complete_days) >= 1:
        d1 = complete_days[-1]
        hist_c = hist[hist["date_dt"].isin(complete_days)].copy()
    else:
        # fallback: try >=80 slots (legacy)
        ok_days = day_cnt[day_cnt >= 80].index.tolist()
        if not ok_days:
            raise ValueError(f"No sufficiently complete day (<{target_date}) to compute RT state.")
        d1 = ok_days[-1]
        hist_c = hist[hist["date_dt"].isin(ok_days)].copy()

    # Choose d7 as the 7th most recent complete day (if possible)
    if len(complete_days) >= 7:
        d7 = complete_days[-7]
    else:
        d7 = d1 - dt.timedelta(days=6)

    # Per-slot mean of last 7 complete-day RT values
    rt_mean7 = (
        hist_c.sort_values(["date_dt", "slot_id"])
        .groupby("slot_id")[COL_RT]
        .apply(lambda s: float(np.mean(s.values[-7:])) if len(s) >= 1 else np.nan)
        .reset_index(name="RT_mean_7")
    )

    # RT on d1 and d7 (average across possible duplicates)
    rt_d1 = hist[hist["date_dt"] == d1].groupby("slot_id")[COL_RT].mean().reset_index(name="RT_d1")
    rt_d7 = hist[hist["date_dt"] == d7].groupby("slot_id")[COL_RT].mean().reset_index(name="RT_d7")

    state = rt_mean7.merge(rt_d1, on="slot_id", how="left").merge(rt_d7, on="slot_id", how="left")
    state["RT_d7"] = state["RT_d7"].fillna(state["RT_d1"])  # if missing d7, degrade gracefully
    state["RT_vol_1d"] = (state["RT_d1"] - state["RT_d7"]).astype(float)

    rt_vol_1d_daylevel = float(np.nanmean(np.abs(state["RT_vol_1d"].values)))
    return state, rt_vol_1d_daylevel


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
    arm_day_neg_l10: Optional[float] = ARM_DAY_NEG_L10_DEFAULT,
    p_gate_neg_l10: Optional[float] = P_GATE_NEG_L10_DEFAULT,
    p_full_neg_l10: Optional[float] = P_FULL_NEG_L10_DEFAULT,



    # v2.5 spike budget trigger
    spike_budget_enable: bool = SPIKE_BUDGET_ENABLE_DEFAULT,
    spike_score_mode: str = SPIKE_SCORE_MODE_DEFAULT,
    spike_budget_pts: int = SPIKE_BUDGET_PTS_DEFAULT,
    spike_budget_seeds: int = SPIKE_BUDGET_SEEDS_DEFAULT,
    spike_budget_block_radius: int = SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT,
    spike_budget_min_gap: int = SPIKE_BUDGET_MIN_GAP_DEFAULT,
    spike_apply_to_p50: bool = SPIKE_APPLY_TO_P50_DEFAULT,

    # PS damping override (analysis/backtest): overrides bundle ps_alpha WITHOUT retraining
    ps_alpha_override: Optional[float] = None,

    # outputs
    export_trading_brief: bool = True,
    trading_brief_out_path: Optional[str] = None,
) -> None:
    bundle_path = os.path.join(model_dir, "rt_routeB_v2_6.joblib")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}. Run train first.")

    bundle = joblib.load(bundle_path)
    th_spike = float(bundle["th_spike"])
    th_neg = float(bundle["th_neg"])

    b1_models = bundle["b1_models"]
    ps_models = bundle.get("ps_models", None)
    ps_sign_as_load = int(bundle.get("ps_sign_as_load", 1))
    ps_alpha_bundle = float(bundle.get("ps_alpha", 1.0))
    ps_alpha_used = float(ps_alpha_override) if ps_alpha_override is not None else ps_alpha_bundle
    if ps_alpha_override is not None:
        print(f"[PRED] ps_alpha override: {ps_alpha_bundle:.3f} -> {ps_alpha_used:.3f}")
    else:
        print(f"[PRED] ps_alpha used (bundle): {ps_alpha_used:.3f}")
    ps_clip = bundle.get("ps_clip", [-5000.0, 5000.0])

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
    # Day-level context will be computed after PS prediction (need NetLoad_eff_fc)


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

    # --- Pumped-storage (PS) prediction (median) and effective net-load features ---
    df_fc["PS_pred_p50"] = 0.0
    df_fc["PS_pred_p10"] = 0.0
    df_fc["PS_pred_p90"] = 0.0
    df_fc["PS_pred_as_load"] = 0.0

    if ps_models is not None:
        # ensure feature columns exist and are numeric
        for c in FEATURE_PS:
            if c not in df_fc.columns:
                df_fc[c] = 0.0
            df_fc[c] = pd.to_numeric(df_fc[c], errors="coerce").fillna(0.0)

        Xps = df_fc[FEATURE_PS].astype(float).values
        ps_p50 = ps_models["p50"].predict(Xps)
        ps_p10 = ps_models["p10"].predict(Xps)
        ps_p90 = ps_models["p90"].predict(Xps)

        lo, hi = float(ps_clip[0]), float(ps_clip[1])
        df_fc["PS_pred_p50"] = np.clip(ps_p50.astype(float), lo, hi)
        df_fc["PS_pred_p10"] = np.clip(ps_p10.astype(float), lo, hi)
        df_fc["PS_pred_p90"] = np.clip(ps_p90.astype(float), lo, hi)
        df_fc["PS_pred_as_load"] = float(ps_alpha_used) * float(ps_sign_as_load) * df_fc["PS_pred_p50"]

    # Build effective net-load and its ramps from NetLoad_fc + PS_pred_as_load
    df_fc = add_eff_netload_from_ps(df_fc, ps_as_load_col="PS_pred_as_load", out_prefix="")

    # Day-level context for negative-price arming guardrails
    pv_fc_max_day = float(pd.to_numeric(df_fc[COL_PV_FC], errors="coerce").fillna(0.0).max())
    netload_eff_min_day = float(pd.to_numeric(df_fc["NetLoad_eff_fc"], errors="coerce").fillna(0.0).min())

    X1 = df_fc[FEATURE_B1].astype(float).values
    d10 = b1_models["p10"].predict(X1)
    d50 = b1_models["p50"].predict(X1)
    d90 = b1_models["p90"].predict(X1)

    df_fc["NetLoad_act_p10"] = df_fc["NetLoad_fc"] + d10
    df_fc["NetLoad_act_p50"] = df_fc["NetLoad_fc"] + d50
    df_fc["NetLoad_act_p90"] = df_fc["NetLoad_fc"] + d90

    df_fc["NetLoad_eff_act_p10"] = df_fc["NetLoad_act_p10"] + df_fc["PS_pred_as_load"]
    df_fc["NetLoad_eff_act_p50"] = df_fc["NetLoad_act_p50"] + df_fc["PS_pred_as_load"]
    df_fc["NetLoad_eff_act_p90"] = df_fc["NetLoad_act_p90"] + df_fc["PS_pred_as_load"]


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
    nl_ok = bool(np.isfinite(netload_eff_min_day) and (netload_eff_min_day <= float(arm_day_neg_netload_min_th)))
    neg_guard_ok = (pv_ok or nl_ok) if bool(arm_day_neg_use_or) else (pv_ok and nl_ok)

    neg_prob_ok = True
    armed_neg = True
    if arm_day_neg is not None and float(arm_day_neg) > 0:
        neg_prob_ok = bool(np.nanmax(p_neg) >= float(arm_day_neg))
        armed_neg = bool(neg_prob_ok and neg_guard_ok)

    # Optional: separate day-level arming for negative *risk band* (lower10) only.
    # If not provided, it defaults to the same arming as arm_day_neg.
    arm_day_neg_l10_eff = arm_day_neg if (arm_day_neg_l10 is None) else arm_day_neg_l10
    neg_prob_ok_l10 = True
    armed_neg_l10 = armed_neg
    if arm_day_neg_l10_eff is not None and float(arm_day_neg_l10_eff) > 0:
        neg_prob_ok_l10 = bool(np.nanmax(p_neg) >= float(arm_day_neg_l10_eff))
        armed_neg_l10 = bool(neg_prob_ok_l10 and neg_guard_ok)

    p_neg_eff = p_neg if armed_neg else np.zeros_like(p_neg)
    p_neg_eff_l10 = p_neg if armed_neg_l10 else np.zeros_like(p_neg)


    # gate outputs (priority: neg -> spike -> normal)
    take_neg = p_neg_eff >= p_gate_neg

    # separate neg gate for lower10 (risk band) if configured
    p_gate_neg_l10_eff = float(p_gate_neg) if (p_gate_neg_l10 is None) else float(p_gate_neg_l10)
    p_full_neg_l10_eff = p_full_neg if (p_full_neg_l10 is None) else p_full_neg_l10
    take_neg_l10 = p_neg_eff_l10 >= p_gate_neg_l10_eff


    # v2.5 spike budget trigger: select a small number of high-risk slots per day.
    spike_selected: np.ndarray
    spike_intervals: List[Tuple[int, int]] = []
    spike_budget_used = False
    if bool(spike_budget_enable) and int(spike_budget_pts) > 0 and int(spike_budget_seeds) > 0:
        spike_selected, spike_intervals = select_spike_budget_trigger(
            p_spike=p_spike,
            netload_fc=df_fc["NetLoad_fc"].values,
            pv_fc=df_fc["PV_fc"].values,
            rt_u90=rtN90,
            score_mode=str(spike_score_mode),
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
        rt_gate_lower10 = np.where(take_neg_l10, rtG90, np.where(take_spike, rtS50, rtN10))
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

        # lower10 (risk band) can use a separate negative gate (more sensitive if configured)
        if p_full_neg_l10_eff is None or float(p_full_neg_l10_eff) <= float(p_gate_neg_l10_eff):
            w_neg_l10 = take_neg_l10.astype(float)  # hard gate for lower10
        else:
            denom_l10 = max(float(p_full_neg_l10_eff) - float(p_gate_neg_l10_eff), 1e-12)
            w_neg_l10 = np.clip((p_neg_eff_l10 - float(p_gate_neg_l10_eff)) / denom_l10, 0.0, 1.0)

        baseL = np.where(take_spike, rtS50, rtN10)
        baseL = np.where(take_neg_l10, rtN10, baseL)
        rt_gate_lower10 = np.where(take_neg_l10, baseL + w_neg_l10 * (rtG90 - baseL), baseL)
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

        "neg_armed_l10": np.full_like(p_neg, int(armed_neg_l10), dtype=int),
        "neg_prob_ok_l10": np.full_like(p_neg, int(neg_prob_ok_l10), dtype=int),
        "p_gate_neg_l10_eff": np.full_like(p_neg, float(p_gate_neg_l10_eff), dtype=float),
        "p_full_neg_l10_eff": np.full_like(p_neg, float(p_full_neg_l10_eff) if p_full_neg_l10_eff is not None else np.nan, dtype=float),
        "neg_guard_ok": np.full_like(p_neg, int(neg_guard_ok), dtype=int),
        "neg_guard_pv_ok": np.full_like(p_neg, int(pv_ok), dtype=int),
        "neg_guard_netload_ok": np.full_like(p_neg, int(nl_ok), dtype=int),
        "PV_fc_max_day": np.full_like(p_neg, pv_fc_max_day, dtype=float),
        "NetLoad_fc_min_day": np.full_like(p_neg, netload_eff_min_day, dtype=float),
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
        write_df_with_bilingual_header(writer, sheet_name="pred", df=df_out)


    if bool(export_trading_brief):
        tb_path = trading_brief_out_path or _derive_trading_brief_path(out_path)
        export_trading_brief_excel(
            df_out=df_out,
            out_path=tb_path,
            th_spike=th_spike,
            th_neg=th_neg,
            p_gate_neg=float(p_gate_neg),
            p_gate_spike=float(p_gate_spike),
        )
        print(f"[OK] Saved trading brief: {os.path.abspath(tb_path)}")

    print(f"[OK] Saved prediction: {os.path.abspath(out_path)}")



# =========================
# Trading brief export (simple)
# =========================


def _renorm_probs(p_neg: np.ndarray, p_spike: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Renormalize regime probabilities to ensure they sum to 1 and stay in [0,1].
    We treat p_neg and p_spike as independent heads; p_norm is residual.
    If p_neg + p_spike > 1, we scale them down proportionally.
    """
    p_neg = np.clip(np.asarray(p_neg, dtype=float), 0.0, 1.0)
    p_spike = np.clip(np.asarray(p_spike, dtype=float), 0.0, 1.0)
    s = p_neg + p_spike
    # If both heads overshoot, scale down proportionally so sum==1
    scale = np.ones_like(s)
    mask = s > 1.0
    scale[mask] = 1.0 / np.maximum(s[mask], 1e-12)
    p_neg = p_neg * scale
    p_spike = p_spike * scale
    p_norm = 1.0 - p_neg - p_spike
    p_norm = np.clip(p_norm, 0.0, 1.0)
    # Final small renorm for numerical drift
    total = p_norm + p_neg + p_spike
    total = np.where(total <= 0, 1.0, total)
    p_norm /= total
    p_neg /= total
    p_spike /= total
    return p_norm, p_spike, p_neg


def select_spike_budget_trigger(
    p_spike: np.ndarray,
    netload_fc: np.ndarray,
    pv_fc: np.ndarray,
    rt_u90: Optional[np.ndarray] = None,
    score_mode: str = SPIKE_SCORE_MODE_DEFAULT,
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

    base_spike = np.asarray(p_spike, dtype=float).copy()
    mode = str(score_mode or SPIKE_SCORE_MODE_DEFAULT).strip().lower()
    if mode not in ("p_spike", "u90", "hybrid"):
        mode = "p_spike"


    def _rank01(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
        return r

    def _safe_rank01(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros_like(x, dtype=float)
        x2 = x.copy()
        x2[~finite] = np.nanmin(x2[finite])
        return _rank01(x2)

    base_u90 = None if rt_u90 is None else np.asarray(rt_u90, dtype=float)
    if mode == "u90" and (base_u90 is not None) and (len(base_u90) == n):
        base = _safe_rank01(base_u90)
    elif mode == "hybrid" and (base_u90 is not None) and (len(base_u90) == n):
        base = np.maximum(_safe_rank01(base_spike), _safe_rank01(base_u90))
    else:
        # default: keep original behavior (p_spike)
        base = np.nan_to_num(base_spike, nan=0.0, posinf=1.0, neginf=0.0)

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

def _enforce_monotonic(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure quantiles are monotone: q10 <= q50 <= q90 element-wise."""
    q10 = np.asarray(q10, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q90 = np.asarray(q90, dtype=float)
    q50 = np.maximum(q50, q10)
    q90 = np.maximum(q90, q50)
    return q10, q50, q90

def _derive_trading_brief_path(out_path: str) -> str:
    """Derive trading brief path from the full output path."""
    base, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".xlsx"
    return f"{base}_trading_brief{ext}"



def _add_trading_brief_price_chart(
    out_path: str,
    sheet_table: str = "交易简表",
    sheet_summary: str = "当日摘要",
    anchor: str = "D2",
) -> None:
    """Insert an in-sheet line chart into the trading-brief workbook.
    The chart uses existing columns in sheet '交易简表' and is placed on '当日摘要'.
    Safe no-op if openpyxl chart components are unavailable.
    """
    if openpyxl is None or LineChart is None or Reference is None:
        return
    try:
        wb = openpyxl.load_workbook(out_path)
        if sheet_table not in wb.sheetnames or sheet_summary not in wb.sheetnames:
            return
        ws = wb[sheet_table]
        ws_sum = wb[sheet_summary]

        # Header map
        headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
        def _col(name: str) -> Optional[int]:
            try:
                return headers.index(name) + 1
            except Exception:
                return None

        col_time = _col("时刻")
        col_p50 = _col("主预测价P50(元/MWh)")
        col_p90 = _col("风险上界P90(元/MWh)")
        col_p10 = _col("风险下界P10(元/MWh)")
        col_spike = _col("尖峰压力价P90(元/MWh)")

        if not (col_time and col_p50 and col_p90 and col_p10):
            return

        max_row = ws.max_row

        chart = LineChart()
        chart.title = "实时价格预测曲线（P50 / P10 / P90）"
        chart.y_axis.title = "元/MWh"
        chart.x_axis.title = "时刻"
        chart.legend.position = "r"
        chart.width = 28
        chart.height = 14

        cats = Reference(ws, min_col=col_time, min_row=2, max_row=max_row)

        for c in [col_p50, col_p10, col_p90]:
            data = Reference(ws, min_col=c, min_row=1, max_row=max_row)
            chart.add_data(data, titles_from_data=True)

        # Optional: show spike-pressure price as marker-only series (if present)
        if col_spike:
            data = Reference(ws, min_col=col_spike, min_row=1, max_row=max_row)
            chart.add_data(data, titles_from_data=True)
            try:
                s = chart.series[-1]
                s.marker.symbol = "diamond"
                s.marker.size = 6
                s.graphicalProperties.line.noFill = True
            except Exception:
                pass

        chart.set_categories(cats)
        ws_sum.add_chart(chart, anchor)

        # Add a short note if the area is blank
        if ws_sum.cell(13, 1).value is None:
            ws_sum.cell(13, 1).value = "图示：P50为主预测；P10/P90为风险区间；尖峰压力价仅在尖峰标记点展示（若有）。"

        wb.save(out_path)
    except Exception:
        # Never fail the main export due to chart insertion issues
        return

def _configure_matplotlib_cjk_font() -> Optional[str]:
    """Best-effort configure a CJK-capable font for matplotlib (Windows/Linux).
    Returns the chosen font name (or None).
    """
    try:
        import matplotlib as mpl
        from matplotlib import font_manager as fm

        # Common CJK fonts across Windows / macOS / Linux
        candidates = [
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "WenQuanYi Micro Hei",
            "Source Han Sans SC",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        for name in candidates:
            if name in available:
                mpl.rcParams["font.sans-serif"] = [name]
                mpl.rcParams["axes.unicode_minus"] = False
                return name
        return None
    except Exception:
        return None


def _make_trading_brief_chart_sheet(
    out_path: str,
    th_spike: Optional[float] = None,
    th_neg: float = 0.0,
    sheet_table: str = "交易简表",
    sheet_chart: str = "图表",
) -> None:
    """Create a dedicated '图表' sheet with a richer matplotlib curve chart and remove other helper sheets.

    The chart includes:
    - P50 line
    - P10–P90 risk band
    - negative-price segments highlighted
    - optional spike markers & spike threshold line
    """
    if openpyxl is None:
        return

    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import re
        import os
        from openpyxl.drawing.image import Image as XLImage
        from openpyxl.styles import Font, Alignment
    except Exception:
        return

    try:
        df = pd.read_excel(out_path, sheet_name=sheet_table)
        if df is None or len(df) == 0:
            return

        # Column discovery (robust to slight header changes)
        def _find_col(rx: str) -> Optional[str]:
            for c in df.columns:
                if re.search(rx, str(c)):
                    return c
            return None

        col_time = _find_col(r"时刻") or "时刻"
        col_p50 = _find_col(r"P50") or _find_col(r"主预测")  # fallback
        col_p10 = _find_col(r"P10")
        col_p90 = _find_col(r"P90")  # the first P90 likely upper bound
        col_spike_flag = _find_col(r"尖峰风险标记")
        col_spike_p90 = _find_col(r"尖峰压力价P90")

        if col_p50 is None or col_time not in df.columns:
            return

        times = df[col_time].astype(str).tolist()
        p50 = pd.to_numeric(df[col_p50], errors="coerce").values.astype(float)
        p10 = pd.to_numeric(df[col_p10], errors="coerce").values.astype(float) if col_p10 else None
        p90 = pd.to_numeric(df[col_p90], errors="coerce").values.astype(float) if col_p90 else None

        spike_flag = pd.to_numeric(df[col_spike_flag], errors="coerce").fillna(0).astype(int).values if col_spike_flag else None
        spike_p90 = pd.to_numeric(df[col_spike_p90], errors="coerce").values.astype(float) if col_spike_p90 else None

        # Configure CJK font (best effort)
        _configure_matplotlib_cjk_font()

        x = np.arange(len(times))
        fig = plt.figure(figsize=(14, 6))

        if p10 is not None and p90 is not None:
            plt.fill_between(x, p10, p90, alpha=0.18, label="风险区间 P10–P90")

        plt.plot(x, p50, linewidth=2.0, label="主预测价 P50")

        # Spike markers
        if spike_flag is not None and spike_p90 is not None and int(np.nansum(spike_flag)) > 0:
            idx = np.where(spike_flag == 1)[0]
            plt.scatter(idx, spike_p90[idx], marker="D", s=40, label="尖峰压力价 P90（标记点）")

        # Reference lines
        plt.axhline(float(th_neg), linewidth=1.2, linestyle="--", label=f"{th_neg:g} 元/MWh（负价分界）")
        # Only show spike threshold if it is not too far above current range,
        # otherwise it will compress the visible curve and reduce readability.
        if th_spike is not None and np.isfinite(th_spike):
            show_spike_th = False
            if spike_flag is not None and int(np.nansum(spike_flag)) > 0:
                show_spike_th = True
            elif p90 is not None and np.isfinite(p90).any():
                if float(th_spike) <= float(np.nanmax(p90)) * 1.20:
                    show_spike_th = True
            if show_spike_th:
                plt.axhline(float(th_spike), linewidth=1.0, linestyle=":", label=f"{th_spike:g} 元/MWh（尖峰阈值）")

        # Highlight negative segments (based on P50)
        neg_idx = np.where(p50 < float(th_neg))[0]
        if len(neg_idx) > 0:
            spans = []
            start = int(neg_idx[0])
            prev = int(neg_idx[0])
            for i in neg_idx[1:]:
                i = int(i)
                if i == prev + 1:
                    prev = i
                else:
                    spans.append((start, prev))
                    start = prev = i
            spans.append((start, prev))
            for (a, b) in spans:
                plt.axvspan(a - 0.5, b + 0.5, alpha=0.08)

        # Hour ticks for clarity
        tick_idx = [i for i, t in enumerate(times) if re.match(r"^\d{2}:00$", str(t))]
        if "24:00" in times:
            tick_idx = [i for i in tick_idx if times[i] != "24:00"] + [times.index("24:00")]
        tick_labels = [times[i] for i in tick_idx]
        plt.xticks(tick_idx, tick_labels, rotation=45, ha="right")

        plt.grid(True, alpha=0.25)
        plt.xlabel("时刻（15分钟）")
        plt.ylabel("元/MWh")

        # Title: try to use '日期' column if present
        date_str = None
        if "日期" in df.columns:
            date_str = str(df["日期"].iloc[0])[:10]
        title = f"{date_str or ''} 实时价格预测（含风险区间、负价段高亮、尖峰标记）".strip()
        plt.title(title)

        # Annotate max/min
        if np.isfinite(p50).any():
            imax = int(np.nanargmax(p50))
            imin = int(np.nanargmin(p50))
            plt.scatter([imax, imin], [p50[imax], p50[imin]], s=50)
            plt.annotate(f"最高 {p50[imax]:.1f}\n{times[imax]}", (imax, p50[imax]), textcoords="offset points", xytext=(10, 10))
            plt.annotate(f"最低 {p50[imin]:.1f}\n{times[imin]}", (imin, p50[imin]), textcoords="offset points", xytext=(10, -30))

        plt.legend(loc="upper right")
        plt.margins(x=0.01)
        plt.tight_layout()

        png_path = os.path.splitext(out_path)[0] + "_chart.png"
        plt.savefig(png_path, dpi=200)
        plt.close(fig)

        # Now embed into Excel on a new sheet named '图表', and remove other helper sheets
        wb = openpyxl.load_workbook(out_path)

        # remove all sheets except the main table
        for sname in list(wb.sheetnames):
            if sname != sheet_table and sname != sheet_chart:
                wb.remove(wb[sname])

        # recreate chart sheet at position 2
        if sheet_chart in wb.sheetnames:
            wb.remove(wb[sheet_chart])
        ws_chart = wb.create_sheet(sheet_chart, 1)

        # Summary text
        ws_chart["A1"] = (date_str or "") + " D+1 实时价格预测图（交易简报）"
        ws_chart["A1"].font = Font(size=14, bold=True)
        ws_chart["A1"].alignment = Alignment(horizontal="left", vertical="center")

        neg_cnt = int(np.nansum(p50 < float(th_neg)))
        spike_cnt = int(np.nansum(spike_flag)) if spike_flag is not None else 0
        max_val = float(np.nanmax(p50)); max_t = times[int(np.nanargmax(p50))]
        min_val = float(np.nanmin(p50)); min_t = times[int(np.nanargmin(p50))]

        ws_chart["A2"] = f"P50 最高值：{max_val:.1f} 元/MWh（{max_t}）；最低值：{min_val:.1f} 元/MWh（{min_t}）"
        ws_chart["A3"] = f"负价点数：{neg_cnt} / {len(p50)}（约 {neg_cnt/4:.2f} 小时）；尖峰标记点数：{spike_cnt} / {len(p50)}"
        ws_chart["A2"].font = Font(size=11)
        ws_chart["A3"].font = Font(size=11)

        # Insert image
        img = XLImage(png_path)
        img.width = 1200
        img.height = 520
        ws_chart.add_image(img, "A5")
        ws_chart.column_dimensions["A"].width = 120

        wb.save(out_path)

    except Exception:
        # Do not break the main export due to chart-sheet rendering issues
        return



def export_trading_brief_excel(
    df_out: pd.DataFrame,
    out_path: str,
    th_spike: float,
    th_neg: float,
    p_gate_neg: float,
    p_gate_spike: float,
) -> None:
    """
    Export a compact Excel for the trading desk.
    - Sheet1: 交易简表 (96 rows)
    - Sheet2: 图表（曲线图，给交易团队更直观地看趋势/风险区间/负价段/尖峰标记）
    """
    df = df_out.copy()

    # pick the final P50 used for day-to-day execution
    p50_col = "RT_spike_budget_p50" if "RT_spike_budget_p50" in df.columns else "RT_gate_p50"
    u90_col = "RT_spike_budget_upper90" if "RT_spike_budget_upper90" in df.columns else "RT_gate_upper90"
    l10_col = "RT_gate_lower10" if "RT_gate_lower10" in df.columns else None

    if p50_col not in df.columns:
        raise KeyError(f"Missing required column for trading brief: {p50_col}")

    # flags
    neg_armed = pd.to_numeric(df.get("neg_armed", 0), errors="coerce").fillna(0).astype(int).values
    spike_flag = pd.to_numeric(df.get("spike_selected", 0), errors="coerce").fillna(0).astype(int).values
    p_neg = pd.to_numeric(df.get("p_neg", 0.0), errors="coerce").fillna(0.0).values

    cond_spike = spike_flag == 1
    cond_neg_risk = (neg_armed == 1) & (p_neg >= float(p_gate_neg))

    action = np.select(
        [cond_spike, cond_neg_risk],
        ["尖峰风险：建议用压力价P90防守", "负价风险：关注负价情景与风光预测"],
        default="常规：按主预测价执行",
    ).astype(str)

    brief = pd.DataFrame({
        "日期": df["日期"].values if "日期" in df.columns else df.get(COL_DATE, pd.Series([""] * len(df))).values,
        "时刻": df["时刻"].values if "时刻" in df.columns else df.get("time_str", pd.Series([""] * len(df))).values,
        "主预测价P50(元/MWh)": pd.to_numeric(df[p50_col], errors="coerce").values,
        "风险上界P90(元/MWh)": pd.to_numeric(df.get(u90_col, df[p50_col]), errors="coerce").values,
        "风险下界P10(元/MWh)": pd.to_numeric(df[l10_col], errors="coerce").values if l10_col else np.nan,
        "负价日开关": neg_armed,
        "尖峰风险标记": spike_flag,
        "尖峰压力价P90(元/MWh)": pd.to_numeric(df.get(u90_col, df[p50_col]), errors="coerce").values,
        "操作建议": action,
    })

    # Summary
    p50 = pd.to_numeric(brief["主预测价P50(元/MWh)"], errors="coerce")
    tcol = brief["时刻"].astype(str)
    try:
        i_min = int(np.nanargmin(p50.values))
        i_max = int(np.nanargmax(p50.values))
        p50_min = float(p50.iloc[i_min])
        p50_max = float(p50.iloc[i_max])
        t_min = str(tcol.iloc[i_min])
        t_max = str(tcol.iloc[i_max])
    except Exception:
        p50_min = np.nan
        p50_max = np.nan
        t_min = ""
        t_max = ""

    date_val = str(brief["日期"].iloc[0]) if len(brief) else ""

    summary_rows = [
        ("预测日期", date_val),
        ("主预测P50最低价", p50_min),
        ("最低价时刻", t_min),
        ("主预测P50最高价", p50_max),
        ("最高价时刻", t_max),
        ("尖峰阈值TH", float(th_spike)),
        ("负价阈值TH", float(th_neg)),
        ("尖峰风险点数", int(np.nansum(spike_flag))),
        ("负价日开关", int(np.nanmax(neg_armed) if len(neg_armed) else 0)),
        ("负价风险点数(按p_gate_neg)", int(np.nansum(cond_neg_risk))),
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["指标", "值"])

    df_guide = pd.DataFrame([
        ("主预测价P50", p50_col, "日常执行的主曲线（默认使用 gate 或 spike_budget_p50）。"),
        ("风险上界P90/下界P10", f"{u90_col}/{l10_col}", "用于风控限额、敏感性评估与报价保护。"),
        ("负价日开关", "neg_armed", "当日是否允许负价场景触发（1=允许；0=不触发负价 gate）。"),
        ("尖峰风险标记", "spike_selected", "尖峰预算触发挑选出的少数风险点；建议用压力价做防守。"),
        ("尖峰压力价P90", u90_col, "仅对 spike_selected=1 的点作为“压力情景价”（风控/限额/保护）。"),
    ], columns=["给交易看的字段", "原字段名", "如何使用"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        brief.to_excel(w, sheet_name="交易简表", index=False)

    # Build a dedicated chart sheet, and remove other helper sheets (per trading-desk preference)
    _make_trading_brief_chart_sheet(out_path, th_spike=th_spike, th_neg=th_neg)

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
    arm_day_neg_l10: Optional[float] = ARM_DAY_NEG_L10_DEFAULT,
    p_gate_neg_l10: Optional[float] = P_GATE_NEG_L10_DEFAULT,
    p_full_neg_l10: Optional[float] = P_FULL_NEG_L10_DEFAULT,

    # v2.5 spike budget trigger
    spike_budget_enable: bool = SPIKE_BUDGET_ENABLE_DEFAULT,
    spike_score_mode: str = SPIKE_SCORE_MODE_DEFAULT,
    spike_budget_pts: int = SPIKE_BUDGET_PTS_DEFAULT,
    spike_budget_seeds: int = SPIKE_BUDGET_SEEDS_DEFAULT,
    spike_budget_block_radius: int = SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT,
    spike_budget_min_gap: int = SPIKE_BUDGET_MIN_GAP_DEFAULT,
    spike_apply_to_p50: bool = SPIKE_APPLY_TO_P50_DEFAULT,

    # PS damping override (analysis/backtest): overrides bundle ps_alpha WITHOUT retraining
    ps_alpha_override: Optional[float] = None,
) -> None:
    """
    Backtest using forecast columns from history itself:
      For each date D, use D's forecast columns + flags as forecast input and predict D.
    """
    hist = read_table(history_path)
    hist = add_slot_calendar(hist)
    require_cols(hist, [COL_LOAD_FC, COL_TIE_FC, COL_WIND_FC, COL_PV_FC, COL_RT], where="history backtest base")
    # Ensure optional generation columns exist for consistent NetLoad_fc (missing -> 0.0)
    for _c in [COL_NON_MARKET_NUCLEAR_FC, COL_SELF_SUPPLY_UNITS_FC, COL_LOCAL_POWER_PLANT_FC]:
        if _c not in hist.columns:
            hist[_c] = 0.0

    hist[COL_RT] = pd.to_numeric(hist[COL_RT], errors="coerce")
    hist["时刻"] = hist["time_str"]

    # Load thresholds from trained model (align backtest labels with training thresholds)
    th_spike = SPIKE_THRESHOLD_RT_DEFAULT
    th_neg = NEG_THRESHOLD_RT_DEFAULT
    ps_alpha_bundle = np.nan
    bundle_path = os.path.join(model_dir, "rt_routeB_v2_6.joblib")
    if os.path.exists(bundle_path):
        try:
            _bundle = joblib.load(bundle_path)
            th_spike = float(_bundle.get("th_spike", th_spike))
            th_neg = float(_bundle.get("th_neg", th_neg))
            ps_alpha_bundle = float(_bundle.get("ps_alpha", ps_alpha_bundle))
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

        df_fc = df_day[[
            COL_DATE, "时刻", COL_HOL, COL_WKND,
            COL_LOAD_FC, COL_TIE_FC, COL_WIND_FC, COL_PV_FC,
            COL_NON_MARKET_NUCLEAR_FC, COL_SELF_SUPPLY_UNITS_FC, COL_LOCAL_POWER_PLANT_FC,
        ]].copy()

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
            arm_day_neg_l10=arm_day_neg_l10,
            p_gate_neg_l10=p_gate_neg_l10,
            p_full_neg_l10=p_full_neg_l10,
            spike_budget_enable=spike_budget_enable,
            spike_score_mode=str(spike_score_mode),
            spike_budget_pts=spike_budget_pts,
            spike_budget_seeds=spike_budget_seeds,
            spike_budget_block_radius=spike_budget_block_radius,
            spike_budget_min_gap=spike_budget_min_gap,
            spike_apply_to_p50=spike_apply_to_p50,
            ps_alpha_override=ps_alpha_override,
            export_trading_brief=False,
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
        write_df_with_bilingual_header(w, sheet_name="per_day", df=df_sum)
        for d, df in details.items():
            sheet = d.replace("-", "")[:31]
            write_df_with_bilingual_header(w, sheet_name=sheet, df=df)

        # --- run_info: capture parameters & date coverage to avoid confusion when reusing filenames ---
        try:
            run_info = {
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "history_path": os.path.abspath(history_path),
                "model_dir": os.path.abspath(model_dir),
                "bundle_path": os.path.abspath(os.path.join(model_dir, "rt_routeB_v2_6.joblib")),
                "th_spike": float(th_spike),
                "th_neg": float(th_neg),
                "ps_alpha_bundle": (float(ps_alpha_bundle) if ps_alpha_bundle == ps_alpha_bundle else np.nan),
                "ps_alpha_override": (float(ps_alpha_override) if ps_alpha_override is not None else np.nan),
                "ps_alpha_used": (float(ps_alpha_override) if ps_alpha_override is not None else (float(ps_alpha_bundle) if ps_alpha_bundle == ps_alpha_bundle else np.nan)),
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
            write_df_with_bilingual_header(w, sheet_name="run_info", df=df_run)
        except Exception:
            pass

    print(f"[OK] Saved backtest report: {os.path.abspath(out_path)}")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train Route-B v2.6 models and save to model_dir")
    p_train.add_argument("--history", required=True, help="History CSV/XLSX with full columns")
    p_train.add_argument("--model_dir", required=True, help="Directory to save models")
    p_train.add_argument("--th_spike", type=float, default=SPIKE_THRESHOLD_RT_DEFAULT, help="Spike threshold (default 800)")
    p_train.add_argument("--th_neg", type=float, default=NEG_THRESHOLD_RT_DEFAULT, help="Negative threshold (default 0)")
    p_train.add_argument("--ps_alpha", type=float, default=0.6, help="Damping for pumped-storage effect (0..1). Use <1 to reduce noise.")

    p_pred = sub.add_parser("predict", help="Predict RT for one target date (D+1)")
    p_pred.add_argument("--history", required=True, help="History CSV/XLSX with full columns")
    p_pred.add_argument("--model_dir", required=True, help="Directory containing saved models")
    p_pred.add_argument("--forecast", required=True, help="Forecast CSV/XLSX for D+1 (96 rows for target date)")
    p_pred.add_argument("--date", required=True, help="Target date, e.g. 2026-01-01")
    p_pred.add_argument("--out", required=True, help="Output Excel path")
    p_pred.add_argument("--p_gate_spike", type=float, default=P_GATE_SPIKE_DEFAULT, help="Gate threshold for spike")
    p_pred.add_argument("--p_gate_neg", type=float, default=P_GATE_NEG_DEFAULT, help="Gate threshold for negative")
    p_pred.add_argument("--ps_alpha_override", type=float, default=None, help="Override PS damping ps_alpha during prediction/backtest (no retraining).")
    p_pred.add_argument("--p_full_neg", type=float, default=P_FULL_NEG_DEFAULT, help="Ramp gate: full negative weight threshold (<= p_gate_neg disables ramp)")
    p_pred.add_argument("--arm_day_neg", type=float, default=ARM_DAY_NEG_DEFAULT, help="Day-level arming threshold for negative gate (<=0 disables)")
    p_pred.add_argument("--arm_day_neg_pv_th", type=float, default=ARM_DAY_NEG_PV_TH_DEFAULT, help="PV_fc daily max threshold for negative-day arming guard")
    p_pred.add_argument("--arm_day_neg_netload_min_th", type=float, default=ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT, help="NetLoad_fc daily min threshold for negative-day arming guard")
    p_pred.add_argument("--arm_day_neg_use_or", type=int, default=1, help="Use OR guard (1) or AND guard (0) for (PV vs NetLoad)")
    p_pred.add_argument("--arm_day_neg_l10", type=float, default=ARM_DAY_NEG_L10_DEFAULT,
                        help="Optional: day-level arming threshold for negative risk band (lower10); None=>use arm_day_neg")
    p_pred.add_argument("--p_gate_neg_l10", type=float, default=P_GATE_NEG_L10_DEFAULT,
                        help="Optional: p_neg gate threshold for lower10; None=>use p_gate_neg")
    p_pred.add_argument("--p_full_neg_l10", type=float, default=P_FULL_NEG_L10_DEFAULT,
                        help="Optional: full negative gate threshold for lower10; None=>use p_full_neg")


    # v2.5 spike budget trigger
    p_pred.add_argument("--spike_budget_enable", type=int, default=int(SPIKE_BUDGET_ENABLE_DEFAULT), help="Enable spike budget trigger (1/0)")
    p_pred.add_argument("--spike_score_mode", type=str, default=SPIKE_SCORE_MODE_DEFAULT, choices=["p_spike","u90","hybrid"], help="Spike budget scoring base: p_spike (legacy), u90 (rank(RT_norm_p90)), hybrid (max(rank(p_spike),rank(u90)))")
    p_pred.add_argument("--spike_budget_pts", type=int, default=SPIKE_BUDGET_PTS_DEFAULT, help="Max number of 15-min slots flagged as spike per day")
    p_pred.add_argument("--spike_budget_seeds", type=int, default=SPIKE_BUDGET_SEEDS_DEFAULT, help="Number of spike seed slots before expansion")
    p_pred.add_argument("--spike_budget_block_radius", type=int, default=SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT, help="Expand each seed by +/- radius (15-min slots)")
    p_pred.add_argument("--spike_budget_min_gap", type=int, default=SPIKE_BUDGET_MIN_GAP_DEFAULT, help="Minimum separation between spike seeds (15-min slots)")
    p_pred.add_argument("--spike_apply_to_p50", type=int, default=int(SPIKE_APPLY_TO_P50_DEFAULT), help="If 1, spike budget affects RT_gate_p50; if 0, only affects risk bounds & scenario columns")

    # trading brief (two-file output)
    p_pred.add_argument("--trading_brief", type=int, default=1, help="If 1, also export a simplified trading brief Excel")
    p_pred.add_argument("--trading_out", default="", help="Optional trading brief output path (default derived from --out)")


    p_bt = sub.add_parser("backtest", help="Backtest a list of dates using forecast columns from history")
    p_bt.add_argument("--history", required=True, help="History CSV/XLSX with full columns")
    p_bt.add_argument("--model_dir", required=True, help="Directory containing saved models")
    p_bt.add_argument("--dates", required=True, help="Comma-separated dates, e.g. 2025-11-08,2025-01-26")
    p_bt.add_argument("--ps_alpha_override", type=float, default=None, help="Override PS damping ps_alpha during backtest (no retraining).")
    p_bt.add_argument("--out", required=True, help="Output Excel report path")
    p_bt.add_argument("--p_gate_spike", type=float, default=P_GATE_SPIKE_DEFAULT, help="Gate threshold for spike")
    p_bt.add_argument("--p_gate_neg", type=float, default=P_GATE_NEG_DEFAULT, help="Gate threshold for negative")
    p_bt.add_argument("--arm_day_neg", type=float, default=ARM_DAY_NEG_DEFAULT, help="Day-level arming threshold for negative gate (<=0 disables)")
    p_bt.add_argument("--arm_day_neg_pv_th", type=float, default=ARM_DAY_NEG_PV_TH_DEFAULT, help="PV_fc daily max threshold for negative-day arming guard")
    p_bt.add_argument("--arm_day_neg_netload_min_th", type=float, default=ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT, help="NetLoad_fc daily min threshold for negative-day arming guard")
    p_bt.add_argument("--arm_day_neg_use_or", type=int, default=1, help="Use OR guard (1) or AND guard (0) for (PV vs NetLoad)")
    p_bt.add_argument("--arm_day_neg_l10", type=float, default=ARM_DAY_NEG_L10_DEFAULT,
                     help="Optional: day-level arming threshold for negative risk band (lower10); None=>use arm_day_neg")
    p_bt.add_argument("--p_gate_neg_l10", type=float, default=P_GATE_NEG_L10_DEFAULT,
                     help="Optional: p_neg gate threshold for lower10; None=>use p_gate_neg")
    p_bt.add_argument("--p_full_neg_l10", type=float, default=P_FULL_NEG_L10_DEFAULT,
                     help="Optional: full negative gate threshold for lower10; None=>use p_full_neg")


    # v2.5 spike budget trigger
    p_bt.add_argument("--spike_budget_enable", type=int, default=int(SPIKE_BUDGET_ENABLE_DEFAULT), help="Enable spike budget trigger (1/0)")
    p_bt.add_argument("--spike_score_mode", type=str, default=SPIKE_SCORE_MODE_DEFAULT, choices=["p_spike","u90","hybrid"], help="Spike budget scoring base: p_spike (legacy), u90 (rank(RT_norm_p90)), hybrid (max(rank(p_spike),rank(u90)))")
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
        "cmd": "predict",  # "train" | "predict" | "backtest"
        "history": r"data/价格预测数据集.csv",
        "model_dir": r"models_v2_6",
        "th_spike": 800.0,
        "th_neg": 0.0,
        "ps_alpha": 0.45,  # pumped-storage effect damping (0..1)
        "ps_alpha_override": None,  # if set, overrides bundle ps_alpha for predict/backtest (no retrain)

        # gate thresholds
        "p_gate_spike": 0.03,
        "p_gate_neg": 0.30,  # tune here or pass --p_gate_neg
        "p_full_neg": 0.55,  # ramp gate full neg prob (<=p_gate_neg => hard gate)

        # v2.5 spike budget trigger
        "spike_budget_enable": 1,
        "spike_score_mode": "p_spike",
        "spike_budget_pts": 6,
        "spike_budget_seeds": 2,
        "spike_budget_block_radius": 1,
        "spike_budget_min_gap": 4,
        "spike_apply_to_p50": 0,  # 0=稳健(不改p50); 1=让尖峰预算影响RT_gate_p50

        "arm_day_neg": 0.65,  # 日级激活阈值：只有 max(p_neg) >= 0.65 才允许负价门控
        "arm_day_neg_use_or": 1,  # 1=OR
        "arm_day_neg_l10": 0.58,  # 可选：仅对lower10负价风险带启用的日级阈值（None=>与arm_day_neg一致）
        "p_gate_neg_l10": 0.22,  # 可选：lower10负价门控阈值（None=>与p_gate_neg一致）
        "p_full_neg_l10": 0.45,  # 可选：lower10负价满权重阈值（None=>与p_full_neg一致）
        "arm_day_neg_pv_th": 9000,  # PV 日最大值阈值（MW）：适当下调，避免冬季负价日被 PV 条件卡死
        "arm_day_neg_netload_min_th": 23500,  # NetLoad_eff 日最小值阈值（MW）：从 7000 提到合理量级（核心）

        # predict
        "forecast": r"forecast_input_2026-01-10.xlsx",
        "date": "2026-01-10",
        "out": r"outPut/forcast_2026-01-10.xlsx",
        "trading_brief": 1,
        "trading_out": r"",

        # backtest
        "dates": "2026-01-01,2026-01-02,2026-01-03,2026-01-04,2026-01-05,2026-01-06,2026-01-07,2026-01-08,2025-12-31,2025-12-30,2025-12-29,2025-12-28,2025-12-26,2025-12-25,2025-12-22,2025-12-13,2025-11-08,2025-10-30,2025-10-15,2025-10-12,2025-10-08,2025-10-05,2025-10-04,2025-09-21",

        "report_out": r"outPut/bt_202601091617.xlsx",
    }

    if CONFIG["cmd"] == "train":
        train_and_save(CONFIG["history"], CONFIG["model_dir"], th_spike=float(CONFIG["th_spike"]), th_neg=float(CONFIG["th_neg"]), ps_alpha=float(CONFIG.get("ps_alpha", 0.6)))
    elif CONFIG["cmd"] == "predict":
        predict_one_day(
            CONFIG["history"], CONFIG["model_dir"], CONFIG["forecast"], CONFIG["date"], CONFIG["out"],
            p_gate_spike=float(CONFIG["p_gate_spike"]), p_gate_neg=float(CONFIG["p_gate_neg"]),
            p_full_neg=float(CONFIG.get("p_full_neg", P_FULL_NEG_DEFAULT)),
            arm_day_neg=float(CONFIG.get("arm_day_neg", ARM_DAY_NEG_DEFAULT)),
            arm_day_neg_pv_th=float(CONFIG.get("arm_day_neg_pv_th", ARM_DAY_NEG_PV_TH_DEFAULT)),
            arm_day_neg_netload_min_th=float(CONFIG.get("arm_day_neg_netload_min_th", ARM_DAY_NEG_NETLOAD_MIN_TH_DEFAULT)),
            arm_day_neg_use_or=bool(int(CONFIG.get("arm_day_neg_use_or", 1))),
            arm_day_neg_l10=CONFIG.get("arm_day_neg_l10", ARM_DAY_NEG_L10_DEFAULT),
            p_gate_neg_l10=CONFIG.get("p_gate_neg_l10", P_GATE_NEG_L10_DEFAULT),
            p_full_neg_l10=CONFIG.get("p_full_neg_l10", P_FULL_NEG_L10_DEFAULT),
            spike_budget_enable=bool(int(CONFIG.get("spike_budget_enable", int(SPIKE_BUDGET_ENABLE_DEFAULT)))),
            spike_score_mode=str(CONFIG.get("spike_score_mode", SPIKE_SCORE_MODE_DEFAULT)),
            spike_budget_pts=int(CONFIG.get("spike_budget_pts", SPIKE_BUDGET_PTS_DEFAULT)),
            spike_budget_seeds=int(CONFIG.get("spike_budget_seeds", SPIKE_BUDGET_SEEDS_DEFAULT)),
            spike_budget_block_radius=int(CONFIG.get("spike_budget_block_radius", SPIKE_BUDGET_BLOCK_RADIUS_DEFAULT)),
            spike_budget_min_gap=int(CONFIG.get("spike_budget_min_gap", SPIKE_BUDGET_MIN_GAP_DEFAULT)),
            spike_apply_to_p50=bool(int(CONFIG.get("spike_apply_to_p50", int(SPIKE_APPLY_TO_P50_DEFAULT)))),
            export_trading_brief=bool(int(CONFIG.get("trading_brief", 1))),
            trading_brief_out_path=(str(CONFIG.get("trading_out", "")).strip() or None),
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
            arm_day_neg_l10=CONFIG.get("arm_day_neg_l10", ARM_DAY_NEG_L10_DEFAULT),
            p_gate_neg_l10=CONFIG.get("p_gate_neg_l10", P_GATE_NEG_L10_DEFAULT),
            p_full_neg_l10=CONFIG.get("p_full_neg_l10", P_FULL_NEG_L10_DEFAULT),
            spike_budget_enable=bool(int(CONFIG.get("spike_budget_enable", int(SPIKE_BUDGET_ENABLE_DEFAULT)))),
            spike_score_mode=str(CONFIG.get("spike_score_mode", SPIKE_SCORE_MODE_DEFAULT)),
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
        train_and_save(args.history, args.model_dir, th_spike=float(args.th_spike), th_neg=float(args.th_neg), ps_alpha=float(args.ps_alpha))
    elif args.cmd == "predict":
        predict_one_day(
            args.history, args.model_dir, args.forecast, args.date, args.out,
            p_gate_spike=float(args.p_gate_spike), p_gate_neg=float(args.p_gate_neg),
            p_full_neg=float(args.p_full_neg),
            arm_day_neg=float(args.arm_day_neg),
            arm_day_neg_pv_th=float(args.arm_day_neg_pv_th),
            arm_day_neg_netload_min_th=float(args.arm_day_neg_netload_min_th),
            arm_day_neg_use_or=bool(int(args.arm_day_neg_use_or)),
            arm_day_neg_l10=args.arm_day_neg_l10,
            p_gate_neg_l10=args.p_gate_neg_l10,
            p_full_neg_l10=args.p_full_neg_l10,
            spike_budget_enable=bool(int(args.spike_budget_enable)),
            spike_score_mode=str(args.spike_score_mode),
            spike_budget_pts=int(args.spike_budget_pts),
            spike_budget_seeds=int(args.spike_budget_seeds),
            spike_budget_block_radius=int(args.spike_budget_block_radius),
            spike_budget_min_gap=int(args.spike_budget_min_gap),
            spike_apply_to_p50=bool(int(args.spike_apply_to_p50)),
            ps_alpha_override=(args.ps_alpha_override if args.ps_alpha_override is not None else None),
            export_trading_brief=bool(int(args.trading_brief)),
            trading_brief_out_path=(args.trading_out if str(args.trading_out).strip() else None),
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
            spike_score_mode=str(args.spike_score_mode),
            spike_budget_pts=int(args.spike_budget_pts),
            spike_budget_seeds=int(args.spike_budget_seeds),
            spike_budget_block_radius=int(args.spike_budget_block_radius),
            spike_budget_min_gap=int(args.spike_budget_min_gap),
            spike_apply_to_p50=bool(int(args.spike_apply_to_p50)),
            ps_alpha_override=(args.ps_alpha_override if args.ps_alpha_override is not None else None),
        )
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()