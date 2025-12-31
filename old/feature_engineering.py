# 日型、特征工程

import numpy as np
import pandas as pd
from config import *

def compute_daily_features(df):
    daily = (
        df.groupby(COL_DATE)
          .agg({
              COL_LOAD_FC: "mean",
              COL_WIND_FC: "mean",
              COL_PV_FC: "mean",
              "is_weekend": "max",
              "is_holiday": "max"
          })
          .reset_index()
    )

    daily["pv_ratio"] = daily[COL_PV_FC] / daily[COL_LOAD_FC]
    daily["wind_ratio"] = daily[COL_WIND_FC] / daily[COL_LOAD_FC]
    # 给 pv_refine 用的英文别名（避免 KeyError）
    daily["load_mean"] = daily[COL_LOAD_FC]
    daily["pv_mean"] = daily[COL_PV_FC]
    daily["wind_mean"] = daily[COL_WIND_FC]

    return daily


def classify_day_type(daily):
    pv_th   = daily["pv_ratio"].quantile(PV_RATIO_Q)
    wind_th = daily["wind_ratio"].quantile(WIND_RATIO_Q)
    load_th = daily[COL_LOAD_FC].quantile(LOAD_Q)

    def _classify(row):
        if row["pv_ratio"] >= pv_th:
            return "PV_DOMINANT"
        elif row["wind_ratio"] >= wind_th:
            return "WIND_DOMINANT"
        elif row[COL_LOAD_FC] >= load_th:
            return "HIGH_LOAD"
        else:
            return "CONVENTIONAL"

    daily["day_type"] = daily.apply(_classify, axis=1)
    return daily
