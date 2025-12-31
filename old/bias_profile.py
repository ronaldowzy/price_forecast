# RT-DA 偏差画像

# bias_profile.py
# RT-DA 偏差画像（升级：day_type + hour + net_bin 分层）

import numpy as np
import pandas as pd
from config import *

def _safe_percentile(a, q):
    a = np.asarray(a)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan
    return float(np.percentile(a, q))

def build_bias_profile(df: pd.DataFrame):
    df = df.copy()

    # 1) 计算 RT-DA 偏差
    df["rt_da_diff"] = df[COL_RT_PRICE] - df[COL_DA_PRICE]

    # 2) 构造“净负荷”（全部使用 D-1 可得的预测字段）
    #    注意：这些列在你的数据集中存在；若未来有缺列，可在这里加 errors='ignore' 或缺省为0
    df["net_load_fc"] = (
        df[COL_LOAD_FC]
        - df[COL_WIND_FC]
        - df[COL_PV_FC]
        - df.get("非市场化核电总加(预测)", 0.0)
        - df.get("地方电厂发电总加(预测)", 0.0)
        - df.get("自备机组总加(预测)", 0.0)
        - df.get(COL_TIE_FC, 0.0)
    )

    # 3) 对每个 (day_type, hour) 计算净负荷分层阈值：33% 与 67%
    th = (
        df.groupby(["day_type", "hour"])["net_load_fc"]
          .agg(
              q33=lambda s: _safe_percentile(s.values, 33),
              q67=lambda s: _safe_percentile(s.values, 67),
              n="count"
          )
          .reset_index()
    )

    # 4) 给每条样本打上 net_bin：LOW / MID / HIGH
    df = df.merge(th[["day_type", "hour", "q33", "q67"]], on=["day_type", "hour"], how="left")

    def assign_bin(r):
        x = r["net_load_fc"]
        q33 = r["q33"]
        q67 = r["q67"]
        if np.isnan(x) or np.isnan(q33) or np.isnan(q67):
            return "MID"
        if x < q33:
            return "LOW"
        elif x >= q67:
            return "HIGH"
        else:
            return "MID"

    df["net_bin"] = df.apply(assign_bin, axis=1)

    # 5) 画像：按 (day_type, hour, net_bin) 聚合 RT-DA 的分位数
    profile = (
        df.groupby(["day_type", "hour", "net_bin"])["rt_da_diff"]
          .agg(
              p50=lambda x: np.percentile(x, 50),
              p90=lambda x: np.percentile(x, 90),
              p10=lambda x: np.percentile(x, 10),
              std="std",
              count="count"
          )
          .reset_index()
    )

    # 6) stability 标签（仍沿用 std 中位数规则）
    std_th = profile["std"].median()
    profile["stability"] = np.where(profile["std"] <= std_th, "STABLE", "VOLATILE")

    # 返回：画像 + 阈值表（预测时需要用阈值把目标日 net_load_fc 映射到 bin）
    return {
        "profile": profile,
        "net_thresholds": th
    }

