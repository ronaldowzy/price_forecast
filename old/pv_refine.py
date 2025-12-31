import numpy as np
import pandas as pd

# 用哪些小时评估“PV日中午是否容易崩”
MIDDAY_HOURS = [10, 11, 12, 13, 14]

def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))

def build_pv_volatility_profile(df_hist: pd.DataFrame) -> dict:
    """
    输入：历史明细df（含 day_type, hour, da_price, rt_price，以及预测特征列）
    输出：用于细分PV_STABLE / PV_VOLATILE 的“画像模型”
    """
    # 1) 只拿历史里已经被判为 PV 的日子（兼容PV / PV_DOMINANT）
    pv_mask = df_hist["day_type"].isin(["PV", "PV_DOMINANT"])
    df_pv = df_hist[pv_mask].copy()
    if df_pv.empty:
        raise ValueError("历史中未找到 PV/PV_DOMINANT 日，无法细分 PV_STABLE / PV_VOLATILE")

    # 2) 计算每一天中午“波动/极端程度”得分：使用 |RT-DA| 的 90分位（也可换成std）
    df_pv["diff"] = df_pv["实时价格"] - df_pv["日前价格"]
    df_mid = df_pv[df_pv["hour"].isin(MIDDAY_HOURS)]

    pv_day_score = (
        df_mid.groupby("日期")["diff"]
        .apply(lambda s: float(np.nanpercentile(np.abs(s.values), 90)))
        .reset_index()
        .rename(columns={"diff": "pv_vol_score"})
    )

    # 3) 为每个PV日建立“仅用D-1可得的日特征”（来自预测列的日均）
    # 你可按需扩充，但建议先保持少而稳
    daily_feat = (
        df_pv.groupby("日期")
        .agg({
            "直调负荷(预测)": "mean",
            "风电总加(预测)": "mean",
            "光伏总加(预测)": "mean",
        })
        .reset_index()
        .rename(columns={
            "直调负荷(预测)": "load_mean",
            "风电总加(预测)": "wind_mean",
            "光伏总加(预测)": "pv_mean",
        })
    )
    daily_feat["pv_ratio"] = daily_feat["pv_mean"] / (daily_feat["load_mean"] + 1e-6)
    daily_feat["wind_ratio"] = daily_feat["wind_mean"] / (daily_feat["load_mean"] + 1e-6)

    pv_days = daily_feat.merge(pv_day_score, on="日期", how="inner")

    # 4) 设定阈值：PV日里 pv_vol_score 的 75分位作为“易不稳定”门槛（可改 0.70/0.80）
    th = float(np.nanpercentile(pv_days["pv_vol_score"].values, 75))

    return {
        "pv_days_table": pv_days,   # 每个历史PV日：日特征 + 波动得分
        "threshold": th,
        "k": 25,                    # 近邻数量（你可调 15/25/40）
    }

def refine_pv_day_type(daily_row: pd.Series, pv_profile: dict) -> str:
    """
    输入：某目标日的日特征（load_mean/pv_mean/wind_mean等），以及 pv_profile
    输出：PV_STABLE / PV_VOLATILE（若非PV日则返回原day_type）
    """
    raw = daily_row["day_type"]
    if raw not in ["PV", "PV_DOMINANT"]:
        return raw

    pv_days = pv_profile["pv_days_table"].copy()
    th = pv_profile["threshold"]
    k = pv_profile["k"]

    # 构造目标日特征（仅用D-1可得）
    load = float(daily_row["load_mean"])
    pv = float(daily_row["pv_mean"])
    wind = float(daily_row["wind_mean"])
    pv_ratio = pv / (load + 1e-6)
    wind_ratio = wind / (load + 1e-6)

    target_vec = np.array([load, pv, wind, pv_ratio, wind_ratio], dtype=float)

    # 历史PV日向量
    hist_vecs = pv_days[["load_mean", "pv_mean", "wind_mean", "pv_ratio", "wind_ratio"]].values.astype(float)
    dists = np.array([_euclid(v, target_vec) for v in hist_vecs])

    # 选K近邻，取波动得分均值作为目标日的“预期波动”
    idx = np.argsort(dists)[:min(k, len(dists))]
    est_score = float(np.nanmean(pv_days["pv_vol_score"].values[idx]))

    return "PV_VOLATILE" if est_score >= th else "PV_STABLE"
