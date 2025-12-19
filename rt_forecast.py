# 实时价格预测

# rt_forecast.py
# 实时价格预测（升级：按净负荷分层画像取 P50/P10/P90）

from config import *

def _get_net_bin(day_type, hour, net_load_fc, net_thresholds_df):
    row = net_thresholds_df[
        (net_thresholds_df["day_type"] == day_type) &
        (net_thresholds_df["hour"] == hour)
    ]
    if row.empty:
        return "MID"
    r = row.iloc[0]
    q33, q67 = r["q33"], r["q67"]
    if net_load_fc < q33:
        return "LOW"
    elif net_load_fc >= q67:
        return "HIGH"
    else:
        return "MID"

def forecast_rt_price(da_hourly, bias_bundle, day_type, df_input):
    """
    bias_bundle: build_bias_profile 返回的 dict，包含 profile + net_thresholds
    df_input: 目标日 96点明细（含预测字段）
    """
    profile = bias_bundle["profile"]
    net_thresholds = bias_bundle["net_thresholds"]

    # 预先为 df_input 计算 net_load_fc（与训练一致）
    df_input = df_input.copy()
    df_input["net_load_fc"] = (
        df_input[COL_LOAD_FC]
        - df_input[COL_WIND_FC]
        - df_input[COL_PV_FC]
        - df_input.get("非市场化核电总加(预测)", 0.0)
        - df_input.get("地方电厂发电总加(预测)", 0.0)
        - df_input.get("自备机组总加(预测)", 0.0)
        - df_input.get(COL_TIE_FC, 0.0)
    )

    # interval -> net_load_fc 映射
    net_map = dict(zip(df_input["interval"].values, df_input["net_load_fc"].values))

    results = []
    for interval in range(1, 97):
        hour = min((interval - 1) // 4, 23)
        da = da_hourly.get(hour)
        if da is None:
            continue

        net_load_fc = float(net_map.get(interval, 0.0))
        net_bin = _get_net_bin(day_type, hour, net_load_fc, net_thresholds)

        row = profile[
            (profile.day_type == day_type) &
            (profile.hour == hour) &
            (profile.net_bin == net_bin)
        ]

        # 兜底 1：同日型同小时，但 MID 桶
        if row.empty:
            row = profile[
                (profile.day_type == day_type) &
                (profile.hour == hour) &
                (profile.net_bin == "MID")
            ]

        # 兜底 2：细分 PV 找不到画像时，回退粗 PV
        if row.empty:
            fallback = day_type
            if day_type in ["PV_STABLE", "PV_VOLATILE"]:
                fallback = "PV_DOMINANT"
            row = profile[
                (profile.day_type == fallback) &
                (profile.hour == hour) &
                (profile.net_bin == net_bin)
            ]
            if row.empty:
                row = profile[
                    (profile.day_type == fallback) &
                    (profile.hour == hour) &
                    (profile.net_bin == "MID")
                ]

        if row.empty:
            continue

        r = row.iloc[0]
        results.append({
            "interval": interval,
            "hour": hour,
            "da_pred": da,
            "rt_p50": da + r.p50,
            "rt_p90": da + r.p90,
            "rt_p10": da + r.p10,
            "delta": r.p50,
            "stability": r.stability,
            "day_type": day_type,
            "net_bin": net_bin,
            "net_load_fc": net_load_fc
        })

    return results
