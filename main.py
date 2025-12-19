# 主入口
import pandas as pd

from data_loader import load_and_prepare_data
from feature_engineering import compute_daily_features, classify_day_type
from bias_profile import build_bias_profile
from da_forecast import forecast_da_price
from rt_forecast import forecast_rt_price

from pv_refine import build_pv_volatility_profile, refine_pv_day_type  # 新增

HISTORY_DATA_PATH = "data/价格预测数据集.csv"
INPUT_DATA_PATH = "data/价格预测输入数据.csv"

def main():
    # 1) 历史数据：读取
    df_hist = load_and_prepare_data(HISTORY_DATA_PATH)

    # 2) 历史数据：粗分日型（PV_DOMINANT/WIND_DOMINANT/HIGH_LOAD/CONVENTIONAL）
    daily_hist = compute_daily_features(df_hist)
    daily_hist = classify_day_type(daily_hist)

    # 3) 在“粗分PV日”的基础上，建立PV波动画像（只用历史PV日）
    #    注意：pv_refine 需要 daily_hist 有 load_mean/pv_mean/wind_mean（按我上面第1点改）
    #    这里 build_pv_volatility_profile 用的是明细 df_hist + day_type，因此先把 day_type merge 回 df_hist
    df_hist = df_hist.merge(daily_hist[[ "日期", "day_type", "load_mean", "pv_mean", "wind_mean"]], on="日期", how="left")

    pv_profile = build_pv_volatility_profile(df_hist)

    # 4) 用 pv_profile 把“历史PV日”进一步细分成 PV_STABLE / PV_VOLATILE（写回 daily_hist）
    daily_hist["day_type"] = daily_hist.apply(lambda r: refine_pv_day_type(r, pv_profile), axis=1)

    # 5) 重新把细分后的 day_type merge 回历史明细（覆盖旧 day_type）
    df_hist = df_hist.drop(columns=["day_type"], errors="ignore")
    df_hist = df_hist.merge(daily_hist[["日期", "day_type"]], on="日期", how="left")

    # 6) 基于“细分日型”的历史明细，构建 RT-DA 偏差画像
    bias = build_bias_profile(df_hist)

    # 7) 输入数据：读取（仅目标日）
    df_input = load_and_prepare_data(INPUT_DATA_PATH)
    target_date = df_input["日期"].iloc[0]

    # 8) 输入数据：粗分日型
    daily_input = compute_daily_features(df_input)
    daily_input = classify_day_type(daily_input)

    # 9) 输入数据：若是 PV_DOMINANT，则进一步细分 PV_STABLE / PV_VOLATILE
    daily_input["day_type"] = daily_input.apply(lambda r: refine_pv_day_type(r, pv_profile), axis=1)
    day_type_today = daily_input.loc[daily_input["日期"] == target_date, "day_type"].values[0]

    # 10) 预测 DA（用历史到 target_date 之前的小时均值）
    da_hourly = forecast_da_price(df_hist, target_date)

    # 11) 预测 RT（用细分日型画像）
    rt_result = forecast_rt_price(da_hourly, bias, day_type_today, df_input)

    # 12) 输出
    result_df = pd.DataFrame(rt_result)
    result_df["日期"] = target_date
    result_df["day_type"] = day_type_today
    result_df.to_csv("forecast_result.csv", index=False, encoding="utf-8-sig")

    print(f"预测完成：{target_date.date()} | 日型：{day_type_today}")
    print("结果已输出：forecast_result.csv")

if __name__ == "__main__":
    main()
