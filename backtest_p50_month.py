# backtest_p50_month.py
# 目的：验证 RT_P50 的预测准确性（滚动训练，无泄露）
# 输出：逐日 MAE/RMSE/MAPE + 汇总；并保存逐日 96点预测明细

import os
import pandas as pd
import numpy as np

from data_loader import load_and_prepare_data
from feature_engineering import compute_daily_features, classify_day_type
from pv_refine import build_pv_volatility_profile, refine_pv_day_type
from da_forecast import forecast_da_price
from bias_profile import build_bias_profile
from rt_forecast import forecast_rt_price

# ========= 你需要改的配置 =========
HISTORY_DATA_PATH = "data/价格预测数据集.csv"   # 全量历史（含真实 DA/RT + (预测)字段）
START_DATE = "2025-10-01"
END_DATE   = "2025-10-31"  # 含该日
OUT_DIR = "backtest_outputs"
# =================================

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[m] - y_pred[m]))) if m.any() else np.nan

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m])**2))) if m.any() else np.nan

def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    denom = np.maximum(np.abs(y_true[m]), eps)
    return float(np.mean(np.abs((y_pred[m] - y_true[m]) / denom))) if m.any() else np.nan

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run_one_day(df_all: pd.DataFrame, target_date: pd.Timestamp) -> dict:
    """
    滚动训练：
    - 训练集：所有 日期 < target_date
    - 预测输入：target_date 当天的 (预测) 字段（从全量表里切出来模拟“第二天预测文件”）
    - 评价：target_date 当天真实 RT
    """
    # 1) 切训练集
    df_hist = df_all[df_all["日期"] < target_date].copy()
    if df_hist.empty:
        return {"date": target_date.date(), "status": "SKIP_NO_HISTORY"}

    # 2) 训练集：日特征 -> 日型粗分
    daily_hist = compute_daily_features(df_hist)
    daily_hist = classify_day_type(daily_hist)

    # 3) PV 画像（只用历史 PV 日）
    df_hist2 = df_hist.merge(
        daily_hist[["日期", "day_type", "load_mean", "pv_mean", "wind_mean"]],
        on="日期", how="left"
    )
    pv_profile = build_pv_volatility_profile(df_hist2)

    # 4) PV 细分写回 daily_hist
    daily_hist["day_type"] = daily_hist.apply(lambda r: refine_pv_day_type(r, pv_profile), axis=1)

    # 5) 把细分日型 merge 回训练明细
    df_hist3 = df_hist.drop(columns=["day_type"], errors="ignore")
    df_hist3 = df_hist3.merge(daily_hist[["日期", "day_type"]], on="日期", how="left")

    # 6) 构建偏差画像（你已升级为带净负荷分层也没问题）
    bias = build_bias_profile(df_hist3)

    # 7) 构造“预测输入”（用全量表中目标日的 (预测) 字段模拟 D-1 输入）
    df_input = df_all[df_all["日期"] == target_date].copy()
    if df_input.empty:
        return {"date": target_date.date(), "status": "SKIP_NO_TARGET_ROWS"}

    # 8) 输入日型判定 + PV 细分
    daily_input = compute_daily_features(df_input)
    daily_input = classify_day_type(daily_input)
    daily_input["day_type"] = daily_input.apply(lambda r: refine_pv_day_type(r, pv_profile), axis=1)
    day_type_today = daily_input.loc[daily_input["日期"] == target_date, "day_type"].values[0]

    # 9) 预测 DA（只用历史到 target_date 之前）
    da_hourly = forecast_da_price(df_hist3, target_date)

    # 10) 预测 RT（P50/P10/P90）
    # 注意：你若把 rt_forecast.py 改成需要 df_input（净负荷分层版本），这里传 df_input
    rt_pred = forecast_rt_price(da_hourly, bias, day_type_today, df_input)

    pred_df = pd.DataFrame(rt_pred).sort_values("interval")

    # 11) 取真实 RT（用于评价）
    # 真实 RT 来自 df_input（同一张表里包含“实时价格”真实列）
    true_df = df_input[["interval", "实时价格", "日前价格"]].copy().sort_values("interval")
    merged = pred_df.merge(true_df, on="interval", how="left").rename(
        columns={"实时价格": "rt_true", "日前价格": "da_true"}
    )

    # 12) 指标（核心：P50）
    out = {
        "date": target_date.date(),
        "day_type": day_type_today,
        "mae_p50": mae(merged["rt_true"], merged["rt_p50"]),
        "rmse_p50": rmse(merged["rt_true"], merged["rt_p50"]),
        "mape_p50": mape(merged["rt_true"], merged["rt_p50"]),
        "status": "OK"
    }

    # 额外：按小时 MAE（定位哪个时段最差）
    merged["hour"] = merged["hour"].astype(int)
    hour_mae = merged.groupby("hour").apply(lambda g: mae(g["rt_true"], g["rt_p50"])).reset_index(name="mae_p50")
    out["hour_mae"] = hour_mae

    return out, merged

def main():
    ensure_dir(OUT_DIR)

    df_all = load_and_prepare_data(HISTORY_DATA_PATH)
    df_all = df_all.sort_values(["日期", "interval"]).reset_index(drop=True)

    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)

    all_days = pd.date_range(start, end, freq="D")

    daily_rows = []
    hour_mae_rows = []

    for d in all_days:
        try:
            result = run_one_day(df_all, d)
            if isinstance(result, dict):
                # SKIP
                daily_rows.append(result)
                continue

            summary, merged = result
            daily_rows.append(summary)

            # 保存逐日明细（便于复盘）
            day_str = d.strftime("%Y-%m-%d")
            merged.to_csv(os.path.join(OUT_DIR, f"pred_vs_true_{day_str}.csv"), index=False, encoding="utf-8-sig")

            # 汇总按小时 MAE
            hm = summary["hour_mae"].copy()
            hm["date"] = d.date()
            hm["day_type"] = summary["day_type"]
            hour_mae_rows.append(hm)

            print(f"[OK] {day_str} | {summary['day_type']} | MAE={summary['mae_p50']:.2f}")
        except Exception as e:
            daily_rows.append({"date": d.date(), "status": f"ERROR: {e}"})
            print(f"[ERR] {d.strftime('%Y-%m-%d')} | {e}")

    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(os.path.join(OUT_DIR, "daily_metrics.csv"), index=False, encoding="utf-8-sig")

    ok_df = daily_df[daily_df["status"] == "OK"].copy()
    if not ok_df.empty:
        # 汇总：总体 + 分日型
        overall = {
            "n_days": int(len(ok_df)),
            "mae_mean": float(ok_df["mae_p50"].mean()),
            "mae_p50": float(ok_df["mae_p50"].median()),
            "rmse_mean": float(ok_df["rmse_p50"].mean()),
            "mape_mean": float(ok_df["mape_p50"].mean()),
        }
        print("\n=== Overall ===")
        print(overall)

        by_type = ok_df.groupby("day_type")[["mae_p50", "rmse_p50", "mape_p50"]].agg(["count", "mean", "median"]).reset_index()
        by_type.to_csv(os.path.join(OUT_DIR, "metrics_by_day_type.csv"), index=False, encoding="utf-8-sig")

        print("\n=== By Day Type ===")
        print(by_type)

    if hour_mae_rows:
        hour_df = pd.concat(hour_mae_rows, ignore_index=True)
        hour_df.to_csv(os.path.join(OUT_DIR, "hour_mae.csv"), index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
