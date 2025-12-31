# -*- coding: utf-8 -*-
"""
predict_rt_dplus1_from_daily_excels.py

目的
----
在“市场规则变化导致历史价格水平不可直接外推”的情况下，利用你每天上传的：
- D+1 实际日前出清价（DA）
- D+1 预测负荷（直调负荷）
结合历史数据集中 RT-DA 的统计规律，输出 D+1 实时价格（RT）的预测（P50，附P10/P90区间），并画图。

核心公式
--------
RT_pred(D+1, h) = DA_actual(D+1, h) + Bias_hat(D+1, h)

Bias_hat 的学习来自历史：Bias = RT_true - DA_true。

方法（默认，可解释）
-------------------
使用“条件分箱 + 分位数统计”的画像：
Bias_hat(h | day_class, load_bin)

- day_class：workday / weekend / holiday（历史中如果有“是否节假日/是否周末休息日”则以历史为准；
  预测日默认仅基于周几判断 workday/weekend，可通过 --day_class_override 强制指定）
- load_bin：在历史中按 (day_class, hour) 对“预测直调负荷（日均/小时均值）”做分位数分箱（默认三档 LOW/MID/HIGH）
- Bias_hat 输出：P10/P50/P90

输入文件
--------
1) 历史 CSV（你工程里已有）：包含列
   日期, 时刻, 日前价格, 实时价格, 直调负荷(预测), 是否节假日, 是否周末休息日
   (其它列可有可无，脚本不会强依赖)

2) 负荷预测 .xls（你每天上传）：sheet “直调负荷”，宽表形式：
   第0行是时刻（00:15, 00:30, ... , 24:00），第1行是“预测”数值

3) 日前价格 .xls（你每天上传）：sheet “一次市场运行数据用电侧”，至少包含列：
   时刻（1:00..24:00 或 01:00..24:00）, 节点电价（元MWh）

输出
----
- CSV：rt_pred_<目标日>.csv（小时粒度 1..24，含 DA 实际 + RT 预测 + 区间 + 负荷分箱）
- PNG：rt_pred_<目标日>.png（曲线图）

运行示例
--------
python predict_rt_dplus1_from_daily_excels.py \
  --hist_csv "data/价格预测数据集.csv" \
  --load_xls "data/2025-12-16负荷信息预测.xls" \
  --da_xls   "data/2025-12-16一次市场运行数据用电侧-日前-1.xls" \
  --out_dir  "output"

注意
----
- 本脚本会从文件名中解析日期 YYYY-MM-DD，并默认目标日 = 解析日期 + 1 天。
  如文件名不含日期，可用 --target_date 显式传入 YYYY-MM-DD。
- .xls 读取依赖 xlrd>=2.0.1。
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------
# 时间标签：电力行业“结束时刻”映射
# -----------------------
def _parse_time_to_minutes(label: str) -> Optional[int]:
    label = str(label).strip().replace("：", ":")
    m = re.match(r"^(\d{1,2}):(\d{2})$", label)
    if not m:
        return None
    h = int(m.group(1))
    mi = int(m.group(2))
    return h * 60 + mi


def endlabel_hour(label: str) -> Optional[int]:
    """
    将类似 00:15/01:00/24:00 的“结束时刻标签”映射到 hour=1..24（0:00 -> 0）。
    规则：hour = ceil(minutes/60)，24:00 -> 24。
    """
    mins = _parse_time_to_minutes(label)
    if mins is None:
        return None
    if mins == 0:
        return 0
    if mins == 1440:
        return 24
    if mins < 0 or mins > 1440:
        return None
    return int(math.ceil(mins / 60.0))


def date_from_filename(path: str) -> Optional[dt.date]:
    base = os.path.basename(path)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", base)
    if not m:
        return None
    return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def to_bool_zh(x) -> bool:
    return str(x).strip() in ["是", "Y", "y", "1", "True", "TRUE", "true"]


def infer_day_class_from_date(d: dt.date) -> str:
    # 仅基于周几：周六/周日为 weekend；其余为 workday
    return "weekend" if d.weekday() >= 5 else "workday"


@dataclass
class BiasProfile:
    # MultiIndex: (day_class, hour, load_bin) -> columns p10/p50/p90
    profile: pd.DataFrame
    # MultiIndex: (day_class, hour) -> columns q_low/q_high
    load_q: pd.DataFrame


def build_bias_profile_from_history(hist_csv: str,
                                   q_low: float = 0.33,
                                   q_high: float = 0.67) -> BiasProfile:
    df = pd.read_csv(hist_csv, encoding="utf-8-sig")
    required = ["日期", "时刻", "日前价格", "实时价格", "直调负荷(预测)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"历史CSV缺少必要列：{missing}")

    # day_class
    if "是否节假日" in df.columns and "是否周末休息日" in df.columns:
        is_holiday = df["是否节假日"].apply(to_bool_zh)
        is_weekend = df["是否周末休息日"].apply(to_bool_zh)
        day_class = np.where(is_holiday, "holiday", np.where(is_weekend, "weekend", "workday"))
        df["day_class"] = day_class
    else:
        # 无标识则用日期推断（仅周末/工作日）
        df["date"] = pd.to_datetime(df["日期"]).dt.date
        df["day_class"] = df["date"].apply(infer_day_class_from_date)

    df["date"] = pd.to_datetime(df["日期"])
    df["hour"] = df["时刻"].apply(endlabel_hour)
    df["bias"] = df["实时价格"] - df["日前价格"]
    df["load_fc"] = pd.to_numeric(df["直调负荷(预测)"], errors="coerce")

    df = df.dropna(subset=["bias", "load_fc", "hour"])
    df = df[(df["hour"] >= 1) & (df["hour"] <= 24)]

    # 汇总到小时：每个 (date, day_class, hour) 一条
    hourly = (
        df.groupby(["date", "day_class", "hour"], as_index=False)
          .agg(bias_mean=("bias", "mean"),
               load_fc_mean=("load_fc", "mean"))
    )

    # 分位数阈值（每个 day_class+hour）
    load_q = (
        hourly.groupby(["day_class", "hour"])["load_fc_mean"]
              .quantile([q_low, q_high])
              .unstack()
              .rename(columns={q_low: "q_low", q_high: "q_high"})
    )

    # 分箱
    hourly = hourly.join(load_q, on=["day_class", "hour"])

    def _bin_row(r) -> str:
        if pd.isna(r["q_low"]) or pd.isna(r["q_high"]):
            return "MID"
        if r["load_fc_mean"] <= r["q_low"]:
            return "LOW"
        if r["load_fc_mean"] >= r["q_high"]:
            return "HIGH"
        return "MID"

    hourly["load_bin"] = hourly.apply(_bin_row, axis=1)

    # bias 分位数画像
    prof = (
        hourly.groupby(["day_class", "hour", "load_bin"])["bias_mean"]
              .quantile([0.1, 0.5, 0.9])
              .unstack()
              .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
              .sort_index()
    )

    return BiasProfile(profile=prof, load_q=load_q)


def read_load_forecast_xls(load_xls: str, sheet_name: str = "直调负荷") -> pd.DataFrame:
    """
    读取“负荷信息预测.xls”中指定 sheet 的宽表，返回 long 格式：
    columns: time_label, hour, load_fc
    """
    df = pd.read_excel(load_xls, sheet_name=sheet_name, engine="xlrd")
    # 约定：第0行是时刻，第1行是预测
    times = [str(x).strip() for x in df.iloc[0, 1:97].tolist()]
    vals = pd.to_numeric(df.iloc[1, 1:97], errors="coerce").tolist()
    out = pd.DataFrame({"time_label": times, "load_fc": vals})
    out["hour"] = out["time_label"].apply(endlabel_hour)
    out = out.dropna(subset=["hour", "load_fc"])
    out = out[(out["hour"] >= 1) & (out["hour"] <= 24)]
    return out


def read_da_actual_xls(da_xls: str,
                       sheet_name: Optional[str] = None,
                       time_col: str = "时刻",
                       price_col: str = "节点电价（元MWh）") -> pd.DataFrame:
    """
    读取“日前价格.xls”，返回小时 DA：
    columns: hour, da_actual
    """
    if sheet_name is None:
        sheet_name = 0
    df = pd.read_excel(da_xls, sheet_name=sheet_name, engine="xlrd")
    if time_col not in df.columns:
        raise ValueError(f"日前价格文件缺少列：{time_col}")
    if price_col not in df.columns:
        raise ValueError(f"日前价格文件缺少列：{price_col}")

    df = df[[time_col, price_col]].copy()
    df["hour"] = df[time_col].apply(lambda x: endlabel_hour(str(x)))
    df["da_actual"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["hour", "da_actual"])
    df = df[(df["hour"] >= 1) & (df["hour"] <= 24)]
    df = df.sort_values("hour")
    # 若出现重复 hour，取均值
    df = df.groupby("hour", as_index=False)["da_actual"].mean()
    return df


def predict_bias_for_day(load_hour: pd.DataFrame,
                         day_class: str,
                         bp: BiasProfile) -> pd.DataFrame:
    """
    给定目标日小时负荷预测（hour, load_fc），输出 bias 的 p10/p50/p90 与 load_bin。
    """
    prof = bp.profile
    load_q = bp.load_q

    rows = []
    for _, r in load_hour.iterrows():
        h = int(r["hour"])
        lf = float(r["load_fc"])

        # 阈值分箱
        if (day_class, h) in load_q.index:
            qrow = load_q.loc[(day_class, h)]
            ql, qh = float(qrow["q_low"]), float(qrow["q_high"])
            if lf <= ql:
                lb = "LOW"
            elif lf >= qh:
                lb = "HIGH"
            else:
                lb = "MID"
        else:
            lb = "MID"

        idx = (day_class, h, lb)

        if idx in prof.index:
            b = prof.loc[idx]
        else:
            # 逐级回退：day_class+hour  -> hour -> 全局
            try:
                sub = prof.xs((day_class, h), level=("day_class", "hour"))
                b = sub.median()
            except Exception:
                try:
                    sub = prof.xs(h, level="hour")
                    b = sub.median()
                except Exception:
                    b = pd.Series({"p10": 0.0, "p50": 0.0, "p90": 0.0})

        rows.append({
            "hour": h,
            "load_fc": lf,
            "load_bin": lb,
            "bias_p10": float(b["p10"]),
            "bias_p50": float(b["p50"]),
            "bias_p90": float(b["p90"]),
        })

    return pd.DataFrame(rows).sort_values("hour")


def plot_prediction(pred: pd.DataFrame, target_date: dt.date, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 中文字体（容器内有 Noto Sans CJK）
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK TC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    x = pred["hour"].astype(float).to_numpy()
    y_da = pred["da_actual"].astype(float).to_numpy()
    y50 = pred["rt_pred_p50"].astype(float).to_numpy()
    y10 = pred["rt_pred_p10"].astype(float).to_numpy()
    y90 = pred["rt_pred_p90"].astype(float).to_numpy()

    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, y_da, label="DA 实际", linestyle="-")
    plt.plot(x, y50, label="RT 预测(P50)", linestyle="--")
    plt.fill_between(x, y10, y90, alpha=0.2, label="RT 预测区间(P10-P90)")

    plt.xticks(range(1, 25))
    plt.xlabel("时段（1-24，结束时刻标签）")
    plt.ylabel("价格（元/MWh）")
    plt.title(f"RT 预测（DA 实际 + 偏差画像）- 目标日 {target_date}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist_csv", required=True, help="历史数据集CSV（含日前价格/实时价格/直调负荷(预测)等）")
    ap.add_argument("--load_xls", required=True, help="每日负荷预测xls（例：YYYY-MM-DD负荷信息预测.xls）")
    ap.add_argument("--da_xls", required=True, help="每日日前价格xls（例：YYYY-MM-DD一次市场运行数据用电侧-日前-1.xls）")
    ap.add_argument("--out_dir", default="output", help="输出目录")
    ap.add_argument("--target_date", default="", help="显式指定目标日YYYY-MM-DD；不填则从文件名日期+1推断")
    ap.add_argument("--day_class_override", default="", choices=["", "workday", "weekend", "holiday"],
                    help="强制指定目标日 day_class（可选）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 推断目标日
    if args.target_date:
        target = dt.datetime.strptime(args.target_date, "%Y-%m-%d").date()
    else:
        base = date_from_filename(args.load_xls) or date_from_filename(args.da_xls)
        if base is None:
            raise ValueError("无法从文件名解析日期，请使用 --target_date 显式指定目标日。")
        target = base + dt.timedelta(days=1)

    day_class = args.day_class_override or infer_day_class_from_date(target)

    # 1) 训练 bias 画像（历史）
    bp = build_bias_profile_from_history(args.hist_csv)

    # 2) 读取 D+1 的负荷预测，并聚合到小时
    load_15 = read_load_forecast_xls(args.load_xls, sheet_name="直调负荷")
    load_hour = load_15.groupby("hour", as_index=False)["load_fc"].mean()

    # 3) 读取 D+1 的 DA 实际
    da_hour = read_da_actual_xls(args.da_xls, sheet_name=None, time_col="时刻", price_col="节点电价（元MWh）")

    # 4) 预测 bias，并合成 RT
    bias_pred = predict_bias_for_day(load_hour, day_class, bp)

    pred = da_hour.merge(bias_pred, on="hour", how="left")
    pred["rt_pred_p50"] = pred["da_actual"] + pred["bias_p50"]
    pred["rt_pred_p10"] = pred["da_actual"] + pred["bias_p10"]
    pred["rt_pred_p90"] = pred["da_actual"] + pred["bias_p90"]
    pred.insert(0, "target_date", str(target))
    pred.insert(1, "day_class", day_class)

    out_csv = os.path.join(args.out_dir, f"rt_pred_{target}.csv")
    out_png = os.path.join(args.out_dir, f"rt_pred_{target}.png")
    pred.to_csv(out_csv, index=False, encoding="utf-8-sig")
    plot_prediction(pred, target, out_png)

    print(f"[OK] 目标日: {target} | day_class={day_class}")
    print(f"[OK] 输出CSV: {out_csv}")
    print(f"[OK] 输出PNG: {out_png}")


if __name__ == "__main__":
    main()
