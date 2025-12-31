# oct_2024_daily_report.py
# 目的：参考 backtest_p50_month.py + plot_backtest_v1.py
#       逐日滚动回测并输出 2024-10 整月“每日一张图”（PNG），用于汇报优化进展。
#
# 特点：
# - 训练集：日期 < target_date（无泄露）
# - 预测：输出 DA_pred、RT_P50 以及 RT 区间（P10/P90）
# - 作图：同图展示 预测 vs 真实，并给出关键误差指标（MAE/RMSE/MAPE）
# - 健壮性：自动跳过无历史/无当日数据；RT 缺失点不参与误差计算（由 run_one_day 内部处理）

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from data_loader import load_and_prepare_data
from backtest_p50_month import run_one_day


# =========================
# 中文字体（避免乱码）
# =========================
def set_chinese_font():
    # 常见 Windows/Linux 中文字体候选
    candidates = [
        "Microsoft YaHei", "Microsoft JhengHei", "SimHei",
        "Noto Sans CJK SC", "Noto Sans CJK TC", "WenQuanYi Zen Hei",
        "PingFang SC", "PingFang TC", "Heiti TC", "Arial Unicode MS"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    # 找不到也不报错：仍可出图，只是可能乱码
    plt.rcParams["axes.unicode_minus"] = False
    return None


def interval_to_time(interval: int) -> str:
    """
    interval: 1..96 代表 00:00-00:15 ... 23:45-24:00
    返回时刻标签（该结算点的“结束时刻”），例如 interval=1 -> 00:15, interval=96 -> 24:00
    """
    minutes = interval * 15
    hh = minutes // 60
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"


DAY_TYPE_MAP = {
    "PV_STABLE": "光伏平稳日",
    "PV_VOLATILE": "光伏波动日",
    "PV": "光伏主导日",
    "PV_DOMINANT": "光伏主导日",
    "WIND_DOMINANT": "风电主导日",
    "HIGH_LOAD": "高负荷日",
    "CONVENTIONAL": "常规日",
}


def plot_one_day(merged: pd.DataFrame, summary: dict, save_path: str):
    """
    merged: run_one_day 返回的逐点明细（含 interval/预测/真实）
    summary: run_one_day 返回的汇总（含 mae/rmse/mape/day_type）
    """
    merged = merged.sort_values("interval").copy()

    # x 轴：interval；辅助显示时间标签
    x = merged["interval"].astype(int).values
    xticks = np.linspace(1, 96, 9, dtype=int)
    xtick_labels = [interval_to_time(i) for i in xticks]

    da_pred = merged["da_pred"].astype(float).values
    rt_p50 = merged["rt_p50"].astype(float).values
    rt_p10 = merged.get("rt_p10", pd.Series(np.nan, index=merged.index)).astype(float).values
    rt_p90 = merged.get("rt_p90", pd.Series(np.nan, index=merged.index)).astype(float).values

    da_true = merged.get("da_true", pd.Series(np.nan, index=merged.index)).astype(float).values
    rt_true = merged.get("rt_true", pd.Series(np.nan, index=merged.index)).astype(float).values

    # 风险/机会底色（轻量版，便于汇报）
    band_width = rt_p90 - rt_p10
    abs_delta = np.abs(rt_p50 - da_pred)

    bw_th = np.nanquantile(band_width, 0.75) if np.isfinite(band_width).any() else np.nan
    delta_th = np.nanquantile(abs_delta, 0.75) if np.isfinite(abs_delta).any() else np.nan
    # 高风险：区间宽；机会：价差大
    risk_mask = (band_width >= bw_th) if np.isfinite(bw_th) else np.zeros_like(x, dtype=bool)
    opp_mask = (abs_delta >= delta_th) if np.isfinite(delta_th) else np.zeros_like(x, dtype=bool)

    fig = plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # 风险底色
    if risk_mask.any():
        for i in range(len(x)):
            if risk_mask[i]:
                ax.axvspan(x[i]-0.5, x[i]+0.5, alpha=0.08, label="_nolegend_")

    # 区间带
    ax.fill_between(x, rt_p10, rt_p90, alpha=0.20, label="RT 预测区间 (P10–P90)")

    # 预测线
    ax.plot(x, da_pred, color="tab:blue", linestyle="--", linewidth=2, label="预测日前价格 (DA)")
    ax.plot(x, rt_p50, color="tab:red", linestyle="--", linewidth=2, label="预测实时价格 (RT P50)")

    # 真实线
    ax.plot(x, da_true, color="tab:blue", linestyle="-", linewidth=2, label="真实日前价格 (DA 实际)")
    ax.plot(x, rt_true, color="tab:red", linestyle="-", linewidth=2, label="真实实时价格 (RT 实际)")

    # 机会标注（顶部点标）
    if opp_mask.any():
        ax.scatter(x[opp_mask], rt_p50[opp_mask], s=12, label="方向机会点（|RT_P50-DA|较大）")

    # 标题与注释
    raw_day_type = str(summary.get("day_type", "N/A"))
    day_type_cn = DAY_TYPE_MAP.get(raw_day_type, raw_day_type)

    mae_p50 = summary.get("mae_p50", np.nan)
    rmse_p50 = summary.get("rmse_p50", np.nan)
    mape_p50 = summary.get("mape_p50", np.nan)

    date_str = str(summary.get("date"))
    ax.set_title(f"{date_str} | 日型：{day_type_cn} | MAE={mae_p50:.2f} RMSE={rmse_p50:.2f} MAPE={mape_p50:.2%}")

    ax.set_xlabel("结算点（15分钟）")
    ax.set_ylabel("价格")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2, fontsize=9)

    # 右下角简要面板（汇报用）
    text = (
        f"机会阈值(75%分位)：|RT_P50−DA| ≥ {delta_th:.1f}\n"
        f"风险阈值(75%分位)：P90−P10 ≥ {bw_th:.1f}\n"
        f"备注：高风险底色为区间宽时段\n"
    )
    ax.text(0.98, 0.02, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", alpha=0.08))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def daterange(start_date: str, end_date: str):
    # end_date 含当天
    dates = pd.date_range(start_date, end_date, freq="D")
    for d in dates:
        yield pd.Timestamp(d.date())


def main():
    parser = argparse.ArgumentParser(description="输出 2025-10 整月每日回测图（逐日滚动训练，无泄露）")
    parser.add_argument("--data_path", default="data/价格预测数据集.csv", help="历史全量数据CSV路径（含真实DA/RT + (预测)字段）")
    parser.add_argument("--start", default="2025-10-01", help="开始日期（YYYY-MM-DD）")
    parser.add_argument("--end", default="2025-10-31", help="结束日期（YYYY-MM-DD，含当天）")
    parser.add_argument("--out_dir", default="oct_2025_daily_plots", help="输出目录（会生成 png/ 和 csv/ 子目录）")
    args = parser.parse_args()

    set_chinese_font()

    out_png = os.path.join(args.out_dir, "png")
    out_csv = os.path.join(args.out_dir, "csv")
    os.makedirs(out_png, exist_ok=True)
    os.makedirs(out_csv, exist_ok=True)

    # 读取并预处理全量数据
    df_all = load_and_prepare_data(args.data_path)

    summaries = []
    for d in daterange(args.start, args.end):
        try:
            result = run_one_day(df_all, d)
            if isinstance(result, dict):
                # SKIP/ERROR
                summaries.append(result)
                continue

            summary, merged = result
            summaries.append({k: v for k, v in summary.items() if k != "hour_mae"})

            # 保存逐点明细（方便复核）
            day_tag = d.strftime("%Y%m%d")
            merged_path = os.path.join(out_csv, f"{day_tag}_detail.csv")
            merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

            # 保存图
            fig_path = os.path.join(out_png, f"{day_tag}.png")
            plot_one_day(merged, summary, fig_path)

        except Exception as e:
            summaries.append({"date": d.date(), "status": "ERROR", "error": str(e)})

    # 汇总表
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False, encoding="utf-8-sig")

    print(f"[OK] 输出完成：{args.out_dir}")
    print(f" - 每日图片：{out_png}")
    print(f" - 每日明细：{out_csv}")
    print(f" - 汇总指标：{os.path.join(args.out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()