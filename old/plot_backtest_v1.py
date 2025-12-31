# plot_backtest_v1_2.py
# 预测 vs 真实 回测图（交易员可用版本）
# - 保留两条预测曲线：DA 预测、RT P50 预测
# - 右下角“交易决策面板”：方向机会区 / 决策禁区 / 操作纪律
# - 不再使用底部横向长文字（避免截断/不可读）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# =========================
# 中文字体设置（避免乱码）
# =========================
def set_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "PingFang SC", "Noto Sans CJK SC"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.sans-serif"] = [name]
            break
    mpl.rcParams["axes.unicode_minus"] = False

set_chinese_font()

DAY_TYPE_MAP = {
    # 兼容两套命名（短码 & 长码）
    "HIGH_LOAD": "高负荷紧平衡日",
    "HIGH_LOAD_DAY": "高负荷紧平衡日",

    "PV": "光伏主导低价日",
    "PV_DOMINANT": "光伏主导低价日",

    "PV_STABLE": "光伏主导稳定日",
    "PV_VOLATILE": "光伏主导不稳定日",

    "WIND": "风电主导波动日",
    "WIND_DOMINANT": "风电主导波动日",

    "CONVENTIONAL": "常规平衡日",
    "NORMAL": "常规平衡日"
}

# =========================
# 配置：文件路径 & 目标日期
# =========================
FORECAST_PATH = "forecast_result.csv"
# 注意：这里 HIST 实际读取的是“目标日真实价格”的文件（你的原脚本叫 HISTORY_PATH，但路径写的是 input）
HISTORY_PATH = "data/价格预测输入数据.csv"
TARGET_DATE = "2025-12-16"

COL_DATE = "日期"
COL_TIME = "时刻"
COL_DA_ACTUAL = "日前价格"
COL_RT_ACTUAL = "实时价格"

def time_to_interval(t: str) -> int:
    hh, mm = map(int, t.split(":"))
    if hh == 24:
        return 96
    return hh * 4 + mm // 15 + 1

def interval_to_hhmm(interval: int) -> str:
    """
    interval: 1..96
    returns 'HH:MM'
    """
    m = (interval - 1) * 15
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def mark_contiguous_regions(mask, x_values):
    """
    mask: boolean array aligned with x_values
    returns list of (x_start, x_end) ranges where mask is True contiguously
    """
    ranges = []
    in_region = False
    start = None

    for i, (m, x) in enumerate(zip(mask, x_values)):
        if m and not in_region:
            in_region = True
            start = x
        if in_region and (not m):
            end = x_values[i - 1]
            ranges.append((start, end))
            in_region = False
            start = None

    if in_region:
        ranges.append((start, x_values[-1]))

    return ranges

def ranges_to_str(ranges):
    """
    ranges: list[(start_interval, end_interval)]
    convert to 'HH:MM–HH:MM, ...' using 15-min intervals.
    end time uses end_interval+1 as the segment end.
    """
    if not ranges:
        return "无"

    parts = []
    for (s, e) in ranges:
        s = int(s)
        e = int(e)
        start_t = interval_to_hhmm(s)
        # e interval covers [e-1]*15 to e*15; show end as (e interval end) -> (e+1 start)
        end_t = interval_to_hhmm(min(e + 1, 96))  # cap at 96 -> '23:45'; for display ok
        parts.append(f"{start_t}–{end_t}")
    return "，".join(parts)

def build_decision_panel_text(raw_day_type: str,
                              lock_ranges,
                              leave_ranges,
                              blackout_ranges,
                              delta_th: float,
                              bw_th: float) -> str:
    day_type_cn = DAY_TYPE_MAP.get(raw_day_type, raw_day_type)

    # 日型总原则
    header = f"日型：{day_type_cn}（{raw_day_type}）\n"
    if raw_day_type == "PV_VOLATILE":
        principle = "定性：中午易“断崖+反弹”，禁押单边；以区间与纪律为主。\n"
        p50_note = "提示：P50 仅作方向参考，不作单点锚定。\n"
    elif raw_day_type == "PV_STABLE":
        principle = "定性：中午更可能持续低价/负价，结构相对平滑。\n"
        p50_note = "提示：仍以风险带宽度控制留量。\n"
    else:
        principle = "定性：按日型画像给出 DA/RT 结构与风险区间。\n"
        p50_note = ""

    # 机会与禁区口径
    rules = (
        f"机会门槛：|RT_P50−DA| ≥ {delta_th:.1f}\n"
        f"禁区门槛：P90−P10 ≥ {bw_th:.1f} 或 P10<0\n"
    )

    # 三类时段清单
    lock_line = f"🔵 方向机会（多锁）：{ranges_to_str(lock_ranges)}\n"
    leave_line = f"🟢 方向机会（留量）：{ranges_to_str(leave_ranges)}\n"
    blackout_line = f"⛔ 决策禁区（不押单边）：{ranges_to_str(blackout_ranges)}\n"

    # 操作纪律
    discipline = (
        "操作纪律：\n"
        "- 禁区内：按小时/半小时分段处理 + 设置留量/增锁限额\n"
        "- 禁区外：按方向机会调整整体仓位；差异小不交易\n"
    )

    return header + principle + p50_note + rules + lock_line + leave_line + blackout_line + discipline


# =========================
# 读取预测结果
# =========================
pred = pd.read_csv(FORECAST_PATH).sort_values("interval")

# =========================
# 读取真实价格并筛选目标日（回测用）
# =========================
hist = pd.read_csv(HISTORY_PATH)
hist[COL_DATE] = pd.to_datetime(hist[COL_DATE])
target_date = pd.to_datetime(TARGET_DATE)

hist_day = hist[hist[COL_DATE] == target_date].copy()
hist_day["interval"] = hist_day[COL_TIME].apply(time_to_interval)
hist_day = hist_day.sort_values("interval")

merged = pred.merge(
    hist_day[["interval", COL_DA_ACTUAL, COL_RT_ACTUAL]],
    on="interval",
    how="left"
)

# =========================
# 风险/机会判定规则（可调参数）
# =========================
# 1) 不确定性：区间宽度 = P90 - P10
merged["band_width"] = merged["rt_p90"] - merged["rt_p10"]

# 2) 价差强度：|RT_P50 - DA|
merged["abs_delta"] = (merged["rt_p50"] - merged["da_pred"]).abs()

# 阈值：用分位数自适应（跨季节更稳）
BW_TH = merged["band_width"].quantile(0.75)     # 区间宽度前25%认为“风险偏高”
DELTA_TH = merged["abs_delta"].quantile(0.75)   # 价差幅度前25%认为“差异显著”

# 额外事件：预测低端进入负价区（强风险信号）
merged["pred_neg_risk"] = merged["rt_p10"] < 0

# 决策禁区：更“硬”的禁止押单边区（建议比 high_risk 更保守）
merged["blackout"] = (merged["band_width"] >= BW_TH) | (merged["pred_neg_risk"])

# 方向机会：必须“差异显著 + 不在禁区”
merged["lock_oppty"] = (~merged["blackout"]) & (merged["abs_delta"] >= DELTA_TH) & (merged["rt_p50"] > merged["da_pred"])
merged["leave_oppty"] = (~merged["blackout"]) & (merged["abs_delta"] >= DELTA_TH) & (merged["rt_p50"] < merged["da_pred"])

# 高风险底色：用于视觉提示（可比 blackout 略宽）
merged["high_risk"] = merged["blackout"] | (merged["abs_delta"] >= DELTA_TH)

# 连续区间转 ranges
x = merged["interval"].values
blackout_ranges = mark_contiguous_regions(merged["blackout"].values, x)
lock_ranges = mark_contiguous_regions(merged["lock_oppty"].values, x)
leave_ranges = mark_contiguous_regions(merged["leave_oppty"].values, x)
risk_ranges = mark_contiguous_regions(merged["high_risk"].values, x)

# =========================
# 画图：预测 vs 真实 + 高风险标注
# =========================
da_pred = merged["da_pred"]
rt_p50 = merged["rt_p50"]
rt_p90 = merged["rt_p90"]
rt_p10 = merged["rt_p10"]

da_actual = merged[COL_DA_ACTUAL]
rt_actual = merged[COL_RT_ACTUAL]

raw_day_type = merged["day_type"].iloc[0] if "day_type" in merged.columns else "N/A"
day_type_cn = DAY_TYPE_MAP.get(raw_day_type, raw_day_type)

plt.figure(figsize=(14, 6))
ax = plt.gca()

# 高风险底色（放在最底层）
for (xs, xe) in risk_ranges:
    ax.axvspan(xs, xe, alpha=0.12, label="_nolegend_")

# 预测RT风险区间
ax.fill_between(x, rt_p10, rt_p90, alpha=0.25, label="RT 预测区间 (P10–P90)")

# 预测DA / 预测RT
ax.plot(x, da_pred, linestyle="--", color="blue", linewidth=2, label="预测日前价格 (DA)")
ax.plot(x, rt_p50, linestyle="--", color="red", linewidth=2, label="预测实时价格 (RT P50)")

# 真实DA / 真实RT（回测用；真实场景可不画）
ax.plot(x, da_actual, linewidth=2, color="blue", label="真实日前价格 (DA 实际)")
ax.plot(x, rt_actual, linewidth=2, color="red", label="真实实时价格 (RT 实际)")

# 标题
ax.set_title(
    f"{TARGET_DATE} 预测 vs 真实（回测对比） | 日型：{day_type_cn}（{raw_day_type}）",
    fontsize=14
)
ax.set_xlabel("时段（15分钟 1–96）")
ax.set_ylabel("电价（元/MWh）")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")

# x轴改成小时刻度（更交易友好）
xticks = [1 + 4 * h for h in range(24)]
xlabels = [f"{h:02d}:00" for h in range(24)]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=45)

# =========================
# 右下角：交易决策面板（替代底部横向长文本）
# =========================
decision_text = build_decision_panel_text(
    raw_day_type=raw_day_type,
    lock_ranges=lock_ranges,
    leave_ranges=leave_ranges,
    blackout_ranges=blackout_ranges,
    delta_th=DELTA_TH,
    bw_th=BW_TH
)

ax.text(
    0.98, 0.02, decision_text,
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.90)
)

# 在高风险区间上方加“高风险”标注（可选）
ymax = np.nanmax([rt_p90.max(), da_actual.max(), rt_actual.max()])
for (xs, xe) in risk_ranges:
    mid = (xs + xe) / 2
    ax.text(mid, ymax * 0.98, "高风险", ha="center", va="top", fontsize=9)

plt.tight_layout()
plt.show()

print(f"阈值：|RT_P50−DA| >= {DELTA_TH:.2f}；band_width >= {BW_TH:.2f}；pred_neg_risk={merged['pred_neg_risk'].any()}")
print(f"方向机会（多锁）: {ranges_to_str(lock_ranges)}")
print(f"方向机会（留量）: {ranges_to_str(leave_ranges)}")
print(f"决策禁区（不押单边）: {ranges_to_str(blackout_ranges)}")
