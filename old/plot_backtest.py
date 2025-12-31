import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

def build_explain_text(day_type_cn: str, raw_day_type: str) -> str:
    common = (
        "读图与用法：\n"
        "1) 红>蓝：实时偏贵 → 倾向日前多锁，减少实时敞口\n"
        "2) 红<蓝：实时偏便宜 → 可适当留量到实时（结合风险带）\n"
        "3) 阴影越宽：不确定性越高 → 留量更谨慎\n"
        "4) 底色高亮：高风险时段（区间宽/价差大/可能负价）\n"
    )

    # 日型专属提醒
    if raw_day_type == "PV_VOLATILE":
        extra = (
            "\n日型提醒（光伏主导不稳定日）：\n"
            "- 中午易出现“断崖+反弹”，P50 仅作方向参考，以区间为主\n"
            "- 策略：中午避免押单边；分段锁定/分段留量，控制偏差风险"
        )
    elif raw_day_type == "PV_STABLE":
        extra = (
            "\n日型提醒（光伏主导稳定日）：\n"
            "- 中午更可能持续低价/负价，结构较平滑\n"
            "- 策略：可适当留量到实时，但仍以风险带宽度控制仓位"
        )
    elif raw_day_type == "HIGH_LOAD":
        extra = (
            "\n日型提醒（高负荷紧平衡日）：\n"
            "- 晚高峰易偏贵且波动大\n"
            "- 策略：高峰段倾向多锁；留量要更保守"
        )
    elif raw_day_type == "WIND_DOMINANT":
        extra = (
            "\n日型提醒（风电主导波动日）：\n"
            "- 夜间波动更大，极端价差更常见\n"
            "- 策略：以风险带为准；避免在高亮区激进留量"
        )
    else:
        extra = ""

    return f"日型：{day_type_cn}（{raw_day_type}）\n" + common + extra


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
HISTORY_PATH = "data/价格预测输入数据.csv"
TARGET_DATE = "2025-12-16"

COL_DATE = "日期"
COL_TIME = "时刻"
COL_DA_ACTUAL = "日前价格"
COL_RT_ACTUAL = "实时价格"

def time_to_interval(t):
    hh, mm = map(int, t.split(":"))
    if hh == 24:
        return 96
    return hh * 4 + mm // 15 + 1

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
            end = x_values[i-1]
            ranges.append((start, end))
            in_region = False
            start = None

    if in_region:
        ranges.append((start, x_values[-1]))

    return ranges

# =========================
# 读取预测结果
# =========================
pred = pd.read_csv(FORECAST_PATH).sort_values("interval")

# =========================
# 读取真实价格并筛选目标日
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
# 风险判定规则（可调参数）
# =========================
# 1) 不确定性：区间宽度 = P90 - P10
merged["band_width"] = merged["rt_p90"] - merged["rt_p10"]

# 2) 价差强度：|RT_P50 - DA|
merged["abs_delta"] = (merged["rt_p50"] - merged["da_pred"]).abs()

# 阈值：用分位数自适应（更稳，跨季节不易崩）
BW_TH = merged["band_width"].quantile(0.75)     # 区间宽度前25%认为“风险偏高”
DELTA_TH = merged["abs_delta"].quantile(0.75)   # 价差幅度前25%认为“机会/风险突出”

# 额外事件：预测低端进入负价区（强风险信号）
merged["pred_neg_risk"] = merged["rt_p10"] < 0

# 高风险判定：满足任一条件就标红（你也可以改成 AND 更保守）
merged["high_risk"] = (
    (merged["band_width"] >= BW_TH) |
    (merged["abs_delta"] >= DELTA_TH) |
    (merged["pred_neg_risk"])
)

# =========================
# 画图：预测 vs 真实 + 高风险标注
# =========================
x = merged["interval"]

da_pred = merged["da_pred"]
rt_p50 = merged["rt_p50"]
rt_p90 = merged["rt_p90"]
rt_p10 = merged["rt_p10"]

da_actual = merged[COL_DA_ACTUAL]
rt_actual = merged[COL_RT_ACTUAL]

raw_day_type  = merged["day_type"].iloc[0] if "day_type" in merged.columns else "N/A"
day_type_cn = DAY_TYPE_MAP.get(raw_day_type, raw_day_type)

plt.figure(figsize=(14, 6))
ax = plt.gca()

# 先画“高风险底色”（放在最底层）
risk_ranges = mark_contiguous_regions(merged["high_risk"].values, x.values)
for (xs, xe) in risk_ranges:
    ax.axvspan(xs, xe, alpha=0.12, label="_nolegend_")

# 预测RT风险区间
ax.fill_between(x, rt_p10, rt_p90, alpha=0.25, label="RT 预测区间 (P10–P90)")

# 预测DA / 预测RT
ax.plot(x, da_pred, linestyle="--", color="blue", linewidth=2, label="预测日前价格 (DA)")
ax.plot(x, rt_p50, linestyle="--", color="red", linewidth=2, label="预测实时价格 (RT P50)")

# 真实DA / 真实RT
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
xticks = [1 + 4*h for h in range(24)]
xlabels = [f"{h:02d}:00" for h in range(24)]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=45)

# =========================
# 图内“交易解释”极简说明（给交易员）
# =========================
explain = build_explain_text(day_type_cn, raw_day_type)

ax.text(
    0.99, 0.02, explain,
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

# 在高风险区间上方加“高风险”标注（可选，避免太密）
ymax = np.nanmax([rt_p90.max(), da_actual.max(), rt_actual.max()])
for (xs, xe) in risk_ranges:
    mid = (xs + xe) / 2
    ax.text(mid, ymax * 0.98, "高风险", ha="center", va="top", fontsize=9)

plt.tight_layout()
plt.show()

print(f"高风险阈值：band_width >= {BW_TH:.2f} 或 abs_delta >= {DELTA_TH:.2f} 或 rt_p10 < 0")
