import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# =========================
# 读取预测结果
# =========================
def set_chinese_font():
    candidates = [
        "Microsoft YaHei",   # 微软雅黑（Windows 常见）
        "SimHei",            # 黑体
        "SimSun",            # 宋体
        "PingFang SC",       # mac 常见
        "Noto Sans CJK SC",  # 思源黑体（部分环境有）
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.sans-serif"] = [name]
            break
    mpl.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块

set_chinese_font()

df = pd.read_csv("forecast_result.csv")

# 按 interval 排序，确保画图顺序正确
df = df.sort_values("interval")

# x 轴：15 分钟序号
x = df["interval"]

# 曲线
da = df["da_pred"]
rt_p50 = df["rt_p50"]
rt_p90 = df["rt_p90"]
rt_p10 = df["rt_p10"]

day_type = df["day_type"].iloc[0]

# =========================
# 开始画图
# =========================
plt.figure(figsize=(14, 6))

# RT 风险区间
plt.fill_between(
    x,
    rt_p10,
    rt_p90,
    color="orange",
    alpha=0.25,
    label="RT 预测区间 (P10–P90)"
)

# DA 曲线
plt.plot(
    x,
    da,
    linestyle="--",
    linewidth=2,
    color="blue",
    label="日前价格预测 (DA)"
)

# RT 中位曲线
plt.plot(
    x,
    rt_p50,
    linewidth=2,
    color="red",
    label="实时价格预测 (RT P50)"
)

# =========================
# 图形修饰
# =========================
plt.title(
    f"2025-01-15 价格预测（交易视角） | 日型：{day_type}",
    fontsize=14
)

plt.xlabel("15分钟时段（1–96）")
plt.ylabel("电价（元/MWh）")

plt.legend()

note = "读图：红>蓝 实时偏高；红<蓝 实时偏低\n阴影越宽 不确定性越高(风险更大)"
plt.text(
    0.99, 0.02, note,
    transform=plt.gca().transAxes,
    ha="right", va="bottom",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

# 每小时一个刻度（interval 1,5,9,...）
xticks = [1 + 4*h for h in range(24)]
xlabels = [f"{h:02d}:00" for h in range(24)]
plt.xticks(xticks, xlabels, rotation=45)


plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
