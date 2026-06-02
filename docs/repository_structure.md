# Repository Structure

本文档说明当前仓库的目录组织和各部分用途，以及后续重构计划。

## 当前结构

```
.
├── src/price_forecast/     # 新 Python 包（CLI 入口）
│   ├── __init__.py         # 包初始化，版本号
│   ├── __main__.py         # python -m price_forecast 支持
│   └── cli.py              # CLI 包装器，动态导入主模型脚本
│
├── rt_forecast_b_route_v2_6_9_annotated.py
│                           # 当前活跃的主模型脚本（Route-B 实时电价预测）
│                           # 2815 行，包含完整的特征工程、分位数回归、概率校准
│
├── getData/                # SGCC 山东电力市场数据获取脚本
│   ├── run_市场运行数据用电侧.py      # 日度市场运行数据下载
│   ├── run_负荷信息预测.py            # 负荷预测数据下载
│   └── run_一次市场运行数据用电侧.py  # 单日市场运行数据下载
│
├── data/                   # 数据集（山东省电力市场聚合数据）
│   ├── 价格预测数据集.csv             # 主数据集 (~11MB)
│   ├── 价格预测数据集.xlsx           # Excel 格式 (~11MB)
│   ├── 价格预测数据集_bak1.csv       # 备份 (~6MB)
│   ├── 价格预测输入数据.csv          # 输入数据
│   └── 价格预测输入数据_bak1.csv     # 输入备份
│
├── old/                    # 遗留代码和历史输出
│   ├── legacy_scripts/     # 旧版本单体脚本（v1 ~ v2.6.9，共 22 个版本）
│   ├── main.py             # 旧版模块化入口
│   ├── config.py           # 旧版配置（字段映射、分位数参数）
│   ├── data_loader.py      # 旧版数据加载
│   ├── feature_engineering.py  # 旧版特征工程
│   ├── bias_profile.py     # 偏差分析
│   ├── da_forecast.py      # 日前预测
│   ├── rt_forecast.py      # 实时预测
│   ├── pv_refine.py        # 光伏波动性细化
│   ├── backtest_p50_month.py   # 月度回测
│   ├── 30_daily_report.py  # 日报生成
│   ├── plot_*.py           # 可视化脚本
│   ├── predict_rt_from_dt.py   # 基于 DA 的 RT 预测
│   ├── backtest_outputs/   # 历史回测结果 CSV
│   ├── oct_2024_daily_plots/   # 2024年10月可视化输出
│   └── oct_2025_daily_plots/   # 2025年10月可视化输出
│
├── pyproject.toml          # 包元数据和构建配置
├── requirements.txt        # 依赖列表
└── docs/                   # 文档目录
```

## 各部分说明

### `src/price_forecast/` — 新包结构

采用 `src` layout 的 Python 包。当前 `cli.py` 通过动态导入方式调用根目录的单体脚本，作为过渡方案。

### `rt_forecast_b_route_v2_6_9_annotated.py` — 主模型

Route-B 实时电价预测模型的最新带注释版本，包含：
- 数据加载与预处理
- 特征工程（日类型分类、光伏特征、负荷特征）
- 分位数回归（P10/P50/P90）
- 负电价和尖峰电价处理
- 概率校准
- 抽水蓄能特征
- 15 分钟粒度，每日 96 个数据点

### `getData/` — 数据获取

三个脚本分别对应 SGCC 山东电力市场平台的三个数据接口。**注意：** 脚本需要通过环境变量 `SGCC_TOKEN` 和 `SGCC_SESSIONID` 提供认证凭据。

### `data/` — 数据集

山东省电力市场的聚合运行数据，包含负荷预测、风电/光伏出力、日前/实时电价、节假日标记等字段。详见 [data_privacy.md](data_privacy.md)。

### `old/` — 遗留代码

早期模块化架构的脚本和历史回测输出。这些代码已被根目录的单体脚本取代，保留供参考。

## 后续重构计划

### Phase 1: 模块化拆分（近期）
将 `rt_forecast_b_route_v2_6_9_annotated.py` 拆分到 `src/price_forecast/` 包中：
- `config.py` — 配置参数
- `data_loader.py` — 数据加载
- `feature_engineering.py` — 特征工程
- `models.py` — 分位数回归模型
- `calibration.py` — 概率校准
- `cli.py` — CLI 入口

### Phase 2: 测试与 CI（中期）
- 添加单元测试
- 配置 GitHub Actions CI
- 添加代码质量检查（ruff/mypy）

### Phase 3: 文档与发布（远期）
- API 文档生成
- PyPI 发布流程
- 用户指南
