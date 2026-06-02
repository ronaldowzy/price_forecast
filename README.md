# PriceForecast — 中国电力现货市场实时价格预测与回测工具

> 面向中国电力现货市场的开源实时价格预测与回测工具，当前以 **山东实时电价 15 分钟粒度预测** 为样例。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 项目简介

PriceForecast 是一个专注于中国电力现货市场的价格预测工具。项目从山东电力市场 15 分钟粒度的实时电价出发，提供从数据获取、特征工程、模型训练、实时预测到历史回测的完整工作流。

项目刚刚公开，正在积极开源化中。我们欢迎社区参与，共同完善中国电力市场的开源预测生态。

---

## 功能亮点

- **负价识别与预测** — 自动检测并预测电力现货市场中的负电价事件
- **尖峰捕捉** — 针对价格尖峰（spike）的专项建模与预测能力
- **抽蓄特征建模** — 内含抽水蓄能电站运行特征的提取与建模逻辑
- **净负荷分析** — 支持净负荷（net load）计算，作为核心预测特征之一
- **历史回测引擎** — 内置回测框架，支持多时间段、多模型的离线验证
- **Excel 报告输出** — 预测与回测结果自动生成结构化 Excel 报告，便于业务分析
- **15 分钟粒度** — 适配中国电力现货市场 15 分钟结算周期

---

## 适用场景

| 场景 | 说明 |
|------|------|
| 市场参与者 | 辅助发电企业、售电公司制定报价策略 |
| 学术研究 | 电力市场价格建模、可再生能源消纳研究 |
| 策略回测 | 验证基于价格预测的交易策略历史表现 |
| 市场分析 | 理解电价波动规律，识别负价、尖峰等极端事件 |
| 学习参考 | 学习电力市场数据处理与时间序列预测的实践方法 |

---

## 安装

### 环境要求

- Python 3.10+
- 推荐使用 conda 或 venv 管理虚拟环境

### 从源码安装

```bash
git clone https://github.com/ronaldowzy/price_forecast.git
cd price_forecast
pip install -r requirements.txt
pip install -e .
```

---

## 快速开始

### 1. 数据准备

将电价数据放置于 `data/` 目录下，格式参见下方 [数据格式](#数据格式) 章节。

### 2. 训练模型

```bash
price-forecast train \
  --history "data/价格预测数据集.csv" \
  --model_dir "models_v2_6" \
  --th_spike 800 \
  --th_neg 0
```

### 3. 运行预测

```bash
price-forecast predict \
  --history "data/价格预测数据集.csv" \
  --model_dir "models_v2_6" \
  --forecast "data/价格预测输入数据.csv" \
  --date "2026-01-15" \
  --out "output/RT_pred_2026-01-15.xlsx"
```

### 4. 历史回测

```bash
price-forecast backtest \
  --history "data/价格预测数据集.csv" \
  --model_dir "models_v2_6" \
  --dates "2026-01-01,2026-01-02" \
  --out "output/backtest_v0_1.xlsx"
```

也可以直接调用当前主脚本：

```bash
python3 rt_forecast_b_route_v2_6_9_annotated.py --help
```

---

## 数据格式

项目当前示例数据使用 CSV/XLSX 格式，15 分钟粒度，每日 96 个时段。核心字段包括：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `日期` | date | 交易日期 |
| `时刻` | string | 15 分钟时段标签 |
| `直调负荷(预测)` | float | 负荷预测 |
| `联络线受电负荷(预测)` | float | 联络线受电负荷预测 |
| `风电总加(预测)` | float | 风电预测 |
| `光伏总加(预测)` | float | 光伏预测 |
| `抽蓄(实际)` | float | 抽水蓄能实际出力/负荷特征 |
| `日前价格` | float | 日前价格 |
| `实时价格` | float | 实时价格，训练/回测目标 |

更多可选特征字段请参考 `docs/data_schema.md`（规划中）。

---

## 输出文件说明

| 输出路径 | 说明 |
|----------|------|
| `output/predictions/` | 每日/每时段预测结果（CSV） |
| `output/backtest/` | 回测结果汇总与指标统计 |
| `output/reports/` | Excel 格式分析报告 |
| `output/models/` | 训练保存的模型文件 |
| `output/logs/` | 运行日志 |

Excel 报告通常包含：预测值 vs 实际值对比、误差统计（MAE / RMSE / MAPE）、分时段精度分析、极端事件命中率等。

---

## 路线图

- [x] 山东实时电价 15 分钟粒度预测
- [x] 负电价事件检测与预测
- [x] 价格尖峰捕捉
- [x] 抽水蓄能特征建模
- [x] 净负荷分析
- [x] 历史回测引擎
- [x] Excel 报告自动生成
- [ ] 多省份市场支持（广东、浙江、山西等）
- [ ] 日前/日内市场双轨预测
- [ ] Docker 一键部署
- [ ] Web 可视化仪表板
- [ ] 更多模型集成（Transformer、N-BEATS 等）
- [ ] 完善 API 文档与数据字典
- [ ] CI/CD 自动化测试与发布

---

## 贡献方式

我们欢迎各种形式的贡献！

### 如何参与

1. **Fork** 本仓库
2. 创建你的特性分支：`git checkout -b feature/my-feature`
3. 提交更改：`git commit -m 'feat: add my feature'`
4. 推送到远程：`git push origin feature/my-feature`
5. 提交 **Pull Request**

### 贡献类型

- 报告 Bug 或提出功能建议（Issues）
- 补充文档或翻译
- 提交代码修复或新功能
- 分享使用案例或数据集
- 帮助代码审查

### 开发规范

- 遵循项目现有的代码风格
- 新功能请附带测试用例
- 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范

---

## 免责声明

本项目仅供学习与研究用途，**不构成任何投资建议或交易指导**。

- 电价预测存在固有不确定性，预测结果不保证准确性
- 使用本项目进行的任何交易决策由用户自行承担风险
- 本项目不保证持续维护或更新
- 用户应遵守所在地区电力市场相关法律法规
- 项目贡献者不对因使用本项目产生的任何直接或间接损失负责

使用本项目即表示您已阅读并理解上述声明。

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

## 联系方式

如有问题或建议，欢迎通过 [GitHub Issues](https://github.com/ronaldowzy/price_forecast/issues) 提出。
