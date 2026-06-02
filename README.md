# Price Forecast

山东省电力市场价格预测工具，基于分位数回归的实时电价预测模型。

## 功能

- **日前电价预测 (DA Forecast)** — 基于小时均值的日前价格预测
- **实时电价预测 (RT Forecast)** — 基于偏差分析的实时价格预测（Route-B 模型）
- **回测分析 (Backtest)** — 月度回测与指标评估
- **数据获取 (Data Collection)** — 从 SGCC 山东电力市场平台获取运行数据

## 安装

```bash
pip install -e .
```

或仅安装依赖：

```bash
pip install -r requirements.txt
```

## 使用

```bash
# 通过 CLI 入口运行
python -m price_forecast

# 或直接运行最新模型脚本
python rt_forecast_b_route_v2_6_9_annotated.py
```

### 数据获取

`getData/` 目录下的脚本需要 SGCC 平台的认证凭据，通过环境变量提供：

```bash
export SGCC_TOKEN="your-admin-token"
export SGCC_SESSIONID="your-session-id"
python getData/run_市场运行数据用电侧.py
```

## 目录结构

详见 [docs/repository_structure.md](docs/repository_structure.md)。

## 数据说明

`data/` 目录包含山东省电力市场聚合运行数据（负荷预测、风光出力、日前/实时电价等），不含个人信息。详见 [docs/data_privacy.md](docs/data_privacy.md)。

## 后续计划

- 将单体脚本 `rt_forecast_b_route_v2_6_9_annotated.py` 模块化拆分到 `src/price_forecast/` 包中
- 添加单元测试和 CI 配置
- 完善 API 文档

## 许可证

待定。
