# Contributing to price_forecast

感谢你对本项目的关注！以下是参与贡献的指南。

## 提交 Issue

- 使用 GitHub Issues 报告 bug 或提出功能建议。
- 请尽量提供可复现的步骤、Python 版本、操作系统等信息。
- 涉及数据文件的问题，请注明使用的数据集版本。

## 提交 Pull Request

1. Fork 本仓库并基于 `main` 分支创建你的特性分支：
   ```bash
   git checkout -b feature/your-feature main
   ```
2. 完成修改后确保代码能正常运行，相关测试通过。
3. 提交时请使用清晰的 commit message，推荐格式：
   ```
   <类型>(<范围>): <简要描述>

   类型: feat / fix / docs / style / refactor / test / chore
   ```
   示例：`feat(forecast): 新增 B 路径预测入口`
4. 推送到你的 Fork，然后在 GitHub 上发起 Pull Request。
5. 在 PR 描述中说明改动目的和影响范围。

## 运行测试

```bash
# 如果项目有 pytest 测试
python -m pytest tests/ -v

# 如果只是运行脚本验证
python rt_forecast_b_route_v2_6_9.py
```

## 代码风格

- 遵循 PEP 8 规范。
- 函数和类请添加 docstring。
- 不要提交包含硬编码路径或本地绝对路径的代码。

## 数据文件注意事项

- `data/` 目录下的数据文件仅供本地开发和测试使用。
- 请勿将敏感的市场数据或客户数据提交到仓库。
- 如需使用示例数据，请用脱敏或模拟数据。

## 许可证

提交即表示你同意你的贡献在 [MIT License](LICENSE) 下发布。
