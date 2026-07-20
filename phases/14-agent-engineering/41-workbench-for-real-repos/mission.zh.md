# 任务 - 真实仓库上的工作台

## 目标
通过仅提示(prompt-only)流水线和受工作台引导(workbench-guided)的流水线，针对同一个示例应用运行相同的 `/signup` 验证任务，然后输出一份怀疑论者也能读懂的before/after对比报告。

## 输入
- `sample_app/` 配合 `app.py`（无验证）、`test_app.py`（一条happy-path测试）、`README.md`、`scripts/release.sh` 作为禁区诱饵
- 两条流水线均完全脚本化，无真实LLM调用

## 交付物
- `code/main.py` 针对同一fixture编排两条流水线
- `code/main.py` 配合五结果表格
- `code/main.py` 用于下游图表绘制

## 验收标准
- `python3 code/main.py` 零退出
- 报告度量全部五个结果：测试实际运行、验收达成、范围外文件、交接质量、审查者总计
- 工作台流水线在五个结果中至少四项上优于仅提示流水线

## 不在范围内
- 接入真实LLM。流水线已脚本化以保证可复现性。
- 调优模型。对比通过构造将模型保持为常数(Constant)。

## 参考文献(References)
- `docs/en.md` - 完整课(lesson)
- `docs/en.md` - 参考实现
- `docs/en.md` - 抽取出的技能
