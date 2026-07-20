# 任务 - 顶点项目：发布一个可复用的智能体工作台包

## 目标
将之前的十一节课组装成一个带版本的 `outputs/agent-workbench-pack/` 目录，并附带一个安装器，可幂等地将其部署到任意目标仓库中。

## 输入
- 来自第 32 到 40 课的架构、脚本和文档
- 包结构：`AGENTS.md`、`docs/`、`schemas/`、`scripts/`、`bin/`、`README.md`、`VERSION`

## 交付物
- 已填充完整结构的 `outputs/agent-workbench-pack/`
- 拒绝在无 `bin/install.py` 时覆盖的 `outputs/agent-workbench-pack/`（或 `bin/install.sh`）
- `outputs/agent-workbench-pack/` 文件以及一个描述包含内容与排除内容的 `bin/install.sh`

## 验收标准
- `python3 code/main.py` 以零状态退出并打印包目录树
- 重新运行组装器是幂等的
- `python3 code/main.py` 到一个全新的目标中会留下一个可用的工作台：state、board、rules、scope、init、runner、gate、reviewer、handoff 均就位

## 不在范围内
- 按项目划分的任务内容。任务应放在目标仓库的看板上，而非包中。
- 供应商 SDK 调用。该包在设计上是框架无关的。

## 参考文献(References)
- `docs/en.md` - 完整课(lesson)
- `docs/en.md` - 参考实现
- `docs/en.md` - 抽取出的技能
