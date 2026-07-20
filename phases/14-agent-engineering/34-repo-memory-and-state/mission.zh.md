# 任务 - 仓库记忆与持久状态

## 目标
为 `agent_state.json` 和 `task_board.json` 编写 JSON Schema，构建一个 `StateManager` 用于加载、校验、修改并原子化写入，并证明跨两个回合的往返一致性。

## 输入
- 第 32 课中的三文件工作台结构
- 一个仅使用标准库的校验器，覆盖 required、type、enum、pattern 和 items

## 交付物
- `agent_state.schema.json` 和 `task_board.schema.json` 紧邻代码
- `agent_state.schema.json`、`task_board.schema.json`、`StateManager.load` 采用临时文件加重命名方式写入
- 一个演示运行，跨两个回合修改状态并干净地重新加载

## 验收标准
- `python3 code/main.py` 退出码为零
- 一次错误写入（缺少必填字段、错误枚举）被拒绝，而非被持久化
- 运行后 `python3 code/main.py` 根据 schema 校验通过

## 不在范围内
- SQLite 或外部存储后端。本地文件才是本课的重点。
- LangGraph checkpointers、Letta memory blocks。思路相同，存储不同；不在本课范围内。

## 参考文献(References)
- `docs/en.md` - 完整课(lesson)
- `docs/en.md` - 参考实现
- `docs/en.md` - 抽取出的技能
