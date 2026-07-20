# Mission - 最小智能体工作台

## 目标
在一个全新的 `workdir/` 中搭建三文件最小工作台（路由器、状态、任务板），并证明单个智能体轮次能够读取状态、领取任务、写入作用域并持久化更新后的状态。

## 输入
- 紧邻课(lesson)代码的一个空 `workdir/` 目录
- 对三个文件的了解：`workdir/`、`AGENTS.md`、`agent_state.json`

## 交付物
- `code/main.py`：创建三个文件并运行一个轮次
- `code/main.py`：简短的路由器，指向状态、任务板和验证命令
- `code/main.py`：包含活动任务 id、已触碰文件、下一步动作
- `code/main.py`：包含小型积压任务及状态

## 验收标准
- `python3 code/main.py` 在第一次和第二次运行时均以零状态退出
- 第二次运行从第一次结束的地方继续，而非从头开始
- 脚本打印的 diff 显示该轮次所触碰的那一个文件

## 不在范围内
- 作用域契约、验证门、审查智能体。这些将在后续课(lesson)中叠加。
- 冗长的单体 `AGENTS.md`。路由器刻意保持简短。

## 参考文献(References)
- `docs/en.md` - 完整课(lesson)
- `docs/en.md` - 参考实现
- `docs/en.md` - 抽取出的技能
