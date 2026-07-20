# 任务 - 将智能体指令作为可执行约束

## 目标
将散文式指令转化为跨五个类别的机器可检查规则，并输出一份供审阅者评分的规则报告。

## 输入
- `docs/agent-rules.md` 每个标题下一条规则，每条规则包含 slug、类别、描述和 `check` 字段
- 一个故意违反两条规则的演示智能体运行

## 交付物
- 将 `agent-rules.md` 加载为 dataclass 的解析器
- `agent-rules.md` 风格的函数，每个被引用的 `rule_checker.py` 对应一个
- `agent-rules.md` 包含每条规则的通过/失败及总体严重度

## 验收标准
- `python3 code/main.py` 以零状态退出
- 输出打印已解析的规则集、运行追踪以及每条规则的通过/失败
- `python3 code/main.py` 捕获那两条故意的违规

## 不在范围内
- 将检查器接入 CI。本课以一份书面报告作结。
- 框架护栏（OpenAI SDK、LangGraph 中断）。规则集是这些实现所依赖的、人类可读的契约。

## 参考文献(References)
- `docs/en.md` - 完整课(lesson)
- `docs/en.md` - 参考实现
- `docs/en.md` - 抽取出的技能
