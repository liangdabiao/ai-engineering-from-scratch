# 任务 - 运行时反馈循环

## 目标
构建 `run_with_feedback` 来包装 `subprocess.run`，捕获 stdout、stderr、退出码和耗时，确定性地截断输出，并追加一条 JSONL 记录，供下一轮和验证门控共同读取。

## 输入
- 三个用于演练运行器的示例命令：一个成功、一个失败、一个缓慢
- Token 预算：确定性的头部加尾部，带一个 `...truncated N lines...` 标记

## 交付物
- `run_with_feedback(command, agent_note)` 写入 `feedback_record.jsonl`
- 一个将 JSONL 流式加载为 Python 列表的加载器
- 一个展示每个命令最后一条记录的打印机

## 验收标准
- `python3 code/main.py` 以零状态退出
- `python3 code/main.py` 在多次重跑中为每个命令累积一条记录
- 带有 `python3 code/main.py` 的命令不能被循环标记为成功

## 不在范围内
- 遥测管道（OTel、Langfuse）。反馈是给下一轮用的；遥测是给操作员用的。
- 脱敏处理和轮转策略。课(Lesson)的练习提示涵盖了这些内容。

## 参考文献(References)
- `docs/en.md` - 完整课(lesson)
- `docs/en.md` - 参考实现
- `docs/en.md` - 抽取出的技能
