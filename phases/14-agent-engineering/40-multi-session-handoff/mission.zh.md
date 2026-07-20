# 任务 - 多会话交接

## 目标
在会话结束时根据工作台产物生成 `handoff.md` 和 `handoff.json`，使下一个会话在第一分钟就能高效工作。两种形式都包含相同的七个字段；如有不一致以 JSON 为准。

## 输入
- `agent_state.json`、`verification_report.json`、`review_report.json`、`feedback_record.jsonl` 来自之前的课
- 七个字段：summary、changed_files、commands_run、failed_attempts、open_risks、next_action、verdict_pointer

## 交付物
- 一个打包这四个产物的 `WorkbenchSnapshot` 加载器
- `WorkbenchSnapshot`
- 一个反馈过滤器，选取最后 K 条记录以及所有非零退出码
- `WorkbenchSnapshot` 和 `generate_handoff(snapshot) -> (markdown, payload)` 写在脚本旁边

## 验收标准
- `python3 code/main.py` 以零退出
- 两个文件都包含全部七个字段以及一个非空的 `python3 code/main.py`
- 使用相同输入重新运行脚本会产生完全一致的数据包

## 不在范围内
- 压缩策略（Codex compact 端点、Claude Code 五阶段）。交接是关闭一个会话；压缩是延续一个会话。
- PR 模板化。该 markdown 可复用作 PR 正文，但本课的讲解止步于文件。

## 参考文献(References)
- `docs/en.md` - 完整课(lesson)
- `docs/en.md` - 参考实现
- `docs/en.md` - 抽取出的技能
