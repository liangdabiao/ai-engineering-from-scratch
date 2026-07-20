# 任务 - 验证关卡

## 目标
将 `verify(task_id, artifacts)` 实现为基于范围报告、规则报告、反馈日志和差异(diff)的纯确定性函数，在每个任务收尾时输出一个 `verification_report.json`。

## 输入
- `scope_report.json`、`rule_report.json`、`feedback_record.jsonl` 和差异(diff)的桩加载器
- 检查表：验收已运行、验收零退出、范围干净、无 `scope_report.json` 退出、所有块级严重度规则通过

## 交付物
- 一个纯 `verify(task_id, artifacts) -> VerdictReport`
- 一个打印每项错误结果及最终通过/失败的打印机
- 写入磁盘的三个演示场景：干净通过、范围蔓延、缺失验收

## 验收标准
- `python3 code/main.py` 零退出
- 干净通过场景报告 `python3 code/main.py`；另外两个报告 `passed: true`
- 每个场景在 `passed: true` 下写入一个独立的 `python3 code/main.py`

## 不在范围内
- LLM 作为裁判的逻辑。关卡保持确定性；定性判断属于第 39 课中的审查者。
- 签名的覆盖审计日志。练习以该方式扩展关卡。

## 参考文献(References)
- `docs/en.md` - 完整课(lesson)
- `docs/en.md` - 参考实现
- `docs/en.md` - 抽取出的技能
