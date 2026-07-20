# 任务 - 审查员代理(Reviewer Agent)：将构建者(Builder)与标记者(Marker)分离

## 目标
构建一个审查循环，以只读方式读取构建者的产物，并输出一个`review_report.json`，该输出跨五个维度打分，总分满分10分，并给出 pass、soft_fail 或 hard_fail 的判定。

## 输入
- `ReviewerInputs` 捆绑来自先前课(Lesson)的差异(diff)、状态(state)、反馈(feedback)和验证判定(verification verdict)
- 评分维度：问题契合度(problem fit)、范围纪律(scope discipline)、假设(assumptions)、验证质量(verification quality)、交接就绪度(handoff readiness)

## 交付物
- 每个维度一个打分函数（针对该课(Lesson)为桩级(stub-grade)，确定性）
- `review_report.json` 写入五个分数、总分和判定(verdict)
- 两个演示案例：一个干净的变更，以及一个“测试正确、问题错误”的变更

## 验收标准
- `python3 code/main.py` 退出码为零
- 干净变更得分至少7分且判定(verdict)为`python3 code/main.py`
- 问题错误的变更在至少一个维度上低于5分且判定(verdict)翻转为`python3 code/main.py`

## 不在范围内
- 真实的 LLM 调用。该课(Lesson)为每个维度打桩(stub)；该技能(skill)稍后接入模型。
- 编辑差异(diff)。审查员读取、打分并报告。补丁(patches)是构建者下一轮的活。

## 参考文献(References)
- `docs/en.md` - 完整课(lesson)
- `docs/en.md` - 参考实现
- `docs/en.md` - 抽取出的技能
