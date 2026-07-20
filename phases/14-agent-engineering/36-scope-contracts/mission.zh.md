# 任务目标 - 范围契约与任务边界

## 目标
为每个任务编写一份 `scope_contract.json` 以及一个支持 glob 的检查器，将智能体的 diff 与契约进行比对，并标记任何被禁止或超出范围的写入。

## 输入
- 一份包含允许 glob、禁止 glob、验收命令、回滚段落、所需审批的任务描述
- 两次演示运行：一次在范围内，一次越界

## 交付物
- `scope_contract.json` 模式校验器（JSON Schema 子集，glob 数组）
- 一个从被触碰文件及运行命令生成 `scope_contract.json` 的 diff 解析器
- `scope_contract.json`
- `scope_contract.json` 保存在脚本旁边

## 验收标准
- `python3 code/main.py` 以零状态退出
- 范围内运行报告零违规
- 越界运行报告确切的越界文件及各自原因

## 不在范围内
- 时间预算、网络出口允许列表。本课提供文件 glob；练习提示对其进行扩展。
- 接入运行时中断。本课在报告处结束。

## 参考文献(References)
- `docs/en.md` - 完整课(lesson)
- `docs/en.md` - 参考实现
- `docs/en.md` - 抽取出的技能
