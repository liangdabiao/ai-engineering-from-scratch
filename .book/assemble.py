#!/usr/bin/env python3
# 拼装整书：元数据 + Part 头 + 各章正文。路径相对 .book/。
import os

BASE = os.path.dirname(os.path.abspath(__file__))

META = """# 智能体工程实战：从 ReAct 循环到生产级 Multi-Agent
Agent Engineering in Practice — From the ReAct Loop to Production Multi-Agent Systems

**创建者**: 标叔
**为谁创建**: 会用 API 但说不清 Agent 底层逻辑的开发者；想从"调包"进阶到"造 Agent"的工程师
**基于**: 本文件夹开源课程 ai-engineering-from-scratch（503 课 / 20 阶段），重点蒸馏 Phase 11·14·13
**最后更新**: 2026-07-20
**适用场景**: 系统学智能体原理 + 主流框架实战 + 生产化落地

---

"""

PARTS = [
    ("## Part 1：起步 —— 亲手跑通第一个 Agent", [
        "01-the-loop-changed-an-industry.md",
        "02-react-loop.md",
        "03-tool-registry.md",
        "04-first-real-agent.md",
        "05-from-toy-to-real.md",
    ]),
    ("## Part 2：核心能力 —— Agent 的五脏六腑", [
        "06-memory.md",
        "07-planning.md",
        "08-reflection.md",
        "09-context-engineering.md",
        "10-failure-modes.md",
    ]),
    ("## Part 3：工程化与框架 —— Use It", [
        "11-framework-comparison.md",
        "12-claude-agent-sdk.md",
        "13-openai-agents-sdk.md",
        "14-computer-use.md",
        "15-observability.md",
        "16-prompt-injection-defense.md",
    ]),
    ("## Part 4：进阶与多智能体 —— 案例集", [
        "17-multi-agent-orchestration.md",
        "18-voice-agents.md",
        "19-production-runtime-cost.md",
        "20-capstone-agent-workbench.md",
    ]),
    ("## 附录", [
        "90-appendices.md",
    ]),
]

out = META
for part_title, files in PARTS:
    out += part_title + "\n\n"
    for fn in files:
        path = os.path.join(BASE, "chapters", fn)
        with open(path, encoding="utf-8") as f:
            body = f.read().strip()
        # 去掉章节里重复出现的整书标题（避免重复 H1）
        body = "\n".join(
            ln for ln in body.splitlines()
            if not ln.startswith("# 智能体工程实战")
        ).strip()
        out += body + "\n\n"

dest = os.path.join(BASE, "book.md")
with open(dest, "w", encoding="utf-8") as f:
    f.write(out)

# 简单统计
chapters = sum(len(f) for _, f in PARTS)
chars = len(out)
lines = out.count("\n") + 1
print(f"已生成 {dest}")
print(f"Part 数: {len(PARTS)} | 章节数: {chapters} | 行数: {lines} | 字符数: {chars}")
