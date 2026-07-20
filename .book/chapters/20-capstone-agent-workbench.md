## §20 Capstone：能跑真仓库的 Agent 工作台

### 20.1 一句判断：最小工作台是三个文件

2025 年我看过太多团队：写个 3000 行的 AGENTS.md，就当工作台做完了。

模型加载它，忽略大半，照样在老地方翻车。

结论是反的：一个短根文件做路由，一份状态文件每轮读写，一块任务板记录进退。

三个文件。各司其职。够小，以后能长成真系统。

> **标叔的经验**：三文件救过一个 monorepo
>
> 一个 80 万行的仓库，Agent 每次都从头读全局规则，越跑越偏。我换成 AGENTS.md 路由 + state + board，首轮就少跑 40% 无关文件。模型像换了个人。

### 20.2 三文件长这样

`AGENTS.md` 是路由，不是手册。它只指路：

```markdown
# AGENTS.md（路由版，别写长）
- 你在哪：读 agent_state.json
- 还差啥：读 task_board.json
- 深规则：docs/agent-rules.md
- 怎么算成：跑 `pytest -q`
```

`agent_state.json` 是系统记录。记：当前任务、动过的文件、假设、阻塞、下一步。

`task_board.json` 是队列。每条带 id、goal、owner（`builder`/`reviewer`/`human`）、验收标准。

状态落文件，因为聊天历史不可靠。会话会死，对话会被截。文件不会。

### 20.3 验收闸门：Agent 不能自己判自己完工

Agent 太容易喊"弄好了"。

三种谎最常见："看着对""测试过了""验收达成"。

对策：一个确定性闸门，读已有产物，给 pass/fail。

闸门不靠 LLM 判。LLM 判留给 reviewer。

```python
# 验收闸门：确定性函数，不概率
def verify(artifacts):
    findings = []
    if not artifacts["acceptance_ran"]:        # 验收命令跑过没
        findings.append(("block", "验收未执行"))
    if artifacts["exit_code"] != 0:            # 退出码为零吗
        findings.append(("block", "验收非零退出"))
    if artifacts["forbidden_write"]:           # 写了禁写区吗
        findings.append(("block", "越权写文件"))
    blocked = any(sev == "block" for sev, _ in findings)
    return {"passed": not blocked, "findings": findings}
```

`block` 级发现，Agent 改不了。只能人签字 override，记原因和工号。

闸门接到 CI：没 `passed: true`，不准合。它是工作台的决定性一刀。

### 20.4 Reviewer：写代码的手，不能打分

闸门说 `passed: true`。你合了。两天后发现它解错了半道题。

验收必要，不充分。reviewer 问闸门问不了的问题：

这解的是对的题吗？范围悄悄扩了吗？假设写下来了吗？下个会话接得上吗？

reviewer 是另一个循环，不同系统提示，只读不写。

五维打分，每维 0–2，满分 10。低于 7 软挂，低于 5 硬挂。

| 维度 | 它问的是 |
|------|---------|
| 问题契合 | 解的是题，还是隔壁题 |
| 范围纪律 | 改在契约内，还是偷偷长大 |
| 假设 | 隐藏假设写下来没 |
| 验收质量 | 命令真证了目标，还是证了弱版 |
| 交接就绪 | 下个会话接得住吗 |

| 谁来评 | 确定性 | 定性 | 标叔的结论 |
|--------|--------|------|-----------|
| 验收闸门 | 是 | 否 | 管"做没做对" |
| Reviewer | 否 | 是 | 管"做的对不对" |
| 二者都上 | — | — | 缺一不可 |

> **注意**：reviewer 不能改 diff
>
> 它读 diff、写报告，不补丁 diff。要改，下一轮 builder 改，reviewer 再评。混角色，那道缝隙就没了。

### 20.5 多会话交接：让下个会话第一分钟就干活

会话要结束。活没完。

糟糕交接的代价，每次会话都付：下个 Agent 问"上回到哪了"，答案没了，重跑半小时。

交接包自动生成，七字段：

1. `summary`：做了啥（一段）。
2. `changed_files`：diff 一览。
3. `commands_run`：真跑过啥。
4. `failed_attempts`：试过啥、为啥挂。
5. `open_risks`：下回会咬人的风险。
6. `next_action`：下回第一步干啥。
7. `verdict_pointer`：验收+评审报告路径。

`next_action` 是承重墙。没有它，那是状态报告，不是交接。

```python
# 交接生成器：从产物打包，不手写
def generate_handoff(state, verdict, review, feedback, last_k=10):
    tail = feedback[-last_k:] + [f for f in feedback if f["exit"] != 0]
    return {
        "summary": state["summary"],
        "changed_files": state["touched"],
        "next_action": state["next_action"],   # 下回第一步
        "verdict_pointer": verdict["path"],
        "failed_attempts": tail,
    }
```

交接前先清理：改动提交了、临时文件删了、测试绿了、board 状态真。

脏树上的交接，是把烂摊子转寄出去。

### 20.6 把它们装一起

一个 turn 的完整流：

1. 读 `agent_state.json`，空就拉 `task_board` 下一条。
2. 在范围内改一个文件。
3. 跑验收命令，写 `feedback_record`。
4. 闸门 `verify` 产物，出 `verification_report`。
5. 不过，退给人；过，reviewer 评分出 `review_report`。
6. 都过，清理，生成 `handoff.md` + `handoff.json`。
7. 写回 `agent_state.json`，收工。

这就是全书能力的落点：循环（§02）、工具（§03）、记忆（§06）、规划（§07）、反思（§08）、框架（§11–§13）、安全（§16）、可观测（§15）、成本（§19）。

> **标叔的经验**：工作台是能力的总和
>
> 我搭第一个工作台时，只接了循环和工具。后面每学一章，就往里塞一块：记忆、闸门、reviewer、交接。它慢慢从"能跑"变成"敢上生产"。你也应该这么长。

### 20.7 先给结论

三文件起步，闸门兜底，reviewer 把关，交接续命。

这就是生产级 Agent 工作台的样子。

[全书收尾] 从 ReAct 一个循环，到能跑真仓库的工作台。你已走完这条矿脉。剩下的，是去挖你自己的。
