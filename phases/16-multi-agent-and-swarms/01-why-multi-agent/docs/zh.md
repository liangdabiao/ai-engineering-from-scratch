# 为什么多智能体？

> 单个智能体会碰壁。明智之举不是用更大的智能体，而是用更多的智能体。

**类型：** 学习
**语言：** TypeScript
**先修知识：** 阶段14（智能体工程）
**时间：** 约60分钟

## 学习目标

- 识别单智能体的天花板（上下文溢出、混合专长、顺序瓶颈），并解释何时拆分为多个智能体是正确的做法
- 比较编排模式（流水线、并行扇出、监督者、层级结构），并为给定任务结构选择合适的一种
- 设计一个具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂度（延迟、成本、调试难度）与单智能体简单性之间的权衡

## 问题

你在阶段14构建了一个单智能体。它能工作。它可以读取文件、运行命令、调用API并对结果进行推理。然后你将它指向一个真实的代码库：200个文件、三种语言、依赖基础设施的测试，以及需要在编写代码前研究外部API的要求。

智能体卡住了。不是因为LLM笨，而是因为任务超出了单个智能体循环能处理的范围。上下文窗口被文件内容填满。智能体忘记了40次工具调用前读取的内容。它试图同时充当研究员、编码员和审阅者，结果三者都做得很差。

这就是单智能体的天花板。每当任务需要以下条件时，你都会遇到它：

- **比一个窗口能容纳的更多上下文** —— 读取50个文件会超过20万token
- **不同阶段需要不同专长** —— 研究需要的提示与代码生成不同
- **可以并行进行的工作** —— 为什么要顺序读取三个文件，当你可以同时读取它们？

## 核心概念

### 单智能体的天花板

单个智能体就是一个循环、一个上下文窗口、一个系统提示。想象一下：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三件事会出问题：

1. **上下文饱和** —— 工具结果不断累积。到第30步时，智能体已经消耗了15万token的文件内容、命令输出和先前的推理。第5步的关键细节丢失了。

2. **角色混淆** —— 一个说“你是研究员、编码员、审阅者和测试者”的系统提示会产生一个半研究、半编码、从未完成审阅的智能体。

3. **顺序瓶颈** —— 智能体读取文件A，然后文件B，然后文件C。三次串行的LLM调用。三次串行的工具执行。没有并行性。

### 多智能体解决方案

拆分工作。为每个智能体分配一项工作、一个上下文窗口和一个为该工作调优的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个智能体拥有：
- 一个聚焦的系统提示（“你是代码审阅者。你唯一的工作是发现bug。”）
- 自己的上下文窗口（不被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 实际执行此操作的系统

**Claude Code子智能体** —— 当Claude Code使用`Task`生成子智能体时，它会创建一个具有范围任务的子智能体。父智能体保持其上下文干净。子智能体专注工作并返回摘要。

**Devin** —— 运行一个规划智能体、一个编码智能体和一个浏览器智能体。规划智能体将工作分解为步骤。编码智能体编写代码。浏览器智能体研究文档。每个都有独立的上下文。

**多智能体编码团队（SWE-bench）** —— SWE-bench上表现顶尖的系统使用一个读取代码库的研究员、一个设计修复方案的规划者和一个实现修复的编码员。单智能体系统得分较低。

**ChatGPT深度研究** —— 并行生成多个搜索智能体，每个探索不同的角度，然后综合结果。

### 频谱

多智能体不是二元的。它是一个频谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**单智能体** —— 一个循环，一个提示。适用于简单任务。

**子智能体** —— 父智能体为聚焦的子任务生成子智能体。父智能体维护计划。子智能体报告结果。Claude Code就是如此。

**流水线** —— 智能体顺序运行。智能体A的输出成为智能体B的输入。适用于分阶段工作流：研究 -> 编码 -> 审阅 -> 测试。

**团队** —— 智能体并行运行，共享消息总线。每个都有角色。协调者进行协调。当需要同时使用不同技能时适用。

**集群** —— 许多相同或几乎相同的智能体，共享状态。没有固定协调者。智能体从队列中领取工作。适用于高吞吐量的并行任务。

### 四种多智能体模式

#### 模式1：流水线

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个智能体转换数据并传递下去。易于推理。某个阶段发生故障会阻塞后续流程。

#### 模式2：扇出/扇入

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

将工作分发给并行智能体，然后合并结果。适用于可分解为独立子任务的任务。

#### 模式3：编排器-工作者

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

智能编排器决定做什么，委托给工作者，并综合结果。编排器本身就是一个智能体，具备生成工作者的工具。

#### 模式4：对等群集

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

无中央编排器。智能体之间点对点通信。决策从交互中涌现。调试更困难，但可扩展到大量智能体。

### 何时不使用多智能体

多智能体增加了复杂性。智能体之间的每条消息都是潜在的故障点。调试从“阅读一次对话”变为“追踪跨五个智能体的消息”。

**在以下情况下保持单智能体：**
- 任务适合单一上下文窗口（工作数据低于约10万token）
- 不同阶段不需要不同的系统提示
- 顺序执行足够快
- 任务简单到拆分带来的开销大于价值

**复杂性代价：**
- 每个智能体边界都是一次有损压缩步骤：智能体A的完整上下文被总结为发送给智能体B的消息
- 协调逻辑（谁做什么、何时做、顺序如何）本身就会带来错误
- 延迟增加：N个智能体至少意味着N次串行LLM调用，如果需要来回通信则更多
- 成本倍增：每个智能体独立消耗token

经验法则：如果任务需要少于20次工具调用且适合10万token，保持单智能体。

```figure
swarm-messages
```

## 动手构建

### 步骤1：过载的单智能体

这是一个试图做所有事情的单智能体。它有一个庞大的系统提示和一个容纳研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- 上下文窗口随着每个阶段而增长。到审查步骤时，其中包含研究笔记、代码和先前的推理。
- 系统提示是通用的。无法针对每个阶段进行调优。
- 无法并行运行。

### 步骤2：专业智能体

现在分解它。每个智能体承担一项工作：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专业智能体都有聚焦的提示。每个智能体获得干净的上下文窗口，只包含它需要的输入。

### 步骤3：通过消息协调

通过明确的传递消息将专业智能体连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发送给它的消息。没有上下文污染。研究员的5万token文档阅读永远不会进入审查员的上下文。

### 步骤4：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本使用更多总token（三个智能体，三次独立的LLM调用），但每个智能体的上下文保持干净。由于系统提示专业化，每个阶段的质量得到提升。

## 使用它

本节课将产生一个可复用的提示，用于判断何时使用多智能体。参见`outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专业智能体：一个“测试员”智能体，它从编码员接收代码，从审查员接收审查反馈，然后编写测试
2. 修改流水线，使得审查员可以将反馈发送回编码员进行修订循环（最多2轮）
3. 将顺序流水线转换为扇出：并行运行研究员和“需求分析员”智能体，然后合并它们的输出，再传递给编码员

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  Swarm  |  "AI代理的集体思维"  |  一组对等代理，共享状态，没有固定领导者。行为从局部交互中涌现。  |
|  Orchestrator  |  "老板代理"  |  一个代理，其工具包括生成和管理其他代理。它进行规划和委派，但不一定执行实际工作。  |
|  Coordinator  |  "交警"  |  一个非代理组件（通常是代码而非LLM），根据规则在代理之间路由消息。  |
|  Consensus  |  "代理们达成一致"  |  一种协议，其中多个代理必须达成一致才能继续。用于冲突输出需要解决的情况。  |
|  Emergent behavior  |  "代理自己搞定了"  |  系统级模式，源于代理交互，但未经显式编程。可能有用也可能有害。  |
|  Fan-out / fan-in  |  "代理的MapReduce"  |  将任务分发给并行代理（扇出），然后合并结果（扇入）。  |
|  Message passing  |  "代理互相交谈"  |  代理之间的通信机制：从一个代理发送到另一个代理的结构化数据，取代共享上下文窗口。  |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多代理模式综述
- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 微软的多代理对话框架
- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - Claude Code如何使用Task委派
- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 基于角色的多代理框架
