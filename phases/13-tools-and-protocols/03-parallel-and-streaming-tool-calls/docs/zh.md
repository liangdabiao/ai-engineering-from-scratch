# 并行工具调用与带工具流式传输

> 串行化三次独立天气查询需要三次往返。并行运行它们，总时间缩减为最慢单次调用。所有前沿提供商现在都在单次交互中发出多个工具调用。收益是真实的；实现细节却很微妙。本课将讲解两个部分：并行扇出和流式参数重组，重点强调ID关联陷阱。

**类型：** 构建
**语言：** Python（标准库，线程池+流式测试框架）
**前提条件：** 阶段13·02（函数调用深入探讨）
**时间：** 约75分钟

## 学习目标

- 解释为什么存在`parallel_tool_calls: true`以及何时禁用它。
- 在并行扇出期间将流式参数块关联到正确的工具调用ID。
- 在不提前解析的情况下将部分`parallel_tool_calls: true`字符串重组为完整JSON。
- 运行一个三城市天气基准测试，展示串行与并行延迟对比。

## 问题

没有并行调用，回答“班加罗尔、东京和苏黎世的天气如何”的智能体会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次LLM往返，每次还付出执行器延迟。大约为理想挂钟时间的4倍。

使用并行调用：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次LLM往返。执行器时间是三者中的最大值，而不是总和。OpenAI、Anthropic和Gemini的生产基准测试显示，在扇出工作负载上挂钟时间减少了60%至70%。

代价是关联复杂性。当三个调用无序完成时，你的结果必须携带匹配的`tool_call_id`，以便模型能对齐它们。当结果流式传输时，你必须在执行前将部分参数片段组装成完整JSON。Gemini 3添加了唯一ID，部分是为了解决一个实际问题：对同一工具的两个并行调用无法区分。

## 核心概念

### 启用并行

- **OpenAI.** 默认开启`parallel_tool_calls: true`。设置`false`强制串行。
- **Anthropic.** 通过`parallel_tool_calls: true`实现并行（Claude 3.5及以上默认）。设置`false`为串行。
- **Gemini.** 始终支持并行；`parallel_tool_calls: true`让模型决定。

当工具存在顺序依赖关系时（先`create_file`后`write_file`），或一个调用的输出影响另一个的输入，或速率限制器无法处理扇出时，禁用并行。

### ID关联

模型发出的每个调用都有一个`id`。主机返回的每个结果必须包含相同的id。否则，结果是不明确的。

- **OpenAI.** 在每个工具角色消息上设置`tool_call_id`。
- **Anthropic.** 在每个`tool_use_id`块上设置`tool_call_id`。
- **Gemini.** 在每个`tool_use_id`上设置`tool_call_id`（Gemini 3及以上；Gemini 2按名称匹配，但同名并行调用会出问题）。

### 并发运行调用

主机在每个调用自己的线程、协程或远程工作者上运行其执行器。最简单的测试框架使用线程池；生产环境使用带`asyncio.gather`或结构化并发的asyncio。完成顺序不可预测——id就是标识符。

一个常见错误：按调用列表顺序而非完成顺序回复结果。这通常可行，因为模型只关心`tool_call_id`，但如果结果丢失或重复，无序提交会使调试更困难。建议按完成顺序回复并附带显式id。

### 流式工具调用

当模型流式传输时，`arguments`以片段形式到达。三个并行调用的三个独立块流在线路上交错。你需要为每个id准备一个累加器。

按提供商的具体形式：

- **OpenAI.** 每个块是`choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带`index`（调用列表中的位置）。你按索引累加，第一次出现时读取`id`，并在`finish_reason = "tool_calls"`时解析JSON。
- **Anthropic.** 流事件是`choices[0].delta.tool_calls[i].function.arguments`，然后每个块有一个`index`，类型为`id`（包含id、名称、空输入）。`finish_reason = "tool_calls"`事件携带`message_start`块。`content_block_start`关闭每个块。
- **Gemini.** `choices[0].delta.tool_calls[i].function.arguments`（Gemini 3及以上）发出带有`index`的块，使调用干净地交错。在Gemini 3之前，流式传输一次返回一个完整调用。

### 部分JSON与过早解析陷阱

在`arguments`完成之前你无法解析它。诸如`{"city": "Beng`的部分JSON无效并会引发错误。正确的门槛是提供商的调用结束信号：OpenAI的`finish_reason = "tool_calls"`、Anthropic的`content_block_stop`或Gemini的流结束事件。只有在那时才尝试`json.loads`。更稳健的方法是使用增量JSON解析器，它在结构完成时产生事件；OpenAI的流式指南推荐这种方法，用于显示实时“思考”指示器的用户体验。花括号计数作为完整性测试不可靠（引号字符串或转义内容中的花括号会导致误报），只应作为非正式调试启发式方法使用。

### 无序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

主机回复仍必须引用这些id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

回复的顺序对OpenAI或Anthropic的正确性无关紧要。只要id匹配，Gemini接受任何顺序。

### 基准测试：串行vs并行

`code/main.py`中的测试框架模拟三个执行器，延迟分别为400、600和800毫秒。串行运行总时长为1800毫秒。并行运行时间为max(400, 600, 800) = 800毫秒。差异是恒定的，而不是成比例的，因此节省随着工具数量增加而增长。

现实世界注意事项：并行调用会给下游API带来压力。向一个速率受限的服务进行10路扇出会失败。阶段13·17涵盖了网关级背压；重试语义计划在未来的阶段中介绍。

### 流式扇出挂钟时间

如果模型本身是流式输出，一旦某个调用的参数完成，你就可以开始执行，而不必等待所有调用都结束。这是OpenAI文档中提到的一种优化，但并非所有SDK都暴露出来。本课的测试工具正是这样做的：一旦模拟流产生一个完整的参数对象，宿主就启动该调用。

## 使用它

`code/main.py` 包含两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 依次和并行运行三个模拟天气调用并打印挂钟时间。第二部分重放一个虚假的流式响应（三个并行调用的 `arguments` 块交错在一条流上），并通过 `StreamAccumulator` 按ID重新组装它们。没有LLM，没有网络，只有重新组装逻辑。

需要关注的内容：

- 串行计时器达到1.8秒。并行计时器在相同的虚假延迟上达到0.8秒。
- 累加器（Accumulator）通过按ID缓冲并仅在每个调用的JSON完成时进行解析来处理乱序到达的块。
- 执行器在某个ID的参数完成后立即启动，而不是等待所有流结束。

## 发布

本课产生 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表（Tool Registry），该技能审计哪些工具可以安全并行化（Parallelize）、哪些具有顺序依赖（Ordering Dependencies）、哪些会压垮下游速率限制（Rate Limits）——返回一个修订后的注册表，其中包含每工具的 `parallel_safe` 标志。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与串行的比率大约为 `max/sum`（由于线程调度、序列化和测试开销，实际运行会略微偏离理想值）。在什么延迟分布下并行不再重要？

2. 扩展累加器以处理“调用在流中取消”的情况：丢弃其缓冲区并发出一个 `cancelled` 事件。哪个提供商明确记录了这种情况？检查Anthropic的 `content_block_stop` 语义和OpenAI的 `finish_reason: "length"` 行为。

3. 将线程池替换为 `asyncio.gather`。对两者进行基准测试。由于上下文切换成本较低，你应该会看到异步的小幅优势，但仅当执行器进行真实I/O时。

4. 选择两个不应该并行化的工具（例如 `create_file` 然后是 `write_file`）。在注册表中添加一个 `ordering_dependency` 图，并在此基础上控制并行扇出（Fan-out）。这是依赖感知调度（Dependency-Aware Scheduling）的最小机制，未来的智能体工程阶段会将其形式化。

5. 阅读OpenAI的并行函数调用部分和Anthropic的 `disable_parallel_tool_use` 文档。找出Anthropic建议禁用并行的唯一一种真实工具类型。（提示：对同一资源产生重要变更。）

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  并行工具调用  |  "单轮扇出"  |  模型在单条助手消息中发出多个工具调用  |
|  `parallel_tool_calls`  |  "OpenAI的标志"  |  启用或禁用多调用发射  |
|  `disable_parallel_tool_use`  |  "Anthropic的逆向标志"  |  选择退出标志；默认启用并行  |
|  工具调用ID  |  "关联句柄"  |  每个调用的标识符，结果消息必须回显  |
|  累加器  |  "流缓冲区"  |  用于部分 `arguments` 块的每ID字符串缓冲区  |
|  乱序完成  |  "最快优先"  |  并行调用以不可预测的顺序完成；ID是粘合剂  |
|  依赖图  |  "排序约束"  |  输出作为其他工具输入的工具；不能并行化  |
|  提前解析陷阱  |  "JSON.parse爆炸"  |  尝试解析不完整的 `arguments` 字符串  |
|  `streamFunctionCallArguments`  |  "Gemini 3特性"  |  带每个调用唯一ID的流式参数块  |
|  完成顺序回复  |  "不等待全部"  |  结果按到达顺序回复，以ID为键  |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和选择退出标志
- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) 和结果批处理
- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — Gemini 3的ID关联并行调用
- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — OpenAI流的块式参数重组
- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 带有 [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) 和 `disable_parallel_tool_use`
