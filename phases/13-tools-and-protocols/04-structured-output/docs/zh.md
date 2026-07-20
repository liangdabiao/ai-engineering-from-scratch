# 结构化输出——JSON Schema、Pydantic、Zod、受限解码

> "请模型友好地返回JSON"在5%到15%的情况下会失败，即使在前沿模型上也是如此。结构化输出通过受限解码弥补了这一差距：模型被禁止发出任何违反模式的令牌。OpenAI的严格模式、Anthropic的模式化工具使用、Gemini的`responseSchema`、Pydantic AI的`output_type`和Zod的`.parse`是同一思想的五种表面形式。本课将构建模式验证器和严格模式契约，学习者将在每个生产级提取流程中使用它们。

**类型：** 构建
**语言：** Python（标准库，JSON Schema 2020-12子集）
**前置条件：** 第13阶段·02（函数调用深入）
**时间：** 约75分钟

## 学习目标

- 使用正确的约束（枚举、最小/最大值、必需、模式）为提取目标编写JSON Schema 2020-12。
- 解释为什么严格模式和受限解码提供了与“生成后验证”不同的保证。
- 区分三种失败模式：解析错误、模式违反、模型拒绝。
- 交付一个具有类型化修复和类型化拒绝处理的提取流程。

## 问题

一个读取采购订单邮件的代理需要将自由文本转换为`{customer, line_items, total_usd}`。三种方法。

**方法一：提示生成JSON。** “以JSON回复，包含字段customer、line_items、total_usd。”在前沿模型上85%到95%的情况下有效。失败有六种方式：缺少大括号、尾随逗号、类型错误、虚构字段、在令牌限制处截断、泄漏的散文如“这是你的JSON：”。

**方法二：生成后验证。** 自由生成，解析，根据模式验证，失败时重试。可靠但昂贵——每次重试都要付费，并且截断错误每次出现都多花费一个回合。

**方法三：受限解码。** 提供者在解码时强制执行模式。无效令牌从采样分布中屏蔽。输出保证可解析并保证验证通过。失败缩减为一种模式：拒绝（模型认为输入不符合模式）。

每个2026年的前沿提供者都部署了某种形式的方法三。

- **OpenAI。** 如果模型拒绝，响应中包含`response_format: {type: "json_schema", strict: true}`加上`refusal`。
- **Anthropic。** 对`response_format: {type: "json_schema", strict: true}`输入进行模式强制；`refusal`不存在，但`tool_use`且无工具调用是信号。
- **Gemini。** 在请求级别使用`response_format: {type: "json_schema", strict: true}`；2026年Gemini为选定类型提供令牌级语法约束。
- **Pydantic AI。** `response_format: {type: "json_schema", strict: true}`发出一个结构化的`refusal`，类型为`tool_use`。
- **Zod (TypeScript)。** 运行时解析器，根据Zod模式验证提供者输出；与OpenAI的`response_format: {type: "json_schema", strict: true}`配合使用。

共同点：声明一次模式，端到端强制执行。

## 核心概念

### JSON Schema 2020-12——通用语言

每个提供者都接受JSON Schema 2020-12。你使用最多的构造：

- `type`: `object`、`array`、`string`、`number`、`integer`、`boolean`、`null`之一。
- `type`：字段名到子模式的映射。
- `type`：必须出现的字段名列表。
- `type`：允许值的封闭集合。
- `type` / `object`（数字），`array` / `string` / `number`（字符串）。
- `type`：应用于每个数组元素的子模式。
- `type`：`object`禁止额外字段（默认值因模式而异）。

OpenAI严格模式增加了三个要求：每个属性都必须列在`required`中，`additionalProperties: false`无处不在，并且没有未解决的`$ref`。如果违反这些，API在请求时返回400。

### Pydantic，Python绑定

Pydantic v2通过`model_json_schema()`从数据类形状的模型生成JSON Schema。Pydantic AI对此进行封装，因此你可以编写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

并且代理框架将模式转换为OpenAI严格模式、Anthropic `input_schema`或Gemini `responseSchema`。模型的输出作为类型化的`Invoice`实例返回。验证错误会引发`ValidationError`，并带有类型化错误路径。

### Zod，TypeScript绑定

Zod (`z.object({customer: z.string(), ...})`) 是TS中的等价物。OpenAI的Node SDK公开了`zodResponseFormat(Invoice)`，它会转换为API的JSON Schema负载。

### 拒绝

严格模式不能强制模型回答。如果输入不符合模式（“电子邮件是一首诗，不是发票”），模型会发出一个包含原因的`refusal`字段。你的代码必须将此作为一流结果处理，而不是失败。拒绝也可作为安全信号：要求从受保护内容电子邮件中提取信用卡号的模型会返回一个拒绝，并附上安全原因。

### 开源领域的受限解码

开放权重实现使用三种技术。

1. **基于语法的解码** (`outlines`, `guidance`, `lm-format-enforcer`)：从模式构建确定性有限自动机；每一步屏蔽会违反FSM的令牌的logits。
2. **结合JSON解析器的Logit屏蔽**：与模型同步运行流式JSON解析器；每一步计算有效下一个令牌集。
3. **带验证器的推测解码**：廉价的草稿模型提出令牌，验证器强制执行模式。

商业提供者在幕后选择其中之一。2026年的技术水平在短结构化输出上比普通生成更快，在长输出上速度大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效的JSON。在严格模式下不会发生。在非严格提供者上仍可能发生。
2. **模式违反。** 输出可解析但违反模式。在严格模式下不会发生。在严格模式外常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你在严格模式之外（Anthropic工具使用、非严格OpenAI、旧版Gemini）时，恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常就足够了。三次重试可以捕获弱模型的异常情况。超过三次则表明模式(schema)存在问题：模型在某些输入下无法满足要求，需要修复提示词(prompt)或模式。

### 小模型支持

约束解码(Constrained decoding)适用于小模型。在结构化任务上，使用语法约束的30亿参数开源模型，其性能优于使用原始提示词的700亿参数模型。这是结构化输出在生产环境中至关重要的主要原因：它将可靠性从模型规模中解耦出来。

## 使用它

`code/main.py` 在标准库中内置了一个轻量级的 JSON Schema 2020-12 验证器（支持 types、required、enum、min/max、pattern、items、additionalProperties）。它封装一个 `Invoice` 模式，并通过验证器运行模拟的 LLM 输出，展示了解析错误(Parse error)、模式违例(Schema violation)和拒绝(Refusal)路径。在生产环境中，将模拟输出替换为任何提供者的真实响应。

需要关注的内容：

- 验证器返回一个带路径和消息的类型化 `[ValidationError]` 列表。这就是你想要呈现给重试提示的形状。
- 拒绝分支不会重试。它会记录并返回一个类型化的拒绝。第14章09小节将拒绝作为一种安全信号。
- `[ValidationError]` 检查在对抗性测试输入上触发，展示了严格模式(Strict mode)如何阻止幻觉字段。

## 发布

本课程生成 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、工单、简历等），该技能生成一个与严格模式兼容的 JSON Schema 2020-12 以及一个与之对应的 Pydantic 模型，并内置了类型化的拒绝和重试处理。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 是一个负数。确认验证器通过 `minimum` 约束路径拒绝该输入。

2. 扩展验证器以支持带有鉴别器(discriminator)的 `oneOf`。常见情况：`line_item` 是产品(Product)或服务(Service)，通过 `kind` 标记。严格模式对此有微妙的规则；请查阅 OpenAI 的结构化输出指南。

3. 将相同的发票模式(Invoice schema)编写为 Pydantic BaseModel，并比较 `model_json_schema()` 输出与您手工编写的模式。找出 Pydantic 默认设置而手工版本省略的一个字段。

4. 测量拒绝率。构造十个不应被提取的输入（一首歌词、一个数学证明、一封空白邮件），并通过真实提供者的严格模式运行它们。统计拒绝与幻觉输出的数量。这将成为您进行拒绝感知重试的基准事实。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出一个严格模式明确禁止而普通 JSON Schema 允许的结构。然后设计一个非必要地使用该禁止结构的模式，并将其重构为与严格模式兼容。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  JSON Schema 2020-12  |  "模式规范"  |  每个现代提供者使用的 IETF 草案模式方言  |
|  严格模式  |  "保证的模式"  |  OpenAI 通过约束解码强制实现模式的标志  |
|  约束解码  |  "logit 掩码"  |  解码时实施，屏蔽无效的下一个 token  |
|  拒绝  |  "模型拒绝"  |  当输入无法匹配模式时的类型化结果  |
|  解析错误  |  "无效 JSON"  |  输出未能解析为 JSON；在严格模式下不可能出现  |
|  模式违例  |  "形状错误"  |  已解析但违反了 types / required / enum / range  |
|  `additionalProperties: false`  |  "不允许额外字段"  |  禁止未知字段；OpenAI 严格模式要求  |
|  Pydantic BaseModel  |  "类型化输出"  |  生成并验证 JSON Schema 的 Python 类  |
|  Zod 模式  |  "TypeScript 输出类型"  |  用于提供者输出验证的 TS 运行时模式  |
|  语法约束  |  "开源权重约束解码"  |  基于 FSM 的 logit 掩码，如 outlines / guidance 中所用  |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝和模式要求
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 2024年8月发布文章，解释解码保证
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 类型化 output_type 绑定，序列化到每个提供者
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 规范文档
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 企业部署说明和严格模式注意事项
