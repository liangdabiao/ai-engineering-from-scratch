# 对话状态追踪

> "我想在北边找一家便宜的餐厅……实际上改成中等的……再加意大利菜。"三轮对话，三次状态更新。DST保持槽值字典同步，从而成功预订。

**类型：** 构建
**语言：** Python
**前置条件：** 第5阶段·17（聊天机器人），第5阶段·20（结构化输出）
**时间：** 约75分钟

## 问题

在面向任务的对话系统中，用户目标被编码为一组槽值对：`{cuisine: italian, area: north, price: moderate}`。每个用户轮次可以添加、更改或移除一个槽。系统必须读取整个对话并正确输出当前状态。

只要一个槽出错，系统就会预订错误的餐厅、安排错误的航班或错误地扣款。DST是用户所说与后端执行之间的关键环节。

为什么在2026年尽管有大语言模型，DST仍然重要：

- 合规敏感领域（银行、医疗、机票预订）需要确定性的槽值，而不是自由格式生成。
- 工具使用代理在调用API前仍需要槽解析。
- 多轮修正比看起来更难："实际上不，改成周四。"

现代流程：经典DST概念 + 大语言模型提取器 + 结构化输出护栏。

## 核心概念

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** 模式定义了领域（餐厅、酒店、出租车）及其槽（菜系、区域、价格、人数）。每个槽可以为空，填充来自封闭集合的值（价格：{便宜, 中等, 昂贵}），或自由格式的值（名称："The Copper Kettle"）。

**两种DST形式。**

- **分类。** 对于每个(槽, 候选值)对，预测是/否。适用于闭词表槽。2020年前的标准方法。
- **生成。** 给定对话，生成槽值作为自由文本。适用于开词表槽。现代默认方法。

**指标。** 联合目标准确率（JGA）——每个槽都正确的轮次比例。全有或全无。MultiWOZ 2.4排行榜在2026年最高约83%。

**架构。**

1. **基于规则（槽正则表达式+关键词）。** 窄领域的强基线。可调试。
2. **TripPy / BERT-DST.** 基于BERT编码的复制生成。预大语言模型标准。
3. **LDST（LLaMA + LoRA）。** 指令微调的大语言模型，带有领域槽提示。在MultiWOZ 2.4上达到ChatGPT级别质量。
4. **免本体（2024–26）。** 跳过模式；直接生成槽名和值。处理开放领域。
5. **提示+结构化输出（2024–26）。** 大语言模型配合Pydantic模式+约束解码。5行代码，生产就绪。

### 经典失败模式

- **跨轮共指。** "我们选第一个选项。"需要解析是哪个选项。
- **覆盖 vs 追加。** 用户说"加意大利菜。"是替换菜系还是追加？
- **隐式确认。** "好的"——这是接受了提供的预订吗？
- **修正。** "实际上改成晚上7点。"必须更新时间而不清除其他槽。
- **对之前系统话语的共指。** "是的，那个。"哪个"那个"？

## 动手构建

### 第1步：基于规则的槽提取器

参见`code/main.py`。正则表达式+同义词词典覆盖窄领域典型话语的70%：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在典型词汇之外脆弱。适用于确定性槽确认。

### 第2步：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变条件：

- 永远不要重置用户未触及的槽。
- 显式否定（"算了，菜系不管了"）必须清除。
- 用户修正（"实际上……"）必须覆盖，而非追加。

### 第3步：基于大语言模型的结构化输出DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic保证有效的状态对象。无需正则表达式，无模式不匹配，无幻觉槽。

### 第4步：JGA评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在所有轮次中正确获取所有槽的比例是多少？对于MultiWOZ 2.4，2026年顶级系统：80-83%。你的领域内系统在窄词汇上应超过该比例，否则大语言模型基线会胜出。

### 第5步：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

在检测到修正时，覆盖最后更新的槽而非追加。没有大语言模型辅助很难做好。现代模式：始终让大语言模型从历史中重新生成整个状态，而非增量更新——这自然处理了修正。

## 陷阱

- **全历史重新生成成本。** 让大语言模型每轮重新生成状态总计花费O(n²)个词元。限制历史或总结更早的轮次。
- **模式漂移。** 事后添加新槽会破坏旧训练数据。为你的模式做版本管理。
- **大小写敏感。** "Italian" vs "italian" vs "ITALIAN" —— 处处归一化。
- **隐式继承。** 如果用户之前指定了"4人"，那么对另一个时间的请求不应清除人数。始终传递完整历史。
- **自由格式 vs 封闭集合。** 姓名、时间和地址需要自由格式槽；菜系和区域是封闭的。在模式中混合两者。

## 使用它

2026年技术栈：

|  情况  |  方法  |
|-----------|----------|
|  窄领域（一个或两个意图）  |  基于规则+正则表达式  |
| 宽领域，有标注数据可用 | LDST（在MultiWOZ风格数据上使用LLaMA + LoRA） |
| 宽领域，无标签，生产就绪 | LLM + Instructor + Pydantic schema |
| 语音/口语 | ASR（自动语音识别）+ 标准化器 + LLM-DST |
| 多领域预订流程 | 基于模式指导的LLM，每个领域使用Pydantic模型 |
| 合规敏感 | 规则为主，LLM回退带确认流程 |

## 发布

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单.** 在`code/main.py`中构建基于规则的对话状态跟踪器，用于3个槽位（菜系、区域、价格）。在10个人工对话上测试。测量JGA（联合目标准确率）。
2. **中等.** 使用相同数据集，搭配Instructor + Pydantic + 一个小型LLM。比较JGA。检查最难的轮次。
3. **困难.** 实现两者并路由：规则为主，当基于规则输出少于2个槽位且置信度低时，LLM回退。测量组合JGA和每轮推理成本。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| DST | 对话状态跟踪（Dialogue State Tracking） | 跨对话轮次维护槽值字典。 |
| Slot | 用户意图的单元 | 后端需要的命名参数（菜系、日期）。 |
| Domain | 任务领域 | 餐厅、酒店、出租车——槽位集合。 |
| JGA | 联合目标准确率（Joint Goal Accuracy） | 每个槽位都正确的轮次比例。全有或全无。 |
| MultiWOZ | 基准数据集 | 多领域WOZ数据集；标准DST评估。 |
| 无本体DST（Ontology-free DST） | 无模式 | 直接生成槽位名称和值，无固定列表。 |
| 修正（Correction） | “实际上...” | 覆盖先前填充槽位的轮次。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 标准基准。
- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 用于DST的LLaMA + LoRA指令微调。
- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 基于复制的DST主力。
- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 基于EM的无监督TOD。
- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 标准DST结果。
