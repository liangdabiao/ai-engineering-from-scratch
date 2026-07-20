# 结构化输出：JSON、模式验证、约束解码

> 你的LLM返回字符串，而你的应用需要JSON。这个差距导致的生产系统崩溃比任何模型幻觉都要多。结构化输出是自然语言与类型化数据之间的桥梁。用对了，你的LLM就变成可靠的API；用错了，你就会凌晨三点用正则表达式解析自由文本。

**类型：**构建
**语言：**Python
**前置要求：**阶段10，第01-05课（从头构建LLM）
**时间：**约90分钟
**相关：**阶段5·20（结构化输出与约束解码）涵盖解码器级别的理论（FSM/CFG logit处理器、Outlines、XGrammar）。本节课专注于生产级SDK接口（OpenAI `response_format`、Anthropic工具使用、Instructor）——如果你想了解API底层发生了什么，请先阅读阶段5·20。

## 学习目标

- 使用OpenAI和Anthropic API参数实现JSON模式和模式约束输出
- 构建Pydantic验证层，拒绝格式错误的LLM输出并用错误反馈重试
- 解释约束解码如何在令牌级别强制生成有效JSON而无需后处理
- 设计稳健的提取提示，可靠地将非结构化文本转换为类型化数据结构

## 问题

你问LLM：“从这段文本中提取产品名称、价格和可用性。”它回答：

```
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

这是一个完全正确的答案。但它对你的应用完全没用。你的库存系统需要`{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要一个具有特定键、特定类型和特定值约束的JSON对象。你不需要一个句子。

天真的解决方案：在提示中添加“以JSON格式回复”。这90%的时候有效。另外10%的情况，模型会将JSON包裹在Markdown代码块中，或添加诸如“这是JSON：”的前言，或因为提前闭合括号而产生语法无效的JSON。你的JSON解析器崩溃，管道中断。你添加try/except和重试循环。重试有时会产生不同的数据。现在你在解析问题之上又多了一致性问题。

这不是一个提示工程问题，而是一个解码问题。模型从左到右生成令牌。在每个位置，它从一个超过10万个选项的词汇表中选择最可能的下一个令牌。这些选项中的大多数在任何给定位置都会产生无效JSON。如果模型刚刚输出了`{"price":`，那么下一个令牌必须是数字、引号（表示字符串）、`null`、`true`、`false`或负号。其他任何东西都会产生无效JSON。没有约束，模型可能选择一个完全合理的英语单词，但在语法上却是灾难性的错误。

## 核心概念

### 结构化输出谱系

结构化输出控制有四个层次，每一层都比上一层更可靠。

```mermaid
graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**基于提示**（“以有效JSON回复”）：无强制。模型通常遵守，但有时不遵守。可靠性：约90%。失败模式：Markdown代码块、前言文本、截断输出、结构错误。

**JSON模式**：API保证输出是有效JSON。OpenAI的`response_format: { type: "json_object" }`启用此功能。输出将无错误解析，但可能不符合你预期的模式——多余的键、错误的类型、缺失的字段。

**模式模式**：API接受一个JSON模式并保证输出匹配它。到2026年，每个主要提供商都原生支持此功能：OpenAI的`response_format: { type: "json_schema", json_schema: {...} }`（也作为`tool_choice="required"`）、Anthropic的带有`input_schema`的工具使用，以及Gemini的`response_schema` + `response_mime_type: "application/json"`。输出具有你指定的精确键、类型和约束。

**约束解码**：在生成过程中的每个令牌位置，解码器屏蔽所有会产生无效输出的令牌。如果模式要求一个数字，而模型即将输出一个字母，则该令牌的概率被设为零。模型只能产生导致有效输出的令牌。这就是OpenAI的结构化输出模式以及Outlines和Guidance等库在底层实现的方式。

### JSON模式：契约语言

JSON模式是你告诉模型（或验证层）输出必须具有什么形状的方式。每个主要的结构化输出系统都使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

这个模式规定：输出必须是一个对象，包含一个字符串`product`、一个非负数`price`、一个布尔值`in_stock`以及一个可选的字符串数组`categories`。任何不匹配的输出都会被拒绝。

模式处理困难的情况：嵌套对象、带类型项的数组、枚举（将字符串约束为特定值）、模式匹配（字符串上的正则表达式）以及组合器（用于多态输出的oneOf、anyOf、allOf）。

### Pydantic模式

在Python中，你不必手动编写JSON模式。你定义一个Pydantic模型，它会为你生成模式。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这会产生与上面相同的JSON模式。Instructor库（以及OpenAI的SDK）直接接受Pydantic模型：传入模型类，返回验证后的实例。如果LLM输出不匹配，Instructor会自动重试。

### 函数调用/工具使用

同一问题的另一种接口。不是直接要求模型产生JSON，而是定义带有类型化参数的“工具”（函数）。模型输出带有结构化参数的函数调用。OpenAI称之为“函数调用”，Anthropic称之为“工具使用”。结果是一样的：结构化数据。

```mermaid
graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["User: Extract product info\nfrom this review text"] --> M["Model processes input"]
        M --> TC["Tool Call:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["Validate against\nfunction schema"]
        V --> R["Structured Result:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

当模型需要选择调用哪个函数而不仅仅是填充参数时，工具使用更优先。如果你有10个不同的提取模式，并且模型必须根据输入选择正确的那个，工具使用同时提供了模式选择和结构化输出。

### 常见失败模式

即使有模式强制，结构化输出也可能以微妙的方式失败。

**幻觉值**：输出与模式匹配，但包含虚构数据。当文本显示$348时，模型输出`{"price": 299.99}`。模式验证无法捕捉这一点——类型正确，值错误。

**枚举混淆**：你将字段约束为`["in_stock", "out_of_stock", "preorder"]`。模型输出`"available"` —— 语义正确，但不在允许集合中。良好的约束解码可以防止这一点。基于提示的方法则不能。

**嵌套对象深度**：深度嵌套的模式（4层以上）会产生更多错误。每一层嵌套都是模型可能失去结构追踪的地方。

**数组长度**：模型可能产生数组中的项过多或过少。模式支持`minItems`和`maxItems`，但并非所有提供商都在解码级别强制实施。

**可选字段遗漏**：模型会遗漏那些技术上可选但在你的用例中语义上重要的字段。即使某些数据有时缺失，也请在模式中将它们设为必填——强制模型显式生成`null`。

## 动手构建

### 步骤1：JSON Schema验证器

从头构建一个验证器，检查Python对象是否匹配JSON Schema。这是在输出端运行的，用于验证合规性。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### 步骤2：Pydantic风格模型到Schema的转换

构建一个最小的类到Schema转换器。定义一个Python类，并自动生成其JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### 步骤3：受限令牌过滤器

模拟受限解码。给定部分JSON字符串和模式，确定当前位置哪些令牌类别是有效的。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### 步骤4：提取管道

将所有内容组合成一个提取管道：定义模式，模拟LLM生成结构化输出，验证输出，并处理重试。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### 步骤5：运行完整管道

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## 使用它

### OpenAI结构化输出

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI的结构化输出模式在内部使用受限解码。模型生成的每个令牌都保证产生的输出匹配Pydantic模式。无需重试。无需验证。约束内置于解码过程中。

### Anthropic工具使用

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic通过工具使用实现结构化输出。模型发出一个工具调用，其结构化参数匹配input_schema。结果相同，API表面不同。

### Instructor库

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor包装任何LLM客户端，并添加带验证的自动重试。如果第一次尝试验证失败，它将错误作为上下文发送回模型，并要求其修复输出。这适用于任何提供商，而不仅仅是OpenAI。

## 发布

本课程产出`outputs/prompt-structured-extractor.md`——一个可复用的提示模板，根据给定的模式定义从任何文本中提取结构化数据。传入JSON Schema和非结构化文本，它返回经过验证的JSON。

它还产出`outputs/skill-structured-outputs.md`——一个决策框架，用于根据你的提供商、可靠性要求和模式复杂性选择正确的结构化输出策略。

## 练习

1. 扩展模式验证器以支持`oneOf`（数据必须精确匹配多个模式之一）。这处理多态输出——例如，一个字段可以是`Product`或`Service`对象，它们具有不同的形状。

2. 构建一个“模式差异”工具，比较两个模式并识别破坏性变更（删除的必填字段、更改的类型）与非破坏性变更（添加的可选字段、放宽的约束）。这对于在生产环境中对你的提取模式进行版本控制至关重要。

3. 实现一个更真实的受限解码模拟器。给定一个JSON Schema和一个包含100个令牌（字母、数字、标点符号、关键字）的词汇表，逐步执行生成过程，在每个位置屏蔽无效令牌。测量每个步骤中词汇表有效的百分比。

4. 构建一个提取评估套件。创建50个产品描述，并带有手动标注的JSON输出。在所有50个样本上运行你的提取管道，并测量完全匹配、字段级准确率和类型合规性。确定哪些字段最难正确提取。

5. 为你的提取管道添加“置信度分数”。对于每个提取的字段，估计模型的置信度（基于令牌概率，或通过运行提取3次并测量一致性）。将低置信度字段标记为需要人工审查。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  JSON模式  |  “返回JSON”  |  保证语法上有效的JSON输出但不强制执行任何特定模式的API标志  |
|  结构化输出  |  “类型化JSON”  |  匹配特定JSON Schema（包含正确的键、类型和约束）的输出  |
|  受限解码  |  “引导生成”  |  在每个令牌位置，屏蔽那些会产生无效输出的令牌——保证100%的模式合规性  |
|  JSON Schema  |  “JSON模板”  |  一种声明性语言，用于描述JSON数据的结构、类型和约束（被OpenAPI、JSON Forms等使用）  |
|  Pydantic  |  “Python数据类+”  |  定义具有类型验证的数据模型的Python库，被FastAPI和Instructor用于生成JSON Schema  |
|  函数调用  |  “工具使用”  |  LLM输出一个结构化的函数调用（名称+类型化参数）而非自由文本——OpenAI和Anthropic都支持此功能  |
|  Instructor  |  “面向LLM的Pydantic”  |  包装LLM客户端以返回经过验证的Pydantic实例的Python库，在验证失败时自动重试  |
| Token masking（标记掩码） | "词汇过滤" | 在生成过程中将特定标记的概率设为零，使模型无法生成它们 |
| Schema compliance（架构合规性） | "符合形状" | 输出包含所有必需字段，类型正确，值在约束范围内，且没有额外的不允许字段 |
| Retry loop（重试循环） | "重试直至成功" | 将验证错误发送回模型并要求其修复输出 —— Instructor 自动执行此操作，最多可配置重试次数 |

## 延伸阅读

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- OpenAI API中基于JSON Schema的约束解码的官方文档
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- Outlines论文，描述了如何将JSON Schema编译成用于标记级约束的有限状态机
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- 从任何LLM获取结构化输出的标准库，带有Pydantic验证和重试
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- Claude如何通过工具使用实现结构化输出，使用JSON Schema input_schema
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- 每个主要结构化输出系统使用的模式语言的完整规范
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- 开源约束生成，使用正则表达式和编译成有限状态机的JSON Schema
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- 当前最先进的语法引擎；下推自动机编译，以约100纳秒/令牌的速度屏蔽令牌
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- LMQL论文，将约束解码构建为带有类型和值约束的查询语言
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- 模板驱动的约束生成；与Outlines和XGrammar无关的供应商补充
