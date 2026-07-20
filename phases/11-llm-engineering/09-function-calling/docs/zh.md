# 函数调用与工具使用

> LLM 本身无法执行任何操作。它们只能生成文本，这就是全部能力。它们无法查询天气、检索数据库、发送邮件、运行代码或读取文件。你见过的每一个“AI 代理”本质都是一个生成 JSON 的 LLM，该 JSON 指定了要调用的函数——然后由你的代码实际执行调用。模型是大脑，工具是双手，函数调用是连接它们的神经系统。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 11 第 03 课（结构化输出）
**时间：** 约 75 分钟
**关联：** 阶段 11 · 14（模型上下文协议）——当一个工具在多个主机间共享时，从内联函数调用升级到 MCP 服务器。本课涵盖内联情况；MCP 涵盖协议情况。

## 学习目标

- 实现函数调用循环：定义工具架构，解析模型的工具调用 JSON，执行函数，并返回结果
- 设计具有清晰描述和类型化参数的工具架构，使模型能够可靠调用
- 构建多轮代理循环，链接多个函数调用以回答复杂查询
- 处理函数调用边界情况：并行工具调用、错误传播以及防止无限工具循环

## 问题

你构建了一个聊天机器人。用户问：“现在东京的天气如何？”

模型回应：“我无法获取实时天气数据，但根据季节，东京目前大约 15 摄氏度……”

这是一种披着免责声明的幻觉。模型不知道天气，也永远不会知道。天气每小时都在变化，而模型的训练数据已经过时数月。

正确答案需要调用 OpenWeatherMap API，获取当前温度，并返回真实数值。模型无法调用 API，但你的代码可以。缺失的环节是一个结构化协议，它允许模型说“我需要用这些参数调用天气 API”，并让你的代码执行它并将结果反馈回来。

这就是函数调用。模型输出结构化 JSON，描述要调用哪个函数以及使用哪些参数。你的应用程序执行该函数。结果被放回对话中。模型使用结果生成最终答案。

没有函数调用，LLM 就像是百科全书；有了它，它们就变成了代理。

## 核心概念

### 函数调用循环

每次使用工具交互都遵循相同的 5 步循环。

```mermaid
sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

第 1 步：用户发送消息。第 2 步：模型接收消息以及工具定义（描述可用函数的 JSON Schema）。第 3 步：模型不是回复文本，而是输出工具调用——一个包含函数名和参数的结构化 JSON 对象。第 4 步：你的代码执行函数并捕获结果。第 5 步：结果返回给模型，模型现在拥有真实数据来生成最终答案。

模型从不执行任何操作。它只决定调用什么函数以及使用什么参数。你的代码是执行者。

### 工具定义：JSON Schema 契约

每个工具通过 JSON Schema 定义，告知模型该函数的作用、接受的参数以及这些参数的类型。

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'Tokyo' or 'San Francisco'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}
```

`description` 字段至关重要。模型读取它们来决定何时以及如何使用工具。一个模糊的描述如“获取天气”会导致较差的工具选择，而“获取指定城市的当前天气，返回摄氏温度及天气情况。”则更佳。描述是对工具选择的提示。

### 供应商比较

每个主要供应商都支持函数调用，但 API 表面有所不同。

|  供应商  |  API 参数  |  工具调用格式  |  并行调用  |  强制调用  |
|----------|--------------|-----------------|---------------|----------------|
|  OpenAI (GPT-5, o4)  |  `tools`  |  `tool_calls[].function`  |  支持（每轮多个）  |  `tool_choice="required"`  |
|  Anthropic (Claude 4.6/4.7)  |  `tools`  |  `content[].type="tool_use"`  |  支持（多个块）  |  `tool_choice={"type":"any"}`  |
|  Google (Gemini 3)  |  `function_declarations`  |  `functionCall`  |  支持  |  `function_calling_config`  |
|  开源权重模型 (Llama 4, Qwen3, DeepSeek-V3)  |  Llama 4 原生 `tools`；其他使用 Hermes 或 ChatML  |  混合  |  依赖模型  |  基于提示或如果支持则使用 `tool_choice`  |

到 2026 年，三家闭源供应商已经在几乎相同的基于 JSON Schema 的格式上达成一致。Llama 4 原生配备了与 OpenAI 格式匹配的 `tools` 字段。开源权重微调模型仍各不相同——Hermes 格式（NousResearch）是第三方微调中最常见的。对于跨主机的共享工具，推荐使用 MCP（阶段 11 · 14）而非内联函数调用——服务器对所有模型相同。

### 工具选择：自动、必需、指定

你可以控制模型何时使用工具。

**自动**（默认）：模型自行决定是否调用工具或直接回复。“2+2等于几？”——直接回复。“天气如何？”——调用工具。

**必需**：模型必须至少调用一个工具。当你确认用户意图需要调用工具时使用此选项。防止模型猜测而非查询真实数据。

**指定函数**：强制模型调用特定函数。`tool_choice={"type":"function", "function": {"name": "get_weather"}}` 保证无论查询内容为何，天气工具都会被调用。用于路由——当上游逻辑已经确定了需要哪个工具时使用此选项。

### 并行函数调用

GPT-4o和Claude可以在单个轮次中调用多个函数。用户问：“东京和纽约的天气怎么样？”模型同时输出两个工具调用：

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

你的代码执行两者（理想情况下并发执行），返回两个结果，模型综合生成单个响应。这将往返次数从2次减少到1次。对于每次查询需要5-10次工具调用的智能体，并行调用可将延迟降低60-80%。

### 结构化输出与函数调用

第3课介绍了结构化输出。函数调用使用相同的JSON Schema机制，但目的不同。

**结构化输出**：强制模型以特定形状生成数据。输出是最终产品。例如：从文本中提取产品信息为`{name, price, in_stock}`。

**函数调用**：模型声明执行动作的意图。输出是中间步骤。例如：`get_weather(city="Tokyo")`——模型请求执行一个动作，而非产生最终答案。

当你想要数据提取时使用结构化输出。当你希望模型与外部系统交互时使用函数调用。

### 安全：不可协商的规则

函数调用是你给予LLM的最危险的能力。模型选择要执行什么。如果你的工具集包含数据库查询，模型会构造查询；如果包含shell命令，模型会编写它们。

**规则1：绝对不要直接将模型生成的SQL传递给数据库。** 模型能够且会生成DROP TABLE、UNION注入或返回所有行的查询。始终使用参数化、始终验证、始终使用操作许可名单。

**规则2：许可名单函数。** 模型只能调用你明确定义的函数。永远不要构建通用的“按名称执行任何函数”工具。如果你有50个内部函数，只暴露用户需要的5个。

**规则3：验证参数。** 模型可能传递像`"; DROP TABLE users; --"`这样的城市名。在执行前，根据预期的类型、范围和格式验证每个参数。

**规则4：清理工具结果。** 如果工具返回敏感数据（API密钥、个人身份信息、内部错误），在将其发送回模型之前进行过滤。模型会原样将工具结果包含在其响应中。

**规则5：限制工具调用速率。** 循环中的模型可能会调用工具数百次。设置一个最大值（每次对话10-20次调用是合理的）。打破无限循环。

### 错误处理

工具会失败。API超时。数据库宕机。文件不存在。模型需要知道工具何时失败以及失败原因。

将错误作为结构化工具结果返回，而非异常：

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

模型读取此信息，调整其参数并重试。模型善于从结构化错误消息中自我纠正，但难以从空响应或通用的“出了点问题”错误中恢复。

### MCP：模型上下文协议

MCP是Anthropic为工具互操作性制定的开放标准。不是每个应用程序定义自己的工具，MCP提供了通用协议：工具由MCP服务器提供，由MCP客户端使用（如Claude Code、Cursor或你的应用程序）。

一个MCP服务器可以向任何兼容客户端暴露工具。Postgres MCP服务器为任何兼容MCP的智能体提供数据库访问。GitHub MCP服务器为任何智能体提供仓库访问。工具定义一次，到处使用。

MCP之于函数调用，就像HTTP之于网络。它标准化了传输层，使得工具变得可移植。

## 动手构建

### 步骤1：定义工具注册表

构建一个存储工具定义及其实现的注册表。每个工具有一个JSON Schema定义（模型看到的）和一个Python函数（你的代码执行的）。

```python
import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }
```

### 步骤2：实现5个工具

构建计算器、天气查询、网络搜索模拟器、文件读取器和代码运行器。

```python
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}
```

### 步骤3：注册所有工具

```python
def register_all_tools():
    register_tool(
        "calculator", "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"}, "precision": {"type": "integer", "description": "Decimal places in result", "default": 2}}, "required": ["expression"]},
        calculator,
    )
    register_tool(
        "get_weather", "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"}}, "required": ["city"]},
        get_weather,
    )
    register_tool(
        "web_search", "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3}}, "required": ["query"]},
        web_search,
    )
    register_tool(
        "read_file", "Read the contents of a file. Returns the file content, size, and line count.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"}}, "required": ["path"]},
        read_file,
    )
    register_tool(
        "run_code", "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "language": {"type": "string", "enum": ["python"], "description": "Programming language"}}, "required": ["code"]},
        run_code,
    )
```

### 步骤4：构建函数调用循环

这是核心引擎。它模拟模型决定调用哪个工具，执行工具，并将结果反馈回去。

```python
def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}
```

### 第5步：参数验证

在执行前构建一个验证器，根据JSON Schema检查工具调用参数。

```python
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors
```

### 第6步：运行演示

```python
def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")
```

## 使用它

### OpenAI 函数调用

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)
```

OpenAI 返回工具调用为`response.choices[0].message.tool_calls`。每个调用都有一个`id`，返回结果时必须包含该ID。模型使用此ID将结果与调用匹配。GPT-4o可以在单个响应中返回多个工具调用——迭代执行所有调用。

### Anthropic工具使用

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
# )
#
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

Anthropic 返回工具调用为带有`type: "tool_use"`的内容块。工具结果需要放在一个包含`type: "tool_result"`的用户消息中。注意关键区别：Anthropic使用`input_schema`进行工具参数定义，而OpenAI使用`parameters`。

### MCP 集成

```python
# MCP servers expose tools over a standardized protocol.
# Any MCP-compatible client can discover and call these tools.
#
# Example: connecting to a Postgres MCP server
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})
```

MCP将工具实现与工具消费解耦。Postgres服务器知道SQL。GitHub服务器知道API。您的智能体只需发现和调用工具——无需为每个集成编写特定于提供商的代码。

## 发布

本课生成`outputs/prompt-tool-designer.md`——一个可重用的提示模板，用于设计工具定义。给定您希望工具执行的操作描述，它会生成包含描述、类型和约束的完整JSON Schema定义。

本课还生成`outputs/skill-function-calling-patterns.md`——一个用于在生产环境中实现函数调用的决策框架，涵盖工具设计、错误处理、安全性和特定于提供商的模式。

## 练习

1. **添加第6个工具：数据库查询。** 实现一个模拟的SQL工具，带有内存表。该工具接受表名和过滤条件（不是原始SQL）。验证表名是否在白名单中，过滤运算符是否限制为`=`、`>`、`<`、`>=`、`<=`。返回匹配的行作为JSON。

2. **实现带错误反馈的重试。** 当工具调用失败时（例如，找不到城市），将错误消息反馈给模型决策函数，并让其修正参数。跟踪每次调用需要重试的次数。每次工具调用最多重试3次。

3. **构建一个多步骤智能体。** 某些查询需要链式工具调用：“读取配置文件并告诉我配置了哪个模型，然后搜索该模型的定价。”实现一个循环，直到模型决定不再需要工具为止，将累积结果传递给每个决策步骤。限制为10次迭代以防止无限循环。

4. **衡量工具选择准确率。** 创建30个包含预期工具名称的测试查询。对全部30个查询运行决策函数，并测量其选择正确工具的百分比。找出哪些查询导致工具之间最混淆。

5. **实现工具调用缓存。** 如果在60秒内使用相同参数调用了相同工具，则返回缓存结果而不是重新执行。使用以`(tool_name, frozenset(args.items()))`为键的字典。在包含20个查询的对话中衡量缓存命中率。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  函数调用  |  “工具使用”  |  模型输出结构化JSON，描述要调用带有特定参数的函数——您的代码执行它，而不是模型  |
|  工具定义  |  “函数模式”  |  描述工具名称、用途、参数和类型的JSON Schema对象——模型读取它以决定何时及如何使用工具  |
|  工具选择  |  “调用模式”  |  控制模型是否必须调用工具（required）、可以调用工具（auto）或必须调用特定工具（named）  |
|  并行调用  |  “多工具”  |  模型在单次交互中输出多个工具调用，减少往返次数——GPT-4o和Claude都支持  |
|  工具结果  |  “函数输出”  |  执行工具后的返回值，作为消息发送回模型，以便模型在响应中使用真实数据  |
|  参数验证  |  “输入检查”  |  在执行工具前验证模型生成的参数是否符合预期的类型、范围和约束  |
|  MCP  |  “工具协议”  |  模型上下文协议——Anthropic的开放标准，通过服务器暴露工具，任何兼容的客户端都可以发现和调用  |
|  智能体循环  |  “ReAct循环”  |  模型决策-工具、代码执行-工具、结果反馈-直至模型有足够信息响应的迭代循环  |
|  工具污染  |  “通过工具的提示注入”  |  一种攻击手段，工具结果包含操纵模型行为的指令——需清理所有工具输出  |
|  速率限制  |  “调用预算”  |  设置每次对话的最大工具调用次数，以防止无限循环和失控的API成本  |

## 延伸阅读

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)——使用GPT-4o进行工具调用的权威参考，包括并行调用、强制调用和结构化参数
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)——Claude的工具使用实现，包含input_schema、多工具响应和tool_choice配置
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)——跨AI应用工具互操作的开放标准，采用服务器/客户端架构
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)——关于训练LLM决定何时以及如何调用外部工具的基础论文
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)——微调LLM以在1,645个API上准确调用并减少幻觉
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)——实时基准测试，比较GPT-4o、Claude、Gemini和开源模型的功能调用准确率
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)——思考-行动-观察循环，它是每个工具调用外层的智能体循环；本课结束时，第14阶段将接续。
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)——五种可组合模式（提示链、路由、并行化、协调器-工作者、评估器-优化器），由单一的工具使用原语构建而成。
