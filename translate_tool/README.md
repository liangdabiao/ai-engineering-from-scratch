# BetterExplained 批量翻译工具

将 BetterExplained 网站的英文 HTML 文章批量翻译为中文，基于 AI 大模型 + 翻译记忆 + 术语表的三层架构。

## 特点

- **翻译记忆库**：相同文本只翻译一次，自动复用（导航栏、页脚等重复内容命中率可达 50%+）
- **术语表管理**：数学/技术术语统一翻译，首次出现自动加中英对照（如"导数(Derivative)"）
- **智能跳过**：自动跳过 MathJax 公式、代码块、HTML 标签、纯数字/URL
- **自动识别已翻译文章**：中文占比 > 30% 的文章自动跳过，不重复翻译
- **断点续传**：进度保存在 `progress.json`，中断后重新运行自动从断点继续
- **429 限流处理**：指数退避重试，稳定应对 API 限流
- **质量检查**：翻译后自动验证 lang 属性、MathJax 完整性、未翻译段落

## 文件结构

```
translate_tool/
├── translate_all.py           # [主入口] 全量翻译脚本，支持断点续传
├── batch_translate.py         # 单篇翻译逻辑，也可独立使用
├── translator.py              # AI 翻译器（OpenAI 兼容 API）
├── html_utils.py              # HTML 解析、节点提取、翻译写回
├── translation_memory.py      # 翻译记忆库管理
├── glossary.py                # 术语表管理
├── verify.py                  # 翻译质量检查工具
├── translation_memory.json    # 翻译记忆数据（自动积累）
├── glossary.json              # 术语表（可手动编辑）
└── progress.json              # 翻译进度（自动维护）
```

## 快速开始

### 1. 安装依赖

```bash
pip install openai beautifulsoup4
```

### 2. 设置 API Key

```powershell
# Windows PowerShell（腾讯云 TokenHub）
$env:TOKENHUB_API_KEY="sk-xxx"

# 或 OpenAI
$env:OPENAI_API_KEY="sk-xxx"
```

### 3. 启动全量翻译

```powershell
# 翻译 articles 目录下的所有文章（推荐方式）
python translate_all.py "D:\My Web Sites\https___betterexplained.com_\betterexplained.com\articles"
```

运行后自动：
- 扫描目录下所有 `*/index.html` 文件
- 跳过已翻译的文章（中文占比 > 30%）
- 跳过 `progress.json` 中已记录的完成项
- 每 5 篇自动保存进度和记忆库
- 遇到中断，重新运行即可从断点继续

### 4. 单篇翻译

```bash
# 翻译单个目录（试运行，不修改文件）
python batch_translate.py ../articles --dry-run

# 正式翻译
python batch_translate.py ../articles

# 指定模型和批次大小
python batch_translate.py ../articles --model=deepseek-v4-flash-202605 --batch-size=20
```

### 5. 质量检查

```bash
# 检查整个目录
python verify.py ../articles

# 检查单个文件
python verify.py ../articles/some-article/index.html
```

检查项包括：
- `<html lang="zh-CN">` 是否正确
- MathJax `$$$` 公式是否配对
- 是否有大段未翻译的英文段落
- HTML 基本结构是否完整

## 核心流程

```
                    ┌─────────────────────┐
                    │  扫描文章目录        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ 已翻译？→ 跳过      │
                    │ (中文占比 > 30%)     │
                    └──────────┬──────────┘
                               │ 否
                    ┌──────────▼──────────┐
                    │  解析 HTML           │
                    │  提取可翻译节点       │
                    │  (跳过公式/代码/标签) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
               ┌────│  查翻译记忆库        │
               │    └──────────┬──────────┘
               │               │
          命中 → 复用     未命中 ↓
               │    ┌──────────▼──────────┐
               │    │  分批送 AI 翻译      │
               │    │  (每批 20 条)        │
               │    │  附带术语表提示       │
               │    └──────────┬──────────┘
               │               │
               └───────┬───────┘
                       │
            ┌──────────▼──────────┐
            │  写回 HTML           │
            │  替换文本节点+属性    │
            │  设置 lang="zh-CN"   │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  更新翻译记忆库       │
            │  保存进度             │
            └─────────────────────┘
```

## 工作原理详解

### 1. HTML 解析（html_utils.py）

使用 BeautifulSoup 解析 HTML，提取两类可翻译内容：

- **文本节点**：`<p>`, `<h1>`-`<h6>`, `<a>`, `<li>`, `<span>` 等标签内的文字
- **属性节点**：`alt`, `title`, `placeholder` 等属性值

**自动跳过**：
- `<script>`, `<style>`, `<code>`, `<pre>` 标签内容
- HTML 注释
- MathJax 公式（`$$$...$$$`）
- 纯数字/符号/URL
- 纯中文文本（已翻译好的内容）

**去重策略**：文本节点和属性节点分别去重。同一文本在页面中出现多次（如导航链接），只翻译一次但写回所有位置。

### 2. 翻译记忆（translation_memory.py）

基于 MD5 哈希的翻译缓存：

```
源文本 → MD5(text)[:12] → 查找 → 命中则直接使用译文
```

**优势**：BetterExplained 所有文章共享同一个 WordPress 模板，导航栏、页脚、侧边栏等内容高度重复。翻译 10 篇文章后，记忆命中率可达 40-50%，大幅减少 API 调用。

### 3. AI 翻译（translator.py）

将待翻译文本组成 JSON 数组，附带术语表，发送给大模型：

- **模型**：当前使用 `deepseek-v4-flash-202605`（腾讯云 TokenHub），也支持 OpenAI 等
- **批次**：每批 20 条文本，避免单次请求过大
- **重试**：最多 3 次，429 限流时指数退避（15s → 30s → 60s）
- **解析**：健壮的 JSON 解析，兼容数组/对象/嵌套等多种返回格式

**翻译规则**（内置 Prompt）：
1. 准确翻译技术术语
2. 保留 HTML 标签和 Markdown 格式
3. 保留 MathJax 公式和代码
4. 中文自然流畅，不生硬直译
5. 专有名词保留原文（Archimedes、BetterExplained 等）
6. 首次出现的重要术语加中英对照：导数(Derivative)

### 4. 翻译写回（html_utils.py）

遍历整个 DOM 树，将译文精确写回原位置：
- 文本节点：替换内容，保留前后空白
- 属性节点：替换指定属性值
- 最后设置 `<html lang="zh-CN">`

### 5. 进度管理（translate_all.py）

- `progress.json` 记录已完成和失败的文章名
- 每 5 篇自动保存进度 + 记忆库
- 重新运行自动跳过已完成的
- 已翻译文章（中文占比 > 30%）自动跳过

## 术语表

编辑 `glossary.json` 添加或修改术语：

```json
{
  "Calculus": "微积分",
  "Derivative": "导数",
  "Integral": "积分",
  "Limit": "极限",
  "Infinitesimal": "无穷小"
}
```

术语表会自动注入翻译 Prompt，确保所有文章对同一术语的翻译一致。

## 配置说明

### 修改翻译模型

编辑 `translate_all.py` 中的 `Translator` 初始化参数：

```python
translator = Translator(
    provider="tokenhub",                    # 提供商
    model="deepseek-v4-flash-202605",       # 模型名
    api_key=api_key,
    request_delay=3.0                        # 请求间隔（秒）
)
```

支持的 provider：
- `tokenhub`：腾讯云 TokenHub（需设置 `TOKENHUB_API_KEY`）
- `openai`：OpenAI 官方 API（需设置 `OPENAI_API_KEY`）

### 修改可翻译标签

编辑 `translate_all.py` 中的 `html_config`：

```python
html_config = {
    'translatable_tags': ['h1', 'h2', 'h3', 'p', 'a', 'li', ...],
    'translatable_attrs': ['alt', 'title', 'placeholder'],
    'skip_tags': ['script', 'style', 'noscript', 'code', 'pre'],
}
```

## 常见问题

### Q: 翻译中断了怎么办？

重新运行同一命令即可。程序会读取 `progress.json`，自动跳过已完成的文章。

### Q: 如何重新翻译某篇文章？

从 `progress.json` 的 `completed` 列表中移除该文章名，然后重新运行。

### Q: 为什么有些文章被跳过？

中文占比 > 30% 的文章会被自动判定为"已翻译"并跳过。如需强制重新翻译，修改 `translate_all.py` 中的 `is_already_translated` 阈值。

### Q: API 返回 429 限流怎么办？

程序会自动指数退避重试。如果频繁限流，可以增大 `request_delay` 或减小 `batch_size`。

## 注意事项

- 翻译前建议使用 git 或备份原始文件
- 首次翻译建议先用单篇测试，确认效果后再全量运行
- 翻译后用 `verify.py` 检查质量
- 重要文章建议人工校对
