#!/usr/bin/env python3
# 静态站生成器：把各章 chapters/*.md 渲染为多页网站（index + 每章一页 + 共享 CSS）。
# 路径相对 .book/。复用 assemble.py 的 PARTS 顺序与 META 元数据。
import os
import re
import markdown

BASE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(BASE, "site")
CH_OUT = os.path.join(SITE, "chapters")
os.makedirs(CH_OUT, exist_ok=True)

META = """# 智能体工程实战：从 ReAct 循环到生产级 Multi-Agent
Agent Engineering in Practice — From the ReAct Loop to Production Multi-Agent Systems

**创建者**: 标叔
**为谁创建**: 会用 API 但说不清 Agent 底层逻辑的开发者；想从"调包"进阶到"造 Agent"的工程师
**基于**: 本文件夹开源课程 ai-engineering-from-scratch（503 课 / 20 阶段），重点蒸馏 Phase 11·14·13
**最后更新**: 2026-07-20
**适用场景**: 系统学智能体原理 + 主流框架实战 + 生产化落地
"""

PARTS = [
    ("Part 1：起步 —— 亲手跑通第一个 Agent", [
        "01-the-loop-changed-an-industry.md",
        "02-react-loop.md",
        "03-tool-registry.md",
        "04-first-real-agent.md",
        "05-from-toy-to-real.md",
    ]),
    ("Part 2：核心能力 —— Agent 的五脏六腑", [
        "06-memory.md",
        "07-planning.md",
        "08-reflection.md",
        "09-context-engineering.md",
        "10-failure-modes.md",
    ]),
    ("Part 3：工程化与框架 —— Use It", [
        "11-framework-comparison.md",
        "12-claude-agent-sdk.md",
        "13-openai-agents-sdk.md",
        "14-computer-use.md",
        "15-observability.md",
        "16-prompt-injection-defense.md",
    ]),
    ("Part 4：进阶与多智能体 —— 案例集", [
        "17-multi-agent-orchestration.md",
        "18-voice-agents.md",
        "19-production-runtime-cost.md",
        "20-capstone-agent-workbench.md",
    ]),
    ("附录", [
        "90-appendices.md",
    ]),
]

MD = markdown.Markdown(extensions=["tables", "fenced_code"])


def render(text: str) -> str:
    MD.reset()
    return MD.convert(text)


def strip_meta(text: str) -> str:
    """去掉章节文件开头的重复元数据块（首个二级标题之前的内容）。"""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("## "):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def chapter_title(text: str) -> str:
    for ln in text.splitlines():
        if ln.startswith("## §") or ln.startswith("## 附录"):
            return ln[3:].strip()
    return "未命名章节"


# ---- 收集章节顺序 ----
order = []  # (slug, title, part_title)
for part_title, files in PARTS:
    for fn in files:
        path = os.path.join(BASE, "chapters", fn)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        body = strip_meta(raw)
        slug = fn[:-3]  # 去掉 .md
        title = chapter_title(body)
        order.append((slug, title, part_title))

# ---- 共享样式 ----
CSS = """\
:root{
  --ink:#1d2129; --sub:#5b6470; --line:#e6e8eb; --bg:#ffffff;
  --panel:#f7f8fa; --accent:#c0392b; --accent2:#2c3e50;
  --code-bg:#f4f5f7; --quote-bg:#fff8f3; --quote-bd:#e8a87c;
  --maxw:760px;
}
*{box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{
  margin:0; color:var(--ink); background:var(--bg);
  font-family:"Noto Serif SC","Source Han Serif SC",Georgia,"Times New Roman",serif;
  font-size:17px; line-height:1.85; -webkit-font-smoothing:antialiased;
}
a{color:var(--accent); text-decoration:none;}
a:hover{text-decoration:underline;}
header.topbar{
  position:sticky; top:0; z-index:20; background:rgba(255,255,255,.92);
  backdrop-filter:blur(6px); border-bottom:1px solid var(--line);
  display:flex; align-items:center; gap:14px; padding:12px 22px;
}
header.topbar .brand{font-weight:700; font-family:"PingFang SC","Microsoft YaHei",sans-serif;
  font-size:15px; color:var(--accent2); letter-spacing:.3px;}
header.topbar .brand a{color:inherit;}
header.topbar nav{margin-left:auto; display:flex; gap:16px; font-family:"PingFang SC",sans-serif; font-size:14px;}
header.topbar nav a{color:var(--sub);}
.layout{display:flex; gap:0; max-width:1180px; margin:0 auto;}
aside.toc{
  width:260px; flex:0 0 260px; position:sticky; top:57px; align-self:flex-start;
  height:calc(100vh - 57px); overflow-y:auto; padding:26px 14px 60px 22px;
  border-right:1px solid var(--line); font-family:"PingFang SC","Microsoft YaHei",sans-serif;
}
aside.toc .pt{font-size:12px; font-weight:700; color:var(--accent); text-transform:uppercase;
  letter-spacing:.5px; margin:18px 0 8px;}
aside.toc a{display:block; color:var(--sub); font-size:13.5px; padding:4px 8px; border-radius:6px;
  line-height:1.45;}
aside.toc a:hover{background:var(--panel); text-decoration:none; color:var(--ink);}
aside.toc a.active{background:var(--quote-bg); color:var(--accent); font-weight:600;}
aside.toc a .num{display:inline-block; min-width:28px; color:var(--accent); font-weight:700;}
main{flex:1; min-width:0; padding:34px 30px 90px;}
.wrap{max-width:var(--maxw); margin:0 auto;}
article h1{font-size:30px; line-height:1.3; margin:.2em 0 .1em; color:var(--accent2);
  font-family:"PingFang SC","Microsoft YaHei",sans-serif;}
article h2{font-size:24px; margin:1.6em 0 .6em; color:var(--accent2);
  font-family:"PingFang SC","Microsoft YaHei",sans-serif; border-left:4px solid var(--accent); padding-left:12px;}
article h3{font-size:19px; margin:1.4em 0 .5em; color:#34495e;
  font-family:"PingFang SC","Microsoft YaHei",sans-serif;}
article p{margin:.7em 0;}
article ul,article ol{padding-left:1.5em;}
article li{margin:.35em 0;}
article code{background:var(--code-bg); padding:.12em .42em; border-radius:5px;
  font-family:"JetBrains Mono",Consolas,Menlo,monospace; font-size:.86em; color:#b5341f;}
article pre{background:var(--code-bg); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; overflow-x:auto; margin:1.1em 0; line-height:1.55;}
article pre code{background:none; padding:0; color:#2d2d2d; font-size:13.5px; line-height:1.6;}
article table{border-collapse:collapse; width:100%; margin:1.2em 0; font-size:15px;
  font-family:"PingFang SC","Microsoft YaHei",sans-serif;}
article th,article td{border:1px solid var(--line); padding:9px 12px; text-align:left; vertical-align:top;}
article thead th{background:var(--panel); color:var(--accent2); font-weight:700;}
article tbody tr:nth-child(even){background:#fcfcfd;}
article blockquote{margin:1.2em 0; padding:14px 18px; background:var(--quote-bg);
  border-left:5px solid var(--quote-bd); border-radius:0 8px 8px 0; color:#5a4632;}
article blockquote p{margin:.4em 0;}
article blockquote strong{color:var(--accent);}
hr{border:none; border-top:1px solid var(--line); margin:2em 0;}
.pager{display:flex; justify-content:space-between; gap:14px; margin-top:48px;
  font-family:"PingFang SC","Microsoft YaHei",sans-serif;}
.pager a{display:block; flex:1; border:1px solid var(--line); border-radius:10px; padding:14px 18px;
  color:var(--ink); background:var(--panel);}
.pager a:hover{text-decoration:none; border-color:var(--accent);}
.pager a .lbl{font-size:12px; color:var(--accent); font-weight:700;}
.pager a .ttl{font-size:14px; color:var(--sub); margin-top:3px;}
.pager a.next{text-align:right;}
/* 封面 */
.cover{max-width:820px; margin:0 auto; padding:54px 30px 80px;}
.cover h1{font-size:38px; line-height:1.25; margin:0 0 .1em; color:var(--accent2);
  font-family:"PingFang SC","Microsoft YaHei",sans-serif;}
.cover .en{color:var(--sub); font-size:16px; margin-bottom:24px; font-style:italic;}
.cover .meta{background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:18px 22px; font-family:"PingFang SC","Microsoft YaHei",sans-serif; font-size:14.5px; line-height:1.9;}
.cover .meta p{margin:.2em 0;}
.cover .cta{display:inline-block; margin:26px 0 10px; background:var(--accent); color:#fff;
  padding:12px 26px; border-radius:10px; font-family:"PingFang SC",sans-serif; font-weight:600;}
.cover .cta:hover{text-decoration:none; opacity:.92;}
.idx-part{font-family:"PingFang SC","Microsoft YaHei",sans-serif; margin-top:30px;}
.idx-part h3{font-size:15px; color:var(--accent); text-transform:uppercase; letter-spacing:.5px;
  border:none; padding:0; margin:0 0 10px;}
.idx-part ol{list-style:none; padding:0; margin:0; display:grid; grid-template-columns:1fr 1fr; gap:8px;}
.idx-part li a{display:block; padding:10px 14px; border:1px solid var(--line); border-radius:9px;
  color:var(--ink); font-size:14.5px; background:#fff;}
.idx-part li a:hover{text-decoration:none; border-color:var(--accent); background:var(--quote-bg);}
.idx-part li a .num{color:var(--accent); font-weight:700; margin-right:8px;}
footer.foot{border-top:1px solid var(--line); color:var(--sub); font-size:13px; text-align:center;
  padding:26px; font-family:"PingFang SC",sans-serif;}
@media (max-width:900px){
  aside.toc{display:none;}
  .idx-part ol{grid-template-columns:1fr;}
  main{padding:24px 18px 70px;}
}
"""

# ---- 侧边目录（所有章节）----
def sidebar_html(active_slug: str) -> str:
    parts_html = []
    for part_title, files in PARTS:
        items = []
        for fn in files:
            slug = fn[:-3]
            # 在 order 中找标题
            title = next((t for s, t, _ in order if s == slug), slug)
            # 章节号
            num = slug.split("-")[0]
            cls = "active" if slug == active_slug else ""
            items.append(
                f'<a class="{cls}" href="{slug}.html"><span class="num">{num}</span>{title}</a>'
            )
        parts_html.append(f'<div class="pt">{part_title}</div>\n' + "\n".join(items))
    return "\n".join(parts_html)


def chapter_page(slug: str, title: str, content_html: str, idx: int) -> str:
    prev_link = ""
    next_link = ""
    if idx > 0:
        ps, pt, _ = order[idx - 1]
        prev_link = (f'<a class="prev" href="{ps}.html">'
                     f'<div class="lbl">← 上一章</div><div class="ttl">{pt}</div></a>')
    else:
        prev_link = '<a class="prev" href="../index.html"><div class="lbl">← 目录</div><div class="ttl">回到封面</div></a>'
    if idx < len(order) - 1:
        ns, nt, _ = order[idx + 1]
        next_link = (f'<a class="next" href="{ns}.html">'
                     f'<div class="lbl">下一章 →</div><div class="ttl">{nt}</div></a>')
    else:
        next_link = ""
    sidebar = sidebar_html(slug)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · 智能体工程实战</title>
<link rel="stylesheet" href="../style.css">
</head>
<body>
<header class="topbar">
  <div class="brand"><a href="../index.html">智能体工程实战</a></div>
  <nav><a href="../index.html">目录</a><a href="../index.html#parts">分卷</a></nav>
</header>
<div class="layout">
<aside class="toc">{sidebar}</aside>
<main><div class="wrap"><article>
{content_html}
<hr>
<div class="pager">{prev_link}{next_link}</div>
</article></div></main>
</div>
<footer class="foot">标叔 · 智能体工程实战 · 基于 ai-engineering-from-scratch 开源课程蒸馏</footer>
</body>
</html>"""


def index_page() -> str:
    # TOC 分组
    parts_html = []
    for part_title, files in PARTS:
        items = []
        for fn in files:
            slug = fn[:-3]
            title = next((t for s, t, _ in order if s == slug), slug)
            num = slug.split("-")[0]
            items.append(f'<li><a href="chapters/{slug}.html"><span class="num">{num}</span>{title}</a></li>')
        parts_html.append(
            f'<div class="idx-part" id="parts"><h3>{part_title}</h3><ol>'
            + "".join(items) + "</ol></div>"
        )
    meta_html = render(META)
    first = order[0][0]
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>智能体工程实战：从 ReAct 循环到生产级 Multi-Agent</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="topbar">
  <div class="brand"><a href="index.html">智能体工程实战</a></div>
  <nav><a href="#parts">分卷</a></nav>
</header>
<div class="cover">
<h1>智能体工程实战</h1>
<div class="en">From the ReAct Loop to Production Multi-Agent Systems</div>
<div class="meta">{meta_html}</div>
<a class="cta" href="chapters/{first}.html">开始阅读 →</a>
{"".join(parts_html)}
</div>
<footer class="foot">标叔 · 智能体工程实战 · 基于 ai-engineering-from-scratch 开源课程蒸馏</footer>
</body>
</html>"""


# ---- 生成 ----
with open(os.path.join(SITE, "style.css"), "w", encoding="utf-8") as f:
    f.write(CSS)

with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as f:
    f.write(index_page())

for idx, (slug, title, part_title) in enumerate(order):
    path = os.path.join(BASE, "chapters", slug + ".md")
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    body = strip_meta(raw)
    html = render(body)
    out = chapter_page(slug, title, html, idx)
    with open(os.path.join(CH_OUT, slug + ".html"), "w", encoding="utf-8") as f:
        f.write(out)

print(f"站点已生成：{SITE}")
print(f"章节页：{len(order)} 个 | index.html + style.css")
