#!/usr/bin/env python3
# 把拼装好的 book.md 渲染为打印友好的 HTML，并补上目录(TOC)与锚点跳转。
# 浏览器打开后「打印 → 另存为 PDF」（A4）即可出书；屏幕上可点目录跳转。
import os
import markdown

BASE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(BASE, "book.md")
dest = os.path.join(BASE, "book.html")

with open(src, encoding="utf-8") as f:
    md = f.read()

# toc 扩展会自动给所有标题加 id，并可通过 MD.toc 拿到目录 HTML
MD = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc", "nl2br"],
    extension_configs={"toc": {"permalink": False, "toc_depth": "2-3"}},
)
html_body = MD.convert(md)
toc_html = MD.toc  # <div class="toc"><ul>... 含锚点

# 在 H1 标题之后插入目录卡片
html_body = html_body.replace(
    "</h1>", f'</h1>\n<div class="toc-card"><h2 class="toc-title">目录</h2>{toc_html}</div>', 1
)

CSS = """
@page {
  size: A4;
  margin: 18mm 16mm;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  font-family: "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
               "Microsoft YaHei", "Hiragino Sans GB", sans-serif;
  color: #1a1a1a;
  line-height: 1.75;
  font-size: 15px;
  max-width: 820px;
  margin: 0 auto;
  padding: 24px;
}
h1 { font-size: 28px; border-bottom: 3px solid #e0533a; padding-bottom: 10px; }
h2 { font-size: 21px; color: #c0392b; margin-top: 34px;
     border-left: 5px solid #e0533a; padding-left: 10px; }
h3 { font-size: 17px; color: #222; margin-top: 22px; }
h4 { font-size: 15px; color: #444; }
code {
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  background: #f4f4f5; padding: 1px 5px; border-radius: 4px; font-size: 13px;
}
pre {
  background: #1e1e2e; color: #e4e4e7; padding: 14px 16px; border-radius: 8px;
  overflow-x: auto; font-size: 12.5px; line-height: 1.5;
}
pre code { background: none; color: inherit; padding: 0; }
blockquote {
  border-left: 4px solid #e0533a; background: #fff6f4; margin: 16px 0;
  padding: 10px 16px; border-radius: 0 6px 6px 0; color: #5a2a22;
}
blockquote p { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13.5px; }
th, td { border: 1px solid #d0d0d0; padding: 7px 10px; text-align: left; vertical-align: top; }
thead th { background: #2d2d3a; color: #fff; }
tbody tr:nth-child(even) { background: #fafafa; }
/* 高亮「标叔的结论」列（最后一列） */
td:last-child, th:last-child { background: #fff3e0; font-weight: 600; }
hr { border: none; border-top: 1px solid #eee; margin: 22px 0; }
ul, ol { padding-left: 22px; }
a { color: #c0392b; text-decoration: none; }
a:hover { text-decoration: underline; }
/* 目录卡片 */
.toc-card {
  border: 1px solid #eadfd9; background: #fffaf7; border-radius: 12px;
  padding: 14px 20px 18px; margin: 22px 0 30px;
}
.toc-title { margin: 4px 0 12px !important; border: none !important; padding: 0 !important; }
div.toc { font-size: 14.5px; }
div.toc > ul { list-style: none; padding-left: 0; }
div.toc ul ul { padding-left: 20px; list-style: none; }
div.toc li { margin: 5px 0; }
div.toc a { color: #34495e; }
div.toc a:hover { color: #c0392b; }
/* 屏幕端：回到目录悬浮按钮 */
.to-top {
  position: fixed; right: 22px; bottom: 22px; z-index: 30;
  background: #c0392b; color: #fff; width: 46px; height: 46px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; font-size: 13px;
  box-shadow: 0 4px 14px rgba(0,0,0,.18); cursor: pointer; text-decoration: none;
}
.to-top:hover { opacity: .92; }
/* 打印：避免代码块/表格跨页断裂 */
pre, table, blockquote { page-break-inside: avoid; }
h2, h3 { page-break-after: avoid; }
@media print {
  body { padding: 0; max-width: none; }
  h1 { page-break-after: always; }
  .to-top { display: none; }
}
"""

doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>智能体工程实战：从 ReAct 循环到生产级 Multi-Agent</title>
<style>{CSS}</style>
</head>
<body>
{html_body}
<a class="to-top" href="#" title="回到顶部">↑目录</a>
</body>
</html>
"""

with open(dest, "w", encoding="utf-8") as f:
    f.write(doc)

print(f"已生成 {dest}（{len(doc)} 字节）")
print("目录与锚点跳转已加入；浏览器打开后：打印 → 另存为 PDF（A4）即可出书。")
