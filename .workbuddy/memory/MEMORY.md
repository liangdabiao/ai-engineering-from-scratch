# 项目长期记忆：技术图书（智能体工程实战）

## 部署约束（用户 2026-07-20 明确，最高优先级）
- **不自行部署 site/ 或任何产物到公网，必须等用户明确确认。**
- 所有写操作只限 `.book/` 内新建/修改文件，**绝不**改动 `phases/`、`README.md` 与根仓库其他内容。
- 已发生：15:39 误触发一次 CloudStudio 部署（`workbuddy_cloudstudio_deploy`，返回空），用户立即叫停并要求核查是否生成对外 URL。
- 后续处理：以"核查是否生成 URL + 等用户确认"为准，绝不重试或换工具再部署。

## 产物与工具链（均在 `.book/`）
- `research.md` / `plan.md`：研究纪要与方案。
- `chapters/01-20.md` + `90-appendices.md`：20 章 + 附录（标叔风格）。
- `assemble.py`：拼装 `book.md`。`qc.py`：全自动 QC（20/20 通过）。
- `md_to_html.py`：渲染打印友好 `book.html`（A4，浏览器"打印→另存为 PDF"）。
- `build_site.py` + `site/`：多页静态站（index + 21 章页 + style.css），本地预览用，未上线。
- 依赖：managed venv 装 `markdown` 3.10.2（`C:\Users\49707\.workbuddy\binaries\python\envs\default`）。

## 风格硬约束（huashu-bookwriter）
第一人称、单句≤25字、禁用词（综上所述/值得注意的是/强大的…）、每章≥1经验框、对比表带"标叔的结论"列、代码带语言标签+关键注释、结尾向前桥接、开头时间线锚点。
