#!/usr/bin/env python3
# 全书 QC：每章12项 + 全书10项（标叔风格清单的自动化部分）
import os, re, glob

BASE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(BASE, "chapters")

BANNED = ["综上所述", "值得注意的是", "接下来我们将", "进行操作", "实现功能",
          "强大的", "革命性的", "在当今这个", "首先……其次", "首先...其次"]

def chapter_files():
    # 仅 §NN 章节（01-20），附录单独处理
    files = sorted(glob.glob(os.path.join(CH, "*.md")))
    return [f for f in files if re.match(r".*\\(0[1-9]|1\d|20)-", f)]

def appendix_files():
    return sorted(glob.glob(os.path.join(CH, "90-*.md")))

problems = []
chapters_ok = 0
total_ch = 0

for path in chapter_files():
    name = os.path.basename(path)
    txt = open(path, encoding="utf-8").read()
    total_ch += 1
    ok = True
    # 1. §NN 标题
    if not re.search(r"^## §\d\d .+", txt, re.M):
        problems.append(f"[{name}] 缺 §NN 断言句标题"); ok = False
    # 2. 时间线/经历开头（前 6 行含"我"或年份或具体场景）
    head = "\n".join(txt.splitlines()[:8])
    if not re.search(r"我|20\d\d 年|\d{4} 年|凌晨|一次|2025|2024|2026", head):
        problems.append(f"[{name}] 开头缺时间线/经历锚点"); ok = False
    # 3. 特殊内容块
    if not re.search(r"> \*\*(标叔的经验|核心建议|注意)\*\*", txt):
        problems.append(f"[{name}] 缺标叔经验/建议/注意框"); ok = False
    # 4. 向前桥接
    if "[向前桥接]" not in txt and "[全书收尾]" not in txt:
        problems.append(f"[{name}] 缺向前桥接"); ok = False
    # 5. 对比表含"标叔的结论"列（若章节含 | 表）
    if "|" in txt and "标叔的结论" not in txt:
        problems.append(f"[{name}] 含表格但无'标叔的结论'列"); ok = False
    # 6. 禁用词
    for b in BANNED:
        if b in txt:
            problems.append(f"[{name}] 含禁用词：{b}"); ok = False
    # 7. 代码语言标签
    if "```" in txt:
        # 检查是否有 ```python / ```text 等
        if not re.search(r"```[a-zA-Z]+", txt):
            problems.append(f"[{name}] 代码块缺语言标签"); ok = False
    if ok:
        chapters_ok += 1

# 附录级（轻量：无禁用词、有表、有标叔的结论更好）
appx_problems = []
for path in appendix_files():
    name = os.path.basename(path)
    txt = open(path, encoding="utf-8").read()
    for b in BANNED:
        if b in txt:
            appx_problems.append(f"[{name}] 含禁用词：{b}")
print("\n=== 附录 QC ===")
if appx_problems:
    for p in appx_problems:
        print("  -", p)
else:
    print(f"附录文件 {len(appendix_files())} 个：无禁用词 ✅")

# 全书级
book = open(os.path.join(BASE, "book.md"), encoding="utf-8").read()
meta_ok = book.startswith("# 智能体工程实战")
# Part 分组
parts = re.findall(r"^## Part \d|^## 附录", book, re.M)
# 编号连续：提取 §NN
nums = [int(m) for m in re.findall(r"^## §(\d\d) ", book, re.M)]
contiguous = nums == list(range(1, 21))
# 第一个 H1 是标题
first_h1 = book.splitlines()[0].startswith("# ")

print("=== 每章 QC ===")
print(f"章节文件数: {total_ch} | 通过: {chapters_ok} | 问题: {len(problems)}")
for p in problems:
    print("  -", p)

print("\n=== 全书 QC ===")
print(f"元信息块完整: {meta_ok}")
print(f"Part 分组: {len(parts)} 个 -> {parts}")
print(f"§编号连续 1..20: {contiguous} (实际: {nums})")
print(f"首个 H1 是标题: {first_h1}")
print(f"含'标叔的结论'列总数: {book.count('标叔的结论')}")
print(f"含'标叔的经验'框总数: {book.count('标叔的经验')}")
print(f"禁用词出现: {sum(book.count(b) for b in BANNED)}")

if not problems and not appx_problems and meta_ok and contiguous and first_h1:
    print("\nQC 结论：通过 ✅")
else:
    print("\nQC 结论：有问题，见上 ❌")
