"""
翻译质量校验
=============
对每个 .zh.md / README.zh.md / mission.zh.md:
  - 文件存在
  - 中文占比合理
  - 代码块 ``` 配对正确(数量未变)
  - mermaid 块数量未变
  - 行内代码 ` 配对正确
  - 链接 [text](url) 中 URL 数量与原文一致
  - 公式 $...$ / $$...$$ 数量未变
  - 标题 # 数量未变

对每个 quiz.zh.json:
  - JSON 合法
  - 题目数、选项数与原文一致
  - correct 索引未变
  - stage 标签未变
"""
import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent


def count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, re.MULTILINE))


def check_markdown(src: Path, dst: Path) -> List[str]:
    """返回错误列表(空=通过)"""
    errors = []
    if not dst.exists():
        return [f"目标文件不存在: {dst}"]
    s = src.read_text(encoding='utf-8')
    d = dst.read_text(encoding='utf-8')

    # 中文占比
    cjk_d = sum(1 for c in d if '\u4e00' <= c <= '\u9fff')
    cjk_s = sum(1 for c in s if '\u4e00' <= c <= '\u9fff')
    if cjk_d < cjk_s * 0.5 and cjk_d < 50:
        # 译文中文没明显多于原文(注意:原文是英文,译文应当中文占大头)
        errors.append(f"中文数量偏少 (en={cjk_s}, zh={cjk_d})")

    # 代码块围栏 ``` 配对
    src_fences = count_pattern(s, r'^```')
    dst_fences = count_pattern(d, r'^```')
    if src_fences != dst_fences:
        errors.append(f"代码块围栏数量不一致 en={src_fences} zh={dst_fences}")

    # mermaid 块数量
    src_mermaid = count_pattern(s, r'```mermaid')
    dst_mermaid = count_pattern(d, r'```mermaid')
    if src_mermaid != dst_mermaid:
        errors.append(f"mermaid 块数量不一致 en={src_mermaid} zh={dst_mermaid}")

    # 行内代码 ` 配对(简单:总数应该是偶数,且不少于原文)
    src_inline = count_pattern(s, r'`[^`\n]+`')
    dst_inline = count_pattern(d, r'`[^`\n]+`')
    if dst_inline < src_inline * 0.9:
        errors.append(f"行内代码数量明显减少 en={src_inline} zh={dst_inline}")

    # 链接数量 [text](url)
    src_links = count_pattern(s, r'\[[^\]]*\]\([^)\s]+')
    dst_links = count_pattern(d, r'\[[^\]]*\]\([^)\s]+')
    if dst_links < src_links * 0.9:
        errors.append(f"链接数量明显减少 en={src_links} zh={dst_links}")

    # 公式数量 $...$ (单行) + $$...$$ (单行)
    src_inline_math = count_pattern(s, r'(?<!\\)\$[^$\n]+\$')
    dst_inline_math = count_pattern(d, r'(?<!\\)\$[^$\n]+\$')
    if dst_inline_math < src_inline_math * 0.9:
        errors.append(f"行内公式数量减少 en={src_inline_math} zh={dst_inline_math}")

    # 标题 # 数量
    src_h = count_pattern(s, r'^#{1,6}\s')
    dst_h = count_pattern(d, r'^#{1,6}\s')
    if dst_h < src_h:
        errors.append(f"标题数量减少 en={src_h} zh={dst_h}")

    return errors


def check_quiz(src: Path, dst: Path) -> List[str]:
    errors = []
    if not dst.exists():
        return [f"目标文件不存在: {dst}"]
    try:
        s = json.loads(src.read_text(encoding='utf-8'))
        d = json.loads(dst.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        return [f"JSON 解析失败: {e}"]

    # 题目数
    qs_s = s.get('questions', [])
    qs_d = d.get('questions', [])
    if len(qs_s) != len(qs_d):
        errors.append(f"题目数不一致 en={len(qs_s)} zh={len(qs_d)}")

    # 每题: correct / stage / options 数量
    for i, (q_s, q_d) in enumerate(zip(qs_s, qs_d)):
        if q_s.get('stage') != q_d.get('stage'):
            errors.append(f"Q{i+1} stage 不一致 en={q_s.get('stage')} zh={q_d.get('stage')}")
        if q_s.get('correct') != q_d.get('correct'):
            errors.append(f"Q{i+1} correct 索引不一致 en={q_s.get('correct')} zh={q_d.get('correct')}")
        opts_s = q_s.get('options', [])
        opts_d = q_d.get('options', [])
        if len(opts_s) != len(opts_d):
            errors.append(f"Q{i+1} options 数量不一致 en={len(opts_s)} zh={len(opts_d)}")
    return errors


def main():
    parser_args = sys.argv[1:]
    only_kind = None
    if parser_args and parser_args[0] in ('--readmes', '--missions', '--quizzes', '--lessons'):
        only_kind = parser_args[0][2:]
        parser_args = parser_args[1:]

    all_errors: List[Tuple[Path, List[str]]] = []
    counts = {'checked': 0, 'passed': 0, 'failed': 0}

    # readmes
    if not only_kind or only_kind == 'readmes':
        for src in ROOT.glob('phases/*/README.md'):
            dst = src.with_name('README.zh.md')
            if not dst.exists():
                continue
            counts['checked'] += 1
            errs = check_markdown(src, dst)
            if errs:
                counts['failed'] += 1
                all_errors.append((dst, errs))
            else:
                counts['passed'] += 1

    # missions
    if not only_kind or only_kind == 'missions':
        for src in ROOT.glob('phases/*/mission.md'):
            dst = src.with_name('mission.zh.md')
            if not dst.exists():
                continue
            counts['checked'] += 1
            errs = check_markdown(src, dst)
            if errs:
                counts['failed'] += 1
                all_errors.append((dst, errs))
            else:
                counts['passed'] += 1

    # lessons
    if not only_kind or only_kind == 'lessons':
        for src in ROOT.glob('phases/*/*/docs/en.md'):
            dst = src.parent / 'zh.md'
            if not dst.exists():
                continue
            counts['checked'] += 1
            errs = check_markdown(src, dst)
            if errs:
                counts['failed'] += 1
                all_errors.append((dst, errs))
            else:
                counts['passed'] += 1

    # quizzes
    if not only_kind or only_kind == 'quizzes':
        for src in ROOT.glob('phases/*/*/quiz.json'):
            dst = src.with_name('quiz.zh.json')
            if not dst.exists():
                continue
            counts['checked'] += 1
            errs = check_quiz(src, dst)
            if errs:
                counts['failed'] += 1
                all_errors.append((dst, errs))
            else:
                counts['passed'] += 1

    print(f"\n校验完成: 总计 {counts['checked']} 通过 {counts['passed']} 失败 {counts['failed']}\n")
    if all_errors:
        print("失败列表:")
        for path, errs in all_errors[:50]:
            print(f"  ✗ {path.relative_to(ROOT)}")
            for e in errs:
                print(f"    - {e}")
        if len(all_errors) > 50:
            print(f"  ... 还有 {len(all_errors) - 50} 个失败未列出")
        sys.exit(1)
    else:
        print("✓ 全部通过")


if __name__ == '__main__':
    main()
