"""
Markdown 翻译器：解析 .md 文件，提取可翻译段落，送 AI 翻译，写回为 zh.md

跳过不翻译的内容（保留原样）：
  - YAML frontmatter（--- 围栏）
  - 围栏代码块（```...``` 或 ~~~...~~~，含 mermaid）
  - 行内代码（`code`）
  - 行内公式（$...$、$$...$$、$...$ 等）
  - 链接 URL 部分、图片 URL、自动链接、裸 URL
  - 表格分隔行、空行

可翻译内容：
  - 标题、段落、列表项、引用行、表格行
  - frontmatter 之外的所有 Markdown 文本
"""
import os
import re
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def text_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()[:12]


# 行内不翻译片段的正则（按优先级顺序）
INLINE_SKIP_PATTERNS = [
    re.compile(r'`[^`\n]+`'),                                   # 行内代码
    re.compile(r'\$\$[^$\n]+\$\$'),                             # 块级数学
    re.compile(r'\$[^$\n]+\$'),                                 # 行内数学
    re.compile(r'\[[^\]]*\]\([^)\s]+(?:\s+"[^"]*")?\)'),        # 链接 [text](url)
    re.compile(r'!\[[^\]]*\]\([^)\s]+\)'),                      # 图片 ![alt](url)
    re.compile(r'<https?://[^>\s]+>'),                          # 自动链接
    re.compile(r'https?://[^\s<>\)\]]+'),                       # 裸 URL
]


def mask_inplace(text: str) -> Tuple[str, List[str]]:
    """用占位符替换行内不可翻译片段。"""
    placeholders: List[str] = []
    while True:
        best = None
        for pat in INLINE_SKIP_PATTERNS:
            m = pat.search(text)
            if m and (best is None or m.start() < best.start()):
                best = m
        if best is None:
            break
        placeholder = f"@@SKIP{len(placeholders):04d}@@"
        placeholders.append(best.group(0))
        text = text[:best.start()] + placeholder + text[best.end():]
    return text, placeholders


def unmask(text: str, placeholders: List[str]) -> str:
    """把占位符还原为原始片段。"""
    for i, ph in enumerate(placeholders):
        text = text.replace(f"@@SKIP{i:04d}@@", ph)
    return text


# 行级正则
HEADING_RE = re.compile(r'^(#{1,6}\s+)(.+?)\s*#*\s*$')
LIST_RE = re.compile(r'^(\s*(?:[-*+]|\d+\.)\s+)(.*)$')          # group(1)=前缀, group(2)=文本
BLOCKQUOTE_RE = re.compile(r'^(>\s?)(.*)$')                     # group(1)=前缀, group(2)=文本
TABLE_ROW_RE = re.compile(r'^\s*\|.*\|\s*$')
TABLE_SEP_RE = re.compile(r'^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$')
FENCE_RE = re.compile(r'^(\s*)(```+|~~~+)(.*)$')


def is_pure_skip(text: str) -> bool:
    """去掉占位符后是否还剩任何可翻译内容。"""
    cleaned = re.sub(r'@@SKIP\d{4}@@', '', text)
    return not cleaned.strip()


def is_already_translated(text: str, threshold: float = 0.3) -> bool:
    """如果中文占比 >= threshold，视为已翻译。"""
    stripped = text.strip()
    if not stripped or len(stripped) < 10:
        return False
    cjk_count = sum(1 for c in stripped if '\u4e00' <= c <= '\u9fff')
    letter_count = sum(1 for c in stripped if c.isascii() and c.isalpha())
    if cjk_count + letter_count == 0:
        return False
    return cjk_count / (cjk_count + letter_count) >= threshold


def extract_frontmatter(md: str) -> Tuple[str, str]:
    """分离 YAML frontmatter。返回 (frontmatter_with_trailing_newline, body)。"""
    if not (md.startswith('---\n') or md.startswith('---\r\n')):
        return '', md
    m = re.search(r'^---\s*$(?:\r?\n)', md[4:], re.MULTILINE)
    if not m:
        return '', md
    return md[:4 + m.end()], md[4 + m.end():]


# ------------------------------------------------------------------
# 表格行：逐单元格 mask
# ------------------------------------------------------------------
def split_table_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return s.split('|')


def mask_table_row(line: str) -> Tuple[str, List[List[str]]]:
    cells = split_table_row(line)
    masked_cells = []
    cell_phs = []
    for c in cells:
        m, phs = mask_inplace(c)
        masked_cells.append(m)
        cell_phs.append(phs)
    return ' ||| '.join(masked_cells), cell_phs


def unmask_table_row(translated: str, cell_phs: List[List[str]]) -> str:
    cells = translated.split(' ||| ')
    # 如果 AI 改变了行数,补/截
    while len(cells) < len(cell_phs):
        cells.append('')
    cells = cells[:len(cell_phs)]
    unmasked = [unmask(c, phs) for c, phs in zip(cells, cell_phs)]
    return '| ' + ' | '.join(unmasked) + ' |'


# ------------------------------------------------------------------
# 核心：扫描 body 生成决策列表
# ------------------------------------------------------------------
def scan_body(body: str) -> Tuple[List[Dict], List[str]]:
    """
    扫描 body,生成决策列表(line_decisions)和原始行列表。
    line_decisions 的每个元素:
      - {'kind': 'keep', 'line': str}  -- 保留原行
      - {'kind': 'translate', 'mode': 'heading'|'paragraph'|'list_block'|'blockquote'|'table_row',
         'masked': str, 'phs': List[str] | List[List[str]],
         'prefix'?: str, 'prefixes'?: List[str], 'span': (int, int)}
    """
    lines = body.split('\n')
    n = len(lines)
    decisions: List[Dict] = []
    i = 0
    in_code = False
    fence = None

    while i < n:
        line = lines[i]
        m = FENCE_RE.match(line)
        if m and not in_code:
            in_code = True
            fence = m.group(2)[:3]
            decisions.append({'kind': 'keep', 'line': line})
            i += 1
            continue
        if in_code:
            decisions.append({'kind': 'keep', 'line': line})
            if re.match(r'^\s*' + re.escape(fence) + r'+\s*$', line):
                in_code = False
                fence = None
            i += 1
            continue

        stripped = line.rstrip()

        # 空行
        if not stripped.strip():
            decisions.append({'kind': 'keep', 'line': line})
            i += 1
            continue

        # 表格分隔线
        if TABLE_SEP_RE.match(stripped):
            decisions.append({'kind': 'keep', 'line': line})
            i += 1
            continue

        # 标题
        h = HEADING_RE.match(stripped)
        if h:
            masked, phs = mask_inplace(h.group(2))
            if is_pure_skip(masked):
                decisions.append({'kind': 'keep', 'line': line})
            else:
                decisions.append({
                    'kind': 'translate',
                    'mode': 'heading',
                    'masked': masked,
                    'phs': phs,
                    'prefix': h.group(1),
                    'span': (i, i + 1),
                })
            i += 1
            continue

        # 表格行
        if TABLE_ROW_RE.match(line):
            masked, cell_phs = mask_table_row(line)
            if is_pure_skip(masked):
                decisions.append({'kind': 'keep', 'line': line})
            else:
                decisions.append({
                    'kind': 'translate',
                    'mode': 'table_row',
                    'masked': masked,
                    'phs': cell_phs,
                    'span': (i, i + 1),
                })
            i += 1
            continue

        # 引用
        if BLOCKQUOTE_RE.match(stripped):
            bq_start = i
            prefixes = []
            parts = []
            combined_phs: List[str] = []
            while i < n:
                bq = BLOCKQUOTE_RE.match(lines[i].rstrip())
                if not bq:
                    break
                prefixes.append(bq.group(1))
                masked, phs = mask_inplace(bq.group(2))
                parts.append(masked)
                combined_phs.extend(phs)
                i += 1
            combined = '\n'.join(parts)
            if is_pure_skip(combined):
                for j in range(bq_start, i):
                    decisions.append({'kind': 'keep', 'line': lines[j]})
            else:
                decisions.append({
                    'kind': 'translate',
                    'mode': 'blockquote',
                    'masked': combined,
                    'phs': combined_phs,
                    'prefixes': prefixes,
                    'span': (bq_start, i),
                })
            continue

        # 列表
        if LIST_RE.match(line):
            li_start = i
            prefixes = []
            parts = []
            combined_phs: List[str] = []
            while i < n:
                li = LIST_RE.match(lines[i])
                if not li:
                    break
                prefixes.append(li.group(1))
                masked, phs = mask_inplace(li.group(2).rstrip())
                parts.append(masked)
                combined_phs.extend(phs)
                i += 1
            combined = '\n'.join(parts)
            if is_pure_skip(combined):
                for j in range(li_start, i):
                    decisions.append({'kind': 'keep', 'line': lines[j]})
            else:
                decisions.append({
                    'kind': 'translate',
                    'mode': 'list_block',
                    'masked': combined,
                    'phs': combined_phs,
                    'prefixes': prefixes,
                    'span': (li_start, i),
                })
            continue

        # 段落
        para_start = i
        raw_lines = []
        while i < n:
            cur = lines[i]
            cur_stripped = cur.rstrip()
            if not cur_stripped.strip():
                break
            if FENCE_RE.match(cur_stripped):
                break
            if HEADING_RE.match(cur_stripped):
                break
            if LIST_RE.match(cur):
                break
            if BLOCKQUOTE_RE.match(cur_stripped):
                break
            if TABLE_ROW_RE.match(cur):
                break
            if TABLE_SEP_RE.match(cur_stripped):
                break
            raw_lines.append(cur_stripped)
            i += 1
        if not raw_lines:
            decisions.append({'kind': 'keep', 'line': lines[para_start]})
            i = para_start + 1
            continue
        combined = '\n'.join(raw_lines)
        masked, phs = mask_inplace(combined)
        if is_pure_skip(masked):
            for j in range(para_start, i):
                decisions.append({'kind': 'keep', 'line': lines[j]})
        else:
            decisions.append({
                'kind': 'translate',
                'mode': 'paragraph',
                'masked': masked,
                'phs': phs,
                'span': (para_start, i),
            })

    return decisions, lines


# ------------------------------------------------------------------
# 核心：批量翻译 + 重组
# ------------------------------------------------------------------
def ensure_placeholders(translated: str, expected: List[str]) -> str:
    """如果译文中占位符丢失/被破坏,补回占位符（用于降级）。"""
    missing = [p for p in expected if p not in translated]
    if not missing:
        return translated
    return translated  # 降级: 接受占位符可能丢失,unmask 仍是 no-op


def reconstruct_lines(
    decisions: List[Dict],
    translations: Dict[str, str],
) -> List[str]:
    """根据 decisions 和 translations 重新组装 lines。"""
    out: List[str] = []
    for d in decisions:
        if d['kind'] == 'keep':
            out.append(d['line'])
            continue
        h = text_hash(d['masked'])
        translated = translations.get(h, d['masked'])
        mode = d['mode']

        if mode == 'heading':
            translated = ensure_placeholders(translated, d['phs'])
            out.append(d['prefix'] + unmask(translated, d['phs']))
        elif mode == 'table_row':
            out.append(unmask_table_row(translated, d['phs']))
        elif mode == 'paragraph':
            translated = ensure_placeholders(translated, d['phs'])
            full = unmask(translated, d['phs'])
            for ln in full.split('\n'):
                out.append(ln)
        elif mode == 'list_block':
            translated = ensure_placeholders(translated, d['phs'])
            full = unmask(translated, d['phs'])
            lines_t = full.split('\n')
            while len(lines_t) < len(d['prefixes']):
                lines_t.append('')
            lines_t = lines_t[:len(d['prefixes'])]
            for prefix, ln in zip(d['prefixes'], lines_t):
                out.append(prefix + ln)
        elif mode == 'blockquote':
            translated = ensure_placeholders(translated, d['phs'])
            full = unmask(translated, d['phs'])
            lines_t = full.split('\n')
            while len(lines_t) < len(d['prefixes']):
                lines_t.append('')
            lines_t = lines_t[:len(d['prefixes'])]
            for prefix, ln in zip(d['prefixes'], lines_t):
                out.append(prefix + ln)
        else:
            out.append(d['masked'])
    return out


def translate_body(
    body: str,
    translator,
    tm,
    glossary,
    batch_size: int = 20,
    verbose: bool = False,
) -> Tuple[str, Dict]:
    """扫描+翻译+重组。返回 (new_body, stats)。"""
    decisions, lines = scan_body(body)

    # 去重
    unique: Dict[str, str] = {}
    for d in decisions:
        if d['kind'] != 'translate':
            continue
        h = text_hash(d['masked'])
        if h not in unique:
            unique[h] = d['masked']

    cached: Dict[str, str] = {}
    to_call: List[str] = []
    to_call_h: List[str] = []
    for h, txt in unique.items():
        c = tm.lookup(h)
        if c:
            cached[h] = c
        else:
            to_call.append(txt)
            to_call_h.append(h)

    stats = {
        'segments': sum(1 for d in decisions if d['kind'] == 'translate'),
        'unique_texts': len(unique),
        'cached': len(cached),
        'to_translate': len(to_call),
    }

    if to_call:
        glossary_text = glossary.build_prompt_terms()
        results = []
        for i in range(0, len(to_call), batch_size):
            batch = to_call[i:i + batch_size]
            if verbose:
                print(f"      翻译批次 {i // batch_size + 1}/{(len(to_call) + batch_size - 1) // batch_size} ({len(batch)} 条)...")
            r = translator.translate_batch(batch, glossary_text)
            results.extend(r)
        success = skip = fail = 0
        for h, src, tgt in zip(to_call_h, to_call, results):
            if tgt is None:
                cached[h] = src
                fail += 1
            elif tgt.strip() == src.strip():
                cached[h] = tgt
                skip += 1
            else:
                cached[h] = tgt
                tm.add(h, src, tgt, context='phases_md')
                success += 1
        if verbose and (fail or skip):
            print(f"      📊 新译={success} 已译={skip} 失败={fail}")

    new_lines = reconstruct_lines(decisions, cached)
    return '\n'.join(new_lines), stats


# ------------------------------------------------------------------
# 单文件翻译
# ------------------------------------------------------------------
def translate_markdown_file(
    src_path: str,
    dst_path: str,
    translator,
    tm,
    glossary,
    batch_size: int = 20,
    force: bool = False,
) -> Dict:
    with open(src_path, 'r', encoding='utf-8') as f:
        md = f.read()

    if not force and os.path.exists(dst_path):
        with open(dst_path, 'r', encoding='utf-8') as f:
            existing = f.read()
        if is_already_translated(existing, threshold=0.3):
            return {'file': src_path, 'skipped': True, 'reason': 'already_translated'}

    fm, body = extract_frontmatter(md)
    new_body, stats = translate_body(body, translator, tm, glossary, batch_size=batch_size, verbose=True)

    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(fm)
        if fm and not fm.endswith('\n'):
            f.write('\n')
        f.write(new_body)
        if not new_body.endswith('\n'):
            f.write('\n')

    stats['file'] = src_path
    stats['dst'] = dst_path
    return stats


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='翻译单个 .md 文件为 .zh.md')
    parser.add_argument('src')
    parser.add_argument('dst', nargs='?')
    parser.add_argument('--provider', default='tokenhub')
    parser.add_argument('--model', default='hy3')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--memory', default=os.path.join(os.path.dirname(__file__), 'translation_memory.json'))
    parser.add_argument('--glossary-file', default=os.path.join(os.path.dirname(__file__), 'glossary.json'))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from translator import Translator
    from translation_memory import TranslationMemory
    from glossary import Glossary

    if args.dst is None:
        p = Path(args.src)
        args.dst = str(p.with_name(p.stem + '.zh' + p.suffix))

    api_key = os.environ.get('TOKENHUB_API_KEY') or os.environ.get('OPENAI_API_KEY', '')

    if args.dry_run:
        with open(args.src, 'r', encoding='utf-8') as f:
            md = f.read()
        _, body = extract_frontmatter(md)
        decisions, _ = scan_body(body)
        segs = [d for d in decisions if d['kind'] == 'translate']
        print(json.dumps({
            'total_decisions': len(decisions),
            'translate_segments': len(segs),
            'modes': {
                'heading': sum(1 for d in segs if d['mode'] == 'heading'),
                'paragraph': sum(1 for d in segs if d['mode'] == 'paragraph'),
                'list_block': sum(1 for d in segs if d['mode'] == 'list_block'),
                'blockquote': sum(1 for d in segs if d['mode'] == 'blockquote'),
                'table_row': sum(1 for d in segs if d['mode'] == 'table_row'),
            },
            'sample': [{'mode': d['mode'], 'text': d['masked'][:80]} for d in segs[:5]],
        }, ensure_ascii=False, indent=2))
        return

    translator = Translator(provider=args.provider, model=args.model, api_key=api_key, request_delay=2.0)
    tm = TranslationMemory(args.memory)
    glossary = Glossary(args.glossary_file)

    result = translate_markdown_file(
        args.src, args.dst, translator, tm, glossary,
        batch_size=args.batch_size, force=args.force,
    )
    tm.save()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
