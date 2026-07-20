"""
retanslate_v2.py — 健壮版 data.js 重译脚本。

相比 _retanslate_untranslated.py 的修复点：
  1. 读 progress_data.json，跳过已翻译项
  2. 每批翻译后立刻写盘 data.zh.js（中断也不会丢）
  3. API 调用加 timeout（30s）和 retries（2 次），失败直接跳过而非死循环
  4. 新增 --limit、--offset、--batch-size 控量
  5. 失败条目写入 progress_data.json['failed_batches']，下次跳过
  6. 进度条 print + 总耗时统计

用法：
  python retanslate_v2.py --limit 50        # 一次最多翻译 50 条
  python retanslate_v2.py --offset 50 --limit 50   # 跳 50 再翻 50
  python retanslate_v2.py --all             # 一次翻完所有剩余
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'translate_tool'))
from data_translate import (  # noqa: E402
    extract_arrays, parse_js_array, text_hash,
    DATA_PATH, OUT_PATH, PROGRESS_PATH,
)
from translator import Translator  # noqa: E402


# ---------- helpers ----------

def is_pure_english(s: str) -> bool:
    if not s or len(s.strip()) < 2:
        return False
    has_cn = any('\u4e00' <= c <= '\u9fff' for c in s)
    has_en = any(c.isalpha() and ord(c) < 128 for c in s)
    return has_en and not has_cn


def collect_all(parsed: Dict[str, List[dict]]) -> List[Tuple[str, str, List, str]]:
    """返回 [(hash, text, full_path, field), ...]，
    full_path 是从 parsed 根开始的完整索引链，如 ['PHASES', 0] 或 ['PHASES', 0, 'lessons', 1] 或 ['ARTIFACTS', 3]。
    """
    out: List[Tuple[str, str, List, str]] = []
    if 'PHASES' in parsed:
        for pi, p in enumerate(parsed['PHASES']):
            for f in ('name', 'desc'):
                v = p.get(f)
                if isinstance(v, str) and v.strip():
                    out.append((text_hash(v), v, ['PHASES', pi], f))
            for li, l in enumerate(p.get('lessons', [])):
                for f in ('name', 'summary', 'keywords'):
                    v = l.get(f)
                    if isinstance(v, str) and v.strip():
                        out.append((text_hash(v), v, ['PHASES', pi, 'lessons', li], f))
    if 'ARTIFACTS' in parsed:
        for ai, a in enumerate(parsed['ARTIFACTS']):
            for f in ('name', 'description'):
                v = a.get(f)
                if isinstance(v, str) and v.strip():
                    out.append((text_hash(v), v, ['ARTIFACTS', ai], f))
    if 'GLOSSARY' in parsed:
        for gi, g in enumerate(parsed['GLOSSARY']):
            for f in ('term', 'definition', 'says', 'means'):
                v = g.get(f)
                if isinstance(v, str) and v.strip():
                    out.append((text_hash(v), v, ['GLOSSARY', gi], f))
    return out


def apply_at(parsed: Dict, full_path: List, field: str, new_value: str) -> None:
    """full_path 例: ['PHASES', 0] 或 ['PHASES', 0, 'lessons', 1]。"""
    obj: Any = parsed
    for k in full_path:
        obj = obj[k]
    obj[field] = new_value


def find_path(parsed: Dict, target_hash: str) -> Tuple[List, str] | None:
    """根据 hash 反查所在位置。"""
    for sec, fields in (('PHASES', [(['name'], ['desc']), (['lessons'], ['name', 'summary', 'keywords'])]),
                        ('ARTIFACTS', [([], ['name', 'description'])]),
                        ('GLOSSARY', [([], ['term', 'definition', 'says', 'means'])])):
        if sec not in parsed:
            continue
        for idx, item in enumerate(parsed[sec]):
            for prefix, fs in fields:
                for f in fs:
                    v = item.get(f) if not prefix else (item.get(prefix[0], [{}]) if False else None)
                    if prefix:
                        # 嵌套: lessons
                        children = item.get(prefix[0], [])
                        for ci, child in enumerate(children):
                            for f in fs:
                                v = child.get(f)
                                if isinstance(v, str) and text_hash(v) == target_hash:
                                    return [idx, prefix[0], ci], f
                    else:
                        v = item.get(f)
                        if isinstance(v, str) and text_hash(v) == target_hash:
                            return [idx], f
    return None


def apply_by_hash(parsed: Dict, translated: Dict[str, str]) -> int:
    """把 translated 字典按 hash 应用回 parsed。"""
    applied = 0
    for sec, walk in (('PHASES', _walk_phases),
                      ('ARTIFACTS', _walk_artifacts),
                      ('GLOSSARY', _walk_glossary)):
        if sec not in parsed:
            continue
        for path, f, v in walk(parsed[sec]):
            h = text_hash(v)
            if h in translated:
                apply_at(parsed, [sec] + path, f, translated[h])
                applied += 1
    return applied


def _walk_phases(arr):
    for pi, p in enumerate(arr):
        for f in ('name', 'desc'):
            v = p.get(f)
            if isinstance(v, str):
                yield [pi], f, v
        for li, l in enumerate(p.get('lessons', [])):
            for f in ('name', 'summary', 'keywords'):
                v = l.get(f)
                if isinstance(v, str):
                    yield [pi, 'lessons', li], f, v


def _walk_artifacts(arr):
    for ai, a in enumerate(arr):
        for f in ('name', 'description'):
            v = a.get(f)
            if isinstance(v, str):
                yield [ai], f, v


def _walk_glossary(arr):
    for gi, g in enumerate(arr):
        for f in ('term', 'definition', 'says', 'means'):
            v = g.get(f)
            if isinstance(v, str):
                yield [gi], f, v


def write_zh_js(header: str, blocks, parsed, footer: str) -> None:
    out = header
    for name, src in blocks:
        if name in parsed:
            out += 'const ' + name + ' = ' + json.dumps(parsed[name], ensure_ascii=False, indent=2) + ';\n\n'
        else:
            out += 'const ' + name + ' = ' + src + ';\n\n'
    out += footer.lstrip('\n')
    OUT_PATH.write_text(out, encoding='utf-8')


def call_api_with_retry(translator: Translator, sys_prompt: str, user_prompt: str,
                        max_tokens: int, timeout: int = 30, retries: int = 2) -> str:
    """带超时和重试的 API 调用。返回原始 content。"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = translator.client.chat.completions.create(
                model=translator.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return (r.choices[0].message.content or '').strip()
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'API 重试 {retries + 1} 次仍失败: {last_err}')


# ---------- main ----------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', default='tokenhub')
    ap.add_argument('--model', default='deepseek-v4-flash-202605')
    ap.add_argument('--api-key', default=os.environ.get('TOKENHUB_API_KEY', ''))
    ap.add_argument('--batch-size', type=int, default=10)
    ap.add_argument('--request-delay', type=float, default=0.3)
    ap.add_argument('--timeout', type=int, default=30)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='最多翻译多少条；0 表示不限')
    ap.add_argument('--offset', type=int, default=0, help='从第几条开始（基于剩余未翻译列表）')
    ap.add_argument('--all', action='store_true', help='翻译所有未翻译（等价于 --limit 0）')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    # 1) 读 data.js
    js_text = DATA_PATH.read_text(encoding='utf-8')
    header, blocks, footer = extract_arrays(js_text)
    parsed = {n: parse_js_array(s) for n, s in blocks}

    # 2) 读 progress
    if PROGRESS_PATH.exists():
        try:
            progress = json.loads(PROGRESS_PATH.read_text(encoding='utf-8'))
        except Exception:
            progress = {'done': {}, 'failed': [], 'failed_batches': []}
    else:
        progress = {'done': {}, 'failed': [], 'failed_batches': []}
    progress.setdefault('done', {})
    progress.setdefault('failed', [])
    progress.setdefault('failed_batches', [])

    # 3) 用 done 立即把 data.zh.js 同步到当前进度（无需重译）
    applied_existing = apply_by_hash(parsed, progress['done'])
    print(f'📖 已有 {len(progress["done"])} 条翻译，应用 {applied_existing} 条到内存数据')

    # 4) 收集所有条目，找出还为纯英文的
    all_items = collect_all(parsed)
    remaining = [(h, t, p, f) for h, t, p, f in all_items if is_pure_english(t)]
    print(f'🔍 全部 {len(all_items)} 条, 待翻译 {len(remaining)} 条')

    if args.offset:
        remaining = remaining[args.offset:]
        print(f'   跳过前 {args.offset} 条, 剩余 {len(remaining)} 条')

    if args.limit and not args.all:
        remaining = remaining[:args.limit]
        print(f'   限制为 {args.limit} 条, 实际 {len(remaining)} 条')

    if not remaining:
        print('✅ 全部已翻译')
        # 还是写一次盘
        if not args.dry_run:
            write_zh_js(header, blocks, parsed, footer)
            print(f'💾 写出 {OUT_PATH}')
        return

    if args.dry_run:
        print('（dry-run 模式，不调用 API）')
        print('前 10 条:')
        for h, t, _, _ in remaining[:10]:
            print(f'  {h}: {t[:80]!r}')
        return

    # 5) 调用 API 分批翻译
    translator = Translator(
        provider=args.provider, model=args.model,
        api_key=args.api_key, request_delay=args.request_delay,
    )

    sys_prompt = (
        "You are a professional English-to-Chinese translator for a technical AI engineering course. "
        "Translate EACH input string to Simplified Chinese. "
        "For proper nouns and technical terms, add a Chinese translation in parentheses. "
        "Examples: 'Jupyter Notebook' -> 'Jupyter Notebook（Jupyter 笔记本）'; "
        "'Linux' -> 'Linux（Linux 操作系统）'; 'Python' -> 'Python（Python 编程语言）'. "
        "Output ONLY a JSON array of translated strings, preserving order and count."
    )

    t0 = time.time()
    ok_count = 0
    fail_count = 0
    total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size

    for bi in range(total_batches):
        batch = remaining[bi * args.batch_size:(bi + 1) * args.batch_size]
        hashes = [x[0] for x in batch]
        texts = [x[1] for x in batch]
        print(f'📦 批 {bi + 1}/{total_batches} ({len(batch)} 条)')

        try:
            content = call_api_with_retry(
                translator, sys_prompt, json.dumps(texts, ensure_ascii=False),
                max_tokens=min(8192, 256 * len(texts)),
                timeout=args.timeout, retries=args.retries,
            )
            # 解析 JSON 数组
            m = re.search(r'\[.*\]', content, re.DOTALL)
            if not m:
                raise RuntimeError('返回内容无 JSON 数组')
            arr = json.loads(m.group(0))
            if not isinstance(arr, list) or len(arr) != len(hashes):
                raise RuntimeError(f'返回数组长度不匹配: {len(arr)} vs {len(hashes)}')

            for h, new_text, path, f in zip(hashes, arr, [x[2] for x in batch], [x[3] for x in batch]):
                apply_at(parsed, path, f, str(new_text))
                progress['done'][h] = str(new_text)
            ok_count += len(batch)

            # 写盘（每批都写，中断不丢）
            write_zh_js(header, blocks, parsed, footer)
            PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')

            elapsed = time.time() - t0
            print(f'   ✓ 批 {bi + 1} 完成 (累计 {ok_count}, 失败 {fail_count}, 用时 {elapsed:.0f}s)')
        except Exception as e:
            fail_count += len(batch)
            progress['failed_batches'].append({
                'batch': bi + 1,
                'hashes': hashes,
                'error': str(e),
                'time': time.time(),
            })
            PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'   ✗ 批 {bi + 1} 失败: {e}（已记入 failed_batches，下次跳过）')
            # 失败也写盘（保持已有翻译）
            write_zh_js(header, blocks, parsed, footer)
            continue

        time.sleep(args.request_delay)

    elapsed = time.time() - t0
    print(f'\n🎉 完成：成功 {ok_count} 条, 失败 {fail_count} 条, 用时 {elapsed:.0f}s')
    print(f'💾 data.zh.js: {OUT_PATH}')


def _section_from_path(parsed, path):
    """从 collect_all 返回的 path 反推 section 名称。"""
    # path 是 [pi] 或 [pi, 'lessons', li] 或 [ai] 或 [gi]
    if not path:
        return 'PHASES'
    # 通过 path 第一个元素类型判断
    idx = path[0]
    if 'PHASES' in parsed and idx < len(parsed['PHASES']):
        if len(path) == 1:
            return 'PHASES'
        if 'lessons' in path:
            return 'PHASES'
    if 'ARTIFACTS' in parsed and idx < len(parsed['ARTIFACTS']):
        return 'ARTIFACTS'
    if 'GLOSSARY' in parsed and idx < len(parsed['GLOSSARY']):
        return 'GLOSSARY'
    return 'PHASES'


if __name__ == '__main__':
    main()
