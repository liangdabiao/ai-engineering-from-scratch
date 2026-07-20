"""
apply_progress.py — 把 progress_data.json 里的"done"应用回 data.zh.js。

为什么需要：
  data_translate.py 在跑完所有批次前从不写 data.zh.js。如果脚本被中断、
  崩溃或被 Ctrl+C，所有翻译结果都躺在 progress_data.json 里，data.zh.js
  仍然是英文。这个脚本只读 progress + data.js，立刻把已翻译的部分落盘。

用法：
  python apply_progress.py            # 应用进度并写盘
  python apply_progress.py --dry-run  # 只统计，不写盘
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'translate_tool'))
from data_translate import (  # noqa: E402
    extract_arrays, parse_js_array, text_hash, DATA_PATH, OUT_PATH, PROGRESS_PATH,
)


def load_progress() -> Dict[str, Any]:
    if not PROGRESS_PATH.exists():
        return {'done': {}, 'failed': []}
    try:
        return json.loads(PROGRESS_PATH.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'⚠️  进度文件损坏: {e}')
        return {'done': {}, 'failed': []}


def apply_translations(parsed: Dict[str, List[dict]], translated: Dict[str, str]) -> int:
    """把 translated 字典按 hash 套用到 parsed 里。返回成功替换数。"""
    applied = 0
    if 'PHASES' in parsed:
        for p in parsed['PHASES']:
            for f in ('name', 'desc'):
                v = p.get(f)
                if isinstance(v, str) and text_hash(v) in translated:
                    p[f] = translated[text_hash(v)]
                    applied += 1
            for l in p.get('lessons', []):
                for f in ('name', 'summary', 'keywords'):
                    v = l.get(f)
                    if isinstance(v, str) and text_hash(v) in translated:
                        l[f] = translated[text_hash(v)]
                        applied += 1
    if 'ARTIFACTS' in parsed:
        for a in parsed['ARTIFACTS']:
            for f in ('name', 'description'):
                v = a.get(f)
                if isinstance(v, str) and text_hash(v) in translated:
                    a[f] = translated[text_hash(v)]
                    applied += 1
    if 'GLOSSARY' in parsed:
        for g in parsed['GLOSSARY']:
            for f in ('term', 'definition', 'says', 'means'):
                v = g.get(f)
                if isinstance(v, str) and text_hash(v) in translated:
                    g[f] = translated[text_hash(v)]
                    applied += 1
    return applied


def to_js(value, indent=2) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    progress = load_progress()
    done = progress.get('done', {})
    print(f'📖 进度文件: {len(done)} 条已翻译, {len(progress.get("failed", []))} 条失败')

    if not DATA_PATH.exists():
        print(f'❌ 找不到源文件: {DATA_PATH}')
        sys.exit(1)
    js_text = DATA_PATH.read_text(encoding='utf-8')
    header, blocks, footer = extract_arrays(js_text)
    parsed = {n: parse_js_array(s) for n, s in blocks}

    # 统计所有可翻译条目
    total = 0
    untranslated_after: List[str] = []
    for sec, fields in (('PHASES', [('name', 'desc'), ('lessons', ['name', 'summary', 'keywords'])]),
                        ('ARTIFACTS', [('name', 'description')]),
                        ('GLOSSARY', [('term', 'definition', 'says', 'means')])):
        if sec not in parsed:
            continue
        if sec == 'PHASES':
            for p in parsed[sec]:
                for f in ('name', 'desc'):
                    v = p.get(f)
                    if isinstance(v, str) and v.strip():
                        total += 1
                        if text_hash(v) not in done:
                            untranslated_after.append(text_hash(v))
                for l in p.get('lessons', []):
                    for f in ('name', 'summary', 'keywords'):
                        v = l.get(f)
                        if isinstance(v, str) and v.strip():
                            total += 1
                            if text_hash(v) not in done:
                                untranslated_after.append(text_hash(v))
        elif sec == 'ARTIFACTS':
            for a in parsed[sec]:
                for f in ('name', 'description'):
                    v = a.get(f)
                    if isinstance(v, str) and v.strip():
                        total += 1
                        if text_hash(v) not in done:
                            untranslated_after.append(text_hash(v))
        elif sec == 'GLOSSARY':
            for g in parsed[sec]:
                for f in ('term', 'definition', 'says', 'means'):
                    v = g.get(f)
                    if isinstance(v, str) and v.strip():
                        total += 1
                        if text_hash(v) not in done:
                            untranslated_after.append(text_hash(v))

    print(f'🔍 全部 {total} 条, 待翻译 {len(untranslated_after)} 条')
    if args.dry_run:
        print('（dry-run 模式，不写盘）')
        return

    # 应用翻译
    applied = apply_translations(parsed, done)
    print(f'✅ 已应用 {applied} 条翻译到数据结构')

    # 重写 data.zh.js
    out = header
    for name, src in blocks:
        if name in parsed:
            out += 'const ' + name + ' = ' + to_js(parsed[name]) + ';\n\n'
        else:
            out += 'const ' + name + ' = ' + src + ';\n\n'
    out += footer.lstrip('\n')

    OUT_PATH.write_text(out, encoding='utf-8')
    print(f'💾 写出 {OUT_PATH} ({len(out)} bytes)')
    print(f'📊 完成度: {(total - len(untranslated_after)) * 100 // total}% '
          f'({total - len(untranslated_after)}/{total})')


if __name__ == '__main__':
    main()
