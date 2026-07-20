"""
data.js 翻译器：把 site/data.js 翻译为 site/data.zh.js
data.js 是自动生成的 JS 文件，包含 PHASES / SKILLS / GLOSSARY 三个数组。
本脚本读取 data.js，提取需要翻译的文本字段，调用翻译 API，
然后生成中文版本的 data.zh.js。

可翻译字段白名单：
  PHASES[i].name, PHASES[i].desc
  PHASES[i].lessons[j].name, .summary, .keywords
  SKILLS[i].name, .description
  GLOSSARY[i].term, .definition
"""
import os
import re
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'site' / 'data.js'
OUT_PATH = ROOT / 'site' / 'data.zh.js'
PROGRESS_PATH = ROOT / 'translate_tool' / 'progress_data.json'

# 添加 translate_tool 到 sys.path
sys.path.insert(0, str(ROOT / 'translate_tool'))
from translator import Translator  # type: ignore


def text_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()[:12]


def extract_arrays(js_text: str) -> Tuple[str, List[Tuple[str, str]], str]:
    """从 data.js 文本中提取 const NAME = [...] 块。
    返回 (header, [(name, source), ...], footer)
    """
    def find_block(name: str) -> str:
        pattern = rf'const\s+{name}\s*=\s*\['
        m = re.search(pattern, js_text)
        if not m:
            raise ValueError(f"can't find {name} in data.js")
        start = m.end() - 1  # '[' 的位置
        i = start
        depth = 0
        in_str = None
        escape = False
        while i < len(js_text):
            ch = js_text[i]
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif in_str:
                if ch == in_str:
                    in_str = None
            elif ch in ('"', "'"):
                in_str = ch
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return js_text[start:i + 1]
            i += 1
        raise ValueError(f"unterminated {name}")

    first_match = re.search(r'const\s+\w+\s*=\s*\[', js_text)
    header = js_text[:first_match.start()]
    last_end = js_text.rfind('];')
    footer = js_text[last_end + 2:]
    blocks: List[Tuple[str, str]] = []
    for name in ('PHASES', 'GLOSSARY', 'ARTIFACTS'):
        try:
            src = find_block(name)
            blocks.append((name, src))
        except ValueError:
            pass
    return header, blocks, footer


def parse_js_array(src: str) -> List[Dict]:
    """把 JS 数组字面量解析为 Python list。data.js 的数组是合法 JSON-ish 格式。
    使用更宽松的解析：只把 "key": "value" 和数组视作有效。
    """
    # 直接尝试 json.loads：data.js 的数组实际上就是合法 JSON
    try:
        return json.loads(src)
    except json.JSONDecodeError as e:
        # 退化方案：使用正则替换 JS-only 语法
        raise RuntimeError(f"无法解析 JS 数组: {e}")


def collect_translatable(phases: List[Dict], skills: List[Dict], glossary: List[Dict]) -> Dict[str, str]:
    """收集所有需要翻译的文本，返回 {hash: text}。"""
    items: Dict[str, str] = {}

    # PHASES
    for p in phases:
        for k in ('name', 'desc'):
            v = p.get(k)
            if v and isinstance(v, str) and v.strip():
                items[text_hash(v)] = v
        for lesson in p.get('lessons', []):
            for k in ('name', 'summary', 'keywords'):
                v = lesson.get(k)
                if v and isinstance(v, str) and v.strip():
                    items[text_hash(v)] = v

    # SKILLS
    for s in skills:
        for k in ('name', 'description'):
            v = s.get(k)
            if v and isinstance(v, str) and v.strip():
                items[text_hash(v)] = v

    # GLOSSARY
    for g in glossary:
        for k in ('term', 'definition'):
            v = g.get(k)
            if v and isinstance(v, str) and v.strip():
                items[text_hash(v)] = v

    return items


def chunked(items: Dict[str, str], batch_size: int) -> List[Dict[str, str]]:
    keys = list(items.keys())
    out = []
    for i in range(0, len(keys), batch_size):
        out.append({k: items[k] for k in keys[i:i + batch_size]})
    return out


def translate_batch(translator: Translator, batch: Dict[str, str]) -> Dict[str, str]:
    """翻译一个 batch，返回 {hash: translated_text}。"""
    hashes = list(batch.keys())
    inputs = [batch[h] for h in hashes]

    # 构建 prompt
    sys_prompt = (
        "You are a professional English-to-Chinese translator for a technical AI engineering course. "
        "Translate each item in the input JSON to Simplified Chinese. "
        "Preserve technical terms, code, and proper nouns. "
        "Keep the same JSON array structure and length. "
        "Output ONLY the JSON array of translated strings."
    )
    user_prompt = json.dumps(inputs, ensure_ascii=False)

    # 优先尝试 JSON 模式
    result: List[str] = []
    try:
        r = translator.client.chat.completions.create(
            model=translator.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=8192,
            response_format={"type": "json_object"} if "deepseek" not in translator.model.lower() else None,
        )
        content = r.choices[0].message.content or ''
        # 尝试解析 JSON
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # 可能是 {"translations": [...]} 或 {hash: text}
                for k in ('translations', 'items', 'results'):
                    if k in data and isinstance(data[k], list):
                        result = [str(x) for x in data[k]]
                        break
                else:
                    # 取第一个 list
                    for v in data.values():
                        if isinstance(v, list):
                            result = [str(x) for x in v]
                            break
            elif isinstance(data, list):
                result = [str(x) for x in data]
        except json.JSONDecodeError:
            # 尝试从纯文本中提取
            content_clean = re.sub(r'```json\s*', '', content)
            content_clean = re.sub(r'```\s*', '', content_clean)
            data = json.loads(content_clean)
            if isinstance(data, list):
                result = [str(x) for x in data]
    except Exception as e:
        # fallback: 重新尝试不带 json_object
        r = translator.client.chat.completions.create(
            model=translator.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": sys_prompt + ' Output JSON array directly.'},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=8192,
        )
        content = r.choices[0].message.content or ''
        # 提取 [...]
        m = re.search(r'\[.*\]', content, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    result = [str(x) for x in data]
            except json.JSONDecodeError:
                # 行分隔 fallback
                result = [line.strip().strip('"').strip("'") for line in content.split('\n') if line.strip()]

    # 数量校验
    if len(result) != len(hashes):
        # fallback：逐条翻译
        result = []
        for h in hashes:
            try:
                r = translator.client.chat.completions.create(
                    model=translator.model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "Translate the following English text to Simplified Chinese. Output ONLY the translation."},
                        {"role": "user", "content": batch[h]},
                    ],
                    max_tokens=1024,
                )
                result.append(r.choices[0].message.content.strip())
            except Exception as ex:
                result.append(batch[h])  # 失败保留原文

    return {h: result[i] if i < len(result) else batch[h] for i, h in enumerate(hashes)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', default='tokenhub')
    parser.add_argument('--model', default='deepseek-v4-flash-202605')
    parser.add_argument('--api-key', default=os.environ.get('TOKENHUB_API_KEY', ''))
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--request-delay', type=float, default=0.5)
    parser.add_argument('--only', default='', help='comma-separated list: phases,glossary,artifacts')
    args = parser.parse_args()

    print(f'📖 读取 {DATA_PATH}')
    js_text = DATA_PATH.read_text(encoding='utf-8')
    header, blocks, footer = extract_arrays(js_text)
    parsed = {}
    for name, src in blocks:
        parsed[name] = parse_js_array(src)
        print(f'   {name}: {len(src)} bytes, {len(parsed[name])} entries')

    only = set(s.strip() for s in args.only.split(',') if s.strip())
    phases = parsed.get('PHASES') if (not only or 'phases' in only) else None
    glossary = parsed.get('GLOSSARY') if (not only or 'glossary' in only) else None
    artifacts = parsed.get('ARTIFACTS') if (not only or 'artifacts' in only) else None

    # 收集翻译目标
    items: Dict[str, str] = {}
    if phases is not None:
        for p in phases:
            for k in ('name', 'desc'):
                v = p.get(k)
                if v and isinstance(v, str) and v.strip():
                    items[text_hash(v)] = v
            for lesson in p.get('lessons', []):
                for k in ('name', 'summary', 'keywords'):
                    v = lesson.get(k)
                    if v and isinstance(v, str) and v.strip():
                        items[text_hash(v)] = v
    if artifacts is not None:
        for a in artifacts:
            for k in ('name', 'description'):
                v = a.get(k)
                if v and isinstance(v, str) and v.strip():
                    items[text_hash(v)] = v
    if glossary is not None:
        for g in glossary:
            for k in ('term', 'definition'):
                v = g.get(k)
                if v and isinstance(v, str) and v.strip():
                    items[text_hash(v)] = v

    print(f'🔍 收集到 {len(items)} 条待翻译文本')

    # 加载进度
    if PROGRESS_PATH.exists():
        try:
            progress = json.loads(PROGRESS_PATH.read_text(encoding='utf-8'))
        except Exception:
            progress = {'done': {}, 'failed': []}
    else:
        progress = {'done': {}, 'failed': []}

    todo = {h: t for h, t in items.items() if h not in progress['done']}
    print(f'   已翻译 {len(progress["done"])} 条, 待翻译 {len(todo)} 条')

    if todo:
        translator = Translator(
            provider=args.provider, model=args.model,
            api_key=args.api_key, request_delay=args.request_delay,
        )
        batches = chunked(todo, args.batch_size)
        print(f'🔄 准备调用 API：{len(batches)} 批, 每批 {args.batch_size} 条')

        for bi, batch in enumerate(batches):
            try:
                translated = translate_batch(translator, batch)
                progress['done'].update(translated)
                if (bi + 1) % 5 == 0 or bi == len(batches) - 1:
                    PROGRESS_PATH.write_text(
                        json.dumps(progress, ensure_ascii=False, indent=2),
                        encoding='utf-8',
                    )
                done = len(progress['done'])
                total = len(items)
                print(f'   批 {bi + 1}/{len(batches)} ✓ 累计 {done}/{total} '
                      f'({done * 100 // total}%)')
            except Exception as e:
                print(f'   批 {bi + 1} 失败: {e}')
                progress['failed'].extend(list(batch.keys()))
                PROGRESS_PATH.write_text(
                    json.dumps(progress, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                )

    # 把翻译结果应用回数据结构
    def apply(translated_map: Dict[str, str], target: Optional[List[Dict]], fields: List[str]):
        if target is None:
            return
        for obj in target:
            for f in fields:
                v = obj.get(f)
                if v and isinstance(v, str):
                    h = text_hash(v)
                    if h in translated_map:
                        obj[f] = translated_map[h]

    apply(progress['done'], phases, ['name', 'desc'])
    if phases is not None:
        for p in phases:
            apply(progress['done'], p.get('lessons', []), ['name', 'summary', 'keywords'])
    apply(progress['done'], artifacts, ['name', 'description'])
    apply(progress['done'], glossary, ['term', 'definition'])

    # 重组成 data.zh.js
    def to_js(value, indent=2) -> str:
        return json.dumps(value, ensure_ascii=False, indent=indent)

    out = header
    for name, src in blocks:
        if name == 'PHASES' and phases is not None:
            out += 'const PHASES = ' + to_js(phases) + ';\n\n'
        elif name == 'GLOSSARY' and glossary is not None:
            out += 'const GLOSSARY = ' + to_js(glossary) + ';\n\n'
        elif name == 'ARTIFACTS' and artifacts is not None:
            out += 'const ARTIFACTS = ' + to_js(artifacts) + ';\n'
        else:
            out += 'const ' + name + ' = ' + src + ';\n\n'
    out += footer.lstrip('\n')

    OUT_PATH.write_text(out, encoding='utf-8')
    print(f'✅ 写出 {OUT_PATH} ({len(out)} bytes)')


if __name__ == '__main__':
    main()
