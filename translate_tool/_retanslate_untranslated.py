"""
data.js 翻译器 - 重译未翻译项
只对未翻译的 hash 进行翻译，更新 data.zh.js
"""
import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from data_translate import (
    extract_arrays, parse_js_array, text_hash, Translator, chunked,
    translate_batch, ROOT
)

DATA_PATH = ROOT / 'site' / 'data.js'
OUT_PATH = ROOT / 'site' / 'data.zh.js'
PROGRESS_PATH = ROOT / 'translate_tool' / 'progress_data.json'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', default='tokenhub')
    parser.add_argument('--model', default='deepseek-v4-flash-202605')
    parser.add_argument('--api-key', default=os.environ.get('TOKENHUB_API_KEY', ''))
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--request-delay', type=float, default=0.5)
    args = parser.parse_args()

    # 读 data.zh.js 作为基础（已经翻译过大部分）
    zh_text = OUT_PATH.read_text(encoding='utf-8')
    header, blocks, footer = extract_arrays(zh_text)
    parsed = {name: parse_js_array(src) for name, src in blocks}

    # 收集所有需要翻译的文本及其在 data.zh.js 中对应位置
    items: dict = {}
    location: dict = {}  # hash -> (section, parent_path, field)
    for sec in ('PHASES', 'ARTIFACTS', 'GLOSSARY'):
        if sec not in parsed:
            continue
        if sec == 'PHASES':
            for pi, p in enumerate(parsed[sec]):
                for k in ('name', 'desc'):
                    v = p.get(k)
                    if v and isinstance(v, str):
                        h = text_hash(v)
                        items[h] = v
                        location[h] = (sec, f'{pi}.{k}', k)
                for li, l in enumerate(p.get('lessons', [])):
                    for k in ('name', 'summary', 'keywords'):
                        v = l.get(k)
                        if v and isinstance(v, str):
                            h = text_hash(v)
                            items[h] = v
                            location[h] = (sec, f'{pi}.lessons.{li}.{k}', k)
        elif sec == 'ARTIFACTS':
            for ai, a in enumerate(parsed[sec]):
                for k in ('name', 'description'):
                    v = a.get(k)
                    if v and isinstance(v, str):
                        h = text_hash(v)
                        items[h] = v
                        location[h] = (sec, f'{ai}.{k}', k)
        elif sec == 'GLOSSARY':
            for gi, g in enumerate(parsed[sec]):
                for k in ('term', 'definition', 'says', 'means'):
                    v = g.get(k)
                    if v and isinstance(v, str):
                        h = text_hash(v)
                        items[h] = v
                        location[h] = (sec, f'{gi}.{k}', k)

    print(f'🔍 data.zh.js 中 {len(items)} 条文本')

    # 找出未翻译的（不含中文字符的）
    def is_english(s: str) -> bool:
        if not s: return False
        # 短文本可能没有中文（如 'Python'）跳过
        if len(s) < 3: return False
        has_cn = any('\u4e00' <= c <= '\u9fff' for c in s)
        has_en = any(c.isalpha() and ord(c) < 128 for c in s)
        return has_en and not has_cn

    untranslated = {h: t for h, t in items.items() if is_english(t)}
    print(f'⚠️  发现 {len(untranslated)} 条未翻译（纯英文）')

    if not untranslated:
        print('✅ 全部已翻译')
        return

    # 加载 API
    translator = Translator(
        provider=args.provider, model=args.model,
        api_key=args.api_key, request_delay=args.request_delay,
    )

    # 强制翻译：使用更强 prompt
    def force_translate_batch(batch: dict) -> dict:
        hashes = list(batch.keys())
        inputs = [batch[h] for h in hashes]
        sys_prompt = (
            "You are a professional English-to-Chinese translator. "
            "Translate EACH item to Simplified Chinese. "
            "For proper nouns or technical terms, add a Chinese translation in parentheses. "
            "Examples: 'Jupyter Notebook' -> 'Jupyter Notebook (Jupyter 笔记本)'; "
            "'Linux' -> 'Linux (Linux)'; 'Python' -> 'Python (Python)'. "
            "Keep all technical terms recognizable. Output ONLY a JSON array of translations."
        )
        user_prompt = json.dumps(inputs, ensure_ascii=False)
        result = []
        try:
            r = translator.client.chat.completions.create(
                model=translator.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=8192,
            )
            content = r.choices[0].message.content or ''
            # 提取 JSON
            m = re.search(r'\[.*\]', content, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    result = [str(x) for x in data]
        except Exception as e:
            print(f'   force_translate error: {e}')

        if len(result) != len(hashes):
            # 兜底逐条
            result = []
            for h in hashes:
                try:
                    r = translator.client.chat.completions.create(
                        model=translator.model,
                        temperature=0.3,
                        messages=[
                            {"role": "system", "content": 'Translate to Chinese, keep technical terms recognizable. Output ONLY translation.'},
                            {"role": "user", "content": batch[h]},
                        ],
                        max_tokens=512,
                    )
                    result.append(r.choices[0].message.content.strip())
                except Exception as ex:
                    result.append(batch[h])
        return {h: result[i] if i < len(result) else batch[h] for i, h in enumerate(hashes)}

    import re
    batches = chunked(untranslated, args.batch_size)
    print(f'🔄 准备重译 {len(batches)} 批')
    new_translations = {}
    for bi, batch in enumerate(batches):
        translated = force_translate_batch(batch)
        new_translations.update(translated)
        if (bi + 1) % 3 == 0 or bi == len(batches) - 1:
            print(f'   批 {bi + 1}/{len(batches)} ✓ 已重译 {len(new_translations)} 条')

    # 应用重译结果到 parsed
    applied = 0
    for h, new_text in new_translations.items():
        if h not in location:
            continue
        sec, path, field = location[h]
        # 解析 path: e.g. 'pi.field' 或 'pi.lessons.li.field'
        parts = path.split('.')
        obj = parsed[sec]
        for p in parts[:-1]:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = obj[p]
        obj[parts[-1]] = new_text
        applied += 1
    print(f'✅ 已应用 {applied} 条重译到数据结构')

    # 重写 data.zh.js
    def to_js(value, indent=2) -> str:
        return json.dumps(value, ensure_ascii=False, indent=indent)

    out = header
    for name, src in blocks:
        if name in parsed:
            out += 'const ' + name + ' = ' + to_js(parsed[name]) + ';\n\n'
        else:
            out += 'const ' + name + ' = ' + src + ';\n\n'
    out += footer.lstrip('\n')
    OUT_PATH.write_text(out, encoding='utf-8')
    print(f'✅ 写出 {OUT_PATH} ({len(out)} bytes)')


if __name__ == '__main__':
    import re
    main()
