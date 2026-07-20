"""
quiz.json 翻译器：只翻译 question/options/explanation 字段,
                保留 stage/correct/lesson/title 字段不变。
                输出 quiz.zh.json。
"""
import os
import re
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Any


def text_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()[:12]


# quiz.json 的 schema:
# {
#   "lesson": "<dir-slug>",
#   "title": "<Lesson Title>",
#   "questions": [
#     {"stage": "pre",   "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": ""},
#     ...
#   ]
# }
#
# 翻译字段白名单:
#   - questions[].question
#   - questions[].options[]
#   - questions[].explanation
#
# 保留字段(结构/索引):
#   - lesson, title, questions[].stage, questions[].correct


def collect_texts(data) -> List[Tuple[str, str]]:
    """
    从 quiz 数据收集所有需要翻译的文本,返回 [(hash, text), ...]
    支持两种 schema:
      A. {"lesson": "...", "title": "...", "questions": [...]}
      B. [...] (直接是 questions 数组)
    """
    items: List[Tuple[str, str]] = []
    questions: List[Dict] = []
    title: str = ''

    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        title = data.get('title', '') or ''
        questions = data.get('questions', []) or []
    else:
        return []

    if title:
        items.append((text_hash(title), title))
    for q in questions:
        if not isinstance(q, dict):
            continue
        q_text = q.get('question', '')
        if q_text:
            items.append((text_hash(q_text), q_text))
        for opt in q.get('options', []) or []:
            if opt:
                items.append((text_hash(opt), opt))
        exp = q.get('explanation', '')
        if exp:
            items.append((text_hash(exp), exp))
    # 去重
    seen = set()
    unique = []
    for h, t in items:
        if h in seen:
            continue
        seen.add(h)
        unique.append((h, t))
    return unique


def apply_translations(data, translations: Dict[str, str]):
    """把译文写回 data 的白名单字段,保留其它字段
    支持 list 和 dict 两种 schema。
    """
    if isinstance(data, list):
        new_list = []
        for q in data:
            if not isinstance(q, dict):
                new_list.append(q)
                continue
            new_q = dict(q)
            if 'question' in q:
                new_q['question'] = translations.get(text_hash(q['question']), q['question'])
            if 'options' in q and q['options']:
                new_q['options'] = [
                    translations.get(text_hash(opt), opt) for opt in q['options']
                ]
            if 'explanation' in q:
                new_q['explanation'] = translations.get(text_hash(q['explanation']), q['explanation'])
            new_list.append(new_q)
        return new_list

    if isinstance(data, dict):
        new_data: Dict[str, Any] = {}
        for k, v in data.items():
            if k == 'questions':
                new_questions = []
                for q in v:
                    if not isinstance(q, dict):
                        new_questions.append(q)
                        continue
                    new_q = dict(q)
                    if 'question' in q:
                        new_q['question'] = translations.get(text_hash(q['question']), q['question'])
                    if 'options' in q and q['options']:
                        new_q['options'] = [
                            translations.get(text_hash(opt), opt) for opt in q['options']
                        ]
                    if 'explanation' in q:
                        new_q['explanation'] = translations.get(text_hash(q['explanation']), q['explanation'])
                    new_questions.append(new_q)
                new_data['questions'] = new_questions
            elif k == 'title':
                new_data['title'] = translations.get(text_hash(v), v)
            else:
                new_data[k] = v
        return new_data

    return data


def translate_quiz_file(
    src_path: str,
    dst_path: str,
    translator,
    tm,
    glossary,
    batch_size: int = 20,
    force: bool = False,
) -> Dict:
    with open(src_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 已翻译则跳过
    if not force and os.path.exists(dst_path):
        with open(dst_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        cjk = sum(1 for c in json.dumps(existing, ensure_ascii=False) if '\u4e00' <= c <= '\u9fff')
        ascii_letters = sum(1 for c in json.dumps(existing, ensure_ascii=False) if c.isascii() and c.isalpha())
        if ascii_letters > 0 and cjk / (cjk + ascii_letters) >= 0.3:
            return {'file': src_path, 'skipped': True, 'reason': 'already_translated'}

    items = collect_texts(data)
    if not items:
        # 空文件,直接复制
        with open(dst_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
        return {'file': src_path, 'items': 0}

    # 查记忆库
    cached: Dict[str, str] = {}
    to_call: List[str] = []
    to_call_h: List[str] = []
    for h, t in items:
        c = tm.lookup(h)
        if c:
            cached[h] = c
        else:
            to_call.append(t)
            to_call_h.append(h)

    print(f"  📝 {os.path.basename(src_path)}: 文本={len(items)} 记忆命中={len(cached)} 待译={len(to_call)}")

    if to_call:
        glossary_text = glossary.build_prompt_terms()
        all_results = []
        for i in range(0, len(to_call), batch_size):
            batch = to_call[i:i + batch_size]
            print(f"    批次 {i // batch_size + 1}/{(len(to_call) + batch_size - 1) // batch_size} ({len(batch)} 条)...")
            r = translator.translate_batch(batch, glossary_text)
            all_results.extend(r)
        for h, src, tgt in zip(to_call_h, to_call, all_results):
            if tgt is None:
                cached[h] = src
            elif tgt.strip() == src.strip():
                cached[h] = tgt
            else:
                cached[h] = tgt
                tm.add(h, src, tgt, context='phases_quiz')

    new_data = apply_translations(data, cached)
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
        f.write('\n')

    return {
        'file': src_path,
        'items': len(items),
        'cached': len(cached),
        'translated': len(to_call),
    }


# CLI
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst', nargs='?')
    parser.add_argument('--provider', default='tokenhub')
    parser.add_argument('--model', default='hy3')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--memory', default=os.path.join(os.path.dirname(__file__), 'translation_memory.json'))
    parser.add_argument('--glossary-file', default=os.path.join(os.path.dirname(__file__), 'glossary.json'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from translator import Translator
    from translation_memory import TranslationMemory
    from glossary import Glossary

    if args.dst is None:
        p = Path(args.src)
        # quiz.json -> quiz.zh.json
        args.dst = str(p.with_name('quiz.zh.json'))

    api_key = os.environ.get('TOKENHUB_API_KEY') or os.environ.get('OPENAI_API_KEY', '')
    translator = Translator(provider=args.provider, model=args.model, api_key=api_key, request_delay=2.0)
    tm = TranslationMemory(args.memory)
    glossary = Glossary(args.glossary_file)

    result = translate_quiz_file(
        args.src, args.dst, translator, tm, glossary,
        batch_size=args.batch_size, force=args.force,
    )
    tm.save()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
