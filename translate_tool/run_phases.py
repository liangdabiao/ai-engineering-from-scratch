"""
phases 全量翻译调度器
====================

按 4 批依次翻译:
  1. README.md       (34 个)  ->  README.zh.md
  2. mission.md      (12 个)  ->  mission.zh.md
  3. quiz.json       (338 个) ->  quiz.zh.json
  4. docs/en.md      (503 个) ->  docs/zh.md

特性:
  - 断点续传(progress.json)
  - 限流退避(指数)
  - 翻译记忆库复用
  - 术语表强制
  - 失败文件可重试
"""
import os
import re
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = Path(__file__).resolve().parent / 'progress_phases.json'


def find_readmes(root: Path) -> List[Path]:
    """phases/*/README.md (阶段) + phases/*/*/README.md + phases/*/*/*/README.md (lesson 及 code 内的)"""
    files = []
    for p in root.glob('phases/*/README.md'):
        files.append(p)
    for p in root.glob('phases/*/*/README.md'):
        files.append(p)
    for p in root.glob('phases/*/*/*/README.md'):
        files.append(p)
    return sorted(set(files))


def find_missions(root: Path) -> List[Path]:
    return sorted(root.glob('phases/*/*/mission.md'))


def find_quizzes(root: Path) -> List[Path]:
    return sorted(root.glob('phases/*/*/quiz.json'))


def find_lessons(root: Path) -> List[Path]:
    return sorted(root.glob('phases/*/*/docs/en.md'))


def load_progress() -> Dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {
        'readmes': {'done': [], 'failed': []},
        'missions': {'done': [], 'failed': []},
        'quizzes': {'done': [], 'failed': []},
        'lessons': {'done': [], 'failed': []},
    }


def save_progress(progress: Dict):
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')


def make_dst(src: Path, kind: str) -> Path:
    """
    给定源文件路径,生成目标路径。
    kind in: readme, mission, quiz, lesson
    """
    if kind == 'readme':
        return src.with_name('README.zh.md')
    if kind == 'mission':
        return src.with_name('mission.zh.md')
    if kind == 'quiz':
        return src.with_name('quiz.zh.json')
    if kind == 'lesson':
        # phases/XX-slug/MM-slug/docs/en.md -> docs/zh.md
        return src.parent / 'zh.md'
    raise ValueError(kind)


def process_batch(
    name: str,
    files: List[Path],
    kind: str,
    translator,
    tm,
    glossary,
    progress: Dict,
    batch_size: int = 20,
    limit: Optional[int] = None,
    retry_only: bool = False,
    force: bool = False,
) -> Dict:
    """
    处理一批文件。返回 {done, failed, skipped, total}
    """
    from md_translate import translate_markdown_file
    from json_translate import translate_quiz_file

    if retry_only:
        done_set = set()
        failed = progress[name].get('failed', [])
        files = [f for f in files if str(f) in failed]
    else:
        done_set = set(progress[name].get('done', []))

    if limit:
        files = files[:limit]

    total = len(files)
    done = 0
    failed = 0
    skipped = 0
    errors = []

    print(f"\n=== 批 {name} ===  共 {total} 个文件  已完成 {len(done_set)}")

    for i, src in enumerate(files, 1):
        if str(src) in done_set and not force:
            skipped += 1
            continue
        dst = make_dst(src, kind)
        try:
            if kind in ('readme', 'mission', 'lesson'):
                result = translate_markdown_file(
                    str(src), str(dst), translator, tm, glossary,
                    batch_size=batch_size, force=force,
                )
            elif kind == 'quiz':
                result = translate_quiz_file(
                    str(src), str(dst), translator, tm, glossary,
                    batch_size=batch_size, force=force,
                )
            else:
                raise ValueError(kind)

            if result.get('skipped'):
                skipped += 1
                print(f"  [{i}/{total}] ⏭  {src.name} ({result.get('reason')})")
                progress[name].setdefault('done', []).append(str(src))
            else:
                done += 1
                progress[name].setdefault('done', []).append(str(src))
                print(f"  [{i}/{total}] ✓  {src.name} (段={result.get('segments', result.get('items', '?'))} 译={result.get('translated', 0)})")
            # 从失败列表移除
            if str(src) in progress[name].setdefault('failed', []):
                progress[name]['failed'].remove(str(src))
        except Exception as e:
            failed += 1
            progress[name].setdefault('failed', []).append(str(src))
            errors.append({'file': str(src), 'error': str(e), 'trace': traceback.format_exc()})
            print(f"  [{i}/{total}] ✗  {src.name} - {e}")

        # 每 10 个文件保存一次进度
        if i % 10 == 0:
            tm.save()
            save_progress(progress)

    # 最终保存
    tm.save()
    save_progress(progress)

    return {'total': total, 'done': done, 'failed': failed, 'skipped': skipped, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description='phases 全量翻译调度器')
    parser.add_argument('--batch', choices=['readmes', 'missions', 'quizzes', 'lessons', 'all'],
                        default='all', help='要翻译的批')
    parser.add_argument('--limit', type=int, help='限制每个批的文件数（试运行用）')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--provider', default='tokenhub')
    parser.add_argument('--model', default='hy3')
    parser.add_argument('--memory', default=str(Path(__file__).parent / 'translation_memory.json'))
    parser.add_argument('--glossary-file', default=str(Path(__file__).parent / 'glossary.json'))
    parser.add_argument('--retry-failed', action='store_true', help='只重试上次失败的文件')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--request-delay', type=float, default=2.0, help='请求间隔（秒）')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from translator import Translator
    from translation_memory import TranslationMemory
    from glossary import Glossary

    api_key = os.environ.get('TOKENHUB_API_KEY') or os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print("ERROR: 需要设置 TOKENHUB_API_KEY 或 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    translator = Translator(provider=args.provider, model=args.model, api_key=api_key, request_delay=args.request_delay)
    tm = TranslationMemory(args.memory)
    glossary = Glossary(args.glossary_file)

    progress = load_progress()
    summary = {}

    if args.batch in ('readmes', 'all'):
        files = find_readmes(ROOT)
        summary['readmes'] = process_batch(
            'readmes', files, 'readme', translator, tm, glossary, progress,
            batch_size=args.batch_size, limit=args.limit,
            retry_only=args.retry_failed, force=args.force,
        )

    if args.batch in ('missions', 'all'):
        files = find_missions(ROOT)
        summary['missions'] = process_batch(
            'missions', files, 'mission', translator, tm, glossary, progress,
            batch_size=args.batch_size, limit=args.limit,
            retry_only=args.retry_failed, force=args.force,
        )

    if args.batch in ('quizzes', 'all'):
        files = find_quizzes(ROOT)
        summary['quizzes'] = process_batch(
            'quizzes', files, 'quiz', translator, tm, glossary, progress,
            batch_size=args.batch_size, limit=args.limit,
            retry_only=args.retry_failed, force=args.force,
        )

    if args.batch in ('lessons', 'all'):
        files = find_lessons(ROOT)
        summary['lessons'] = process_batch(
            'lessons', files, 'lesson', translator, tm, glossary, progress,
            batch_size=args.batch_size, limit=args.limit,
            retry_only=args.retry_failed, force=args.force,
        )

    tm.save()
    save_progress(progress)

    print("\n" + "=" * 60)
    print("  翻译汇总")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:10s}  total={v['total']:4d}  done={v['done']:4d}  failed={v['failed']:3d}  skipped={v['skipped']:4d}")
    print("=" * 60)


if __name__ == '__main__':
    main()
