"""检查翻译进度"""
import json
from pathlib import Path
files = list(Path('phases').glob('*/*/quiz.zh.json'))
print(f'quiz.zh.json files: {len(files)}')
progress = Path('translate_tool/progress_phases.json')
if progress.exists():
    p = json.loads(progress.read_text(encoding='utf-8'))
    print(f'progress quizzes: done={len(p["quizzes"]["done"])}, failed={len(p["quizzes"]["failed"])}')
    print(f'progress lessons: done={len(p["lessons"]["done"])}, failed={len(p["lessons"]["failed"])}')
    print(f'progress readmes: done={len(p["readmes"]["done"])}, failed={len(p["readmes"]["failed"])}')
    print(f'progress missions: done={len(p["missions"]["done"])}, failed={len(p["missions"]["failed"])}')
