"""检查翻译失败原因"""
import json
from pathlib import Path
from collections import Counter
p = json.loads(Path('translate_tool/progress_phases.json').read_text(encoding='utf-8'))
errors = p['quizzes']['failed']
print(f'Total failed: {len(errors)}')
print('Sample failures:')
for f in errors[:10]:
    print(f'  {f}')
