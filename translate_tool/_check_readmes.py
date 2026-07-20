"""查看 README.zh.md 翻译结果"""
from pathlib import Path
for p in sorted(Path('phases').glob('*/README.zh.md')):
    print('---', p, '---')
    print(p.read_text(encoding='utf-8'))
