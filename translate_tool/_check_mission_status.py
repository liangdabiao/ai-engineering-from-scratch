"""检查所有 mission.zh.md 的翻译状态"""
from pathlib import Path
files = sorted(Path('phases').glob('*/*/mission.zh.md'))
print(f'Total: {len(files)}')
for p in files:
    text = p.read_text(encoding='utf-8')
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = sum(1 for c in text if c.isascii() and c.isalpha())
    ratio = cjk / (cjk + en) if (cjk + en) > 0 else 0
    status = '✓ 译' if ratio > 0.3 else '✗ 未译'
    print(f'  {status}  CJK={cjk:4d}  EN={en:4d}  ratio={ratio:.2f}  {p.name}')
