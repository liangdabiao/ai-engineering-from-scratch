"""检查 mission 文件"""
import os
from pathlib import Path
files = sorted(Path('phases').glob('*/mission.md'))
print('mission.md count:', len(files))
for p in files[:5]: print(' ', p)
for root, dirs, fns in os.walk('phases'):
    for fn in fns:
        if 'mission' in fn.lower():
            print('Found:', os.path.join(root, fn))
