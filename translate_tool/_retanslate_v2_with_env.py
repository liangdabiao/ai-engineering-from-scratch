"""包装 retanslate_v2.py: 从 .env 加载 API key"""
import os
import sys
from pathlib import Path

ENV_FILE = Path(__file__).parent / '.env'

def load_env():
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

load_env()
# 把参数原样转发给 retanslate_v2
sys.path.insert(0, str(Path(__file__).parent))
from retanslate_v2 import main
main()
