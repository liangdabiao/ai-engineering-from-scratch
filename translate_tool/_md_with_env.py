"""包装脚本: 用 .env 文件中的 API key 翻译"""
import os
import sys
import subprocess
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

# 转发给 md_translate
sys.path.insert(0, str(Path(__file__).parent))
from md_translate import main
main()
