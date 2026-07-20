"""
翻译记忆库管理
"""

import json
import os
from typing import Dict, Optional, Tuple


class TranslationMemory:
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        self.data = {
            "meta": {
                "version": "1.0",
                "source_lang": "en",
                "target_lang": "zh-CN",
            },
            "entries": {}
        }
        self._load()
    
    def _load(self):
        """从文件加载翻译记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                if 'entries' not in self.data:
                    self.data['entries'] = {}
            except (json.JSONDecodeError, IOError):
                pass
    
    def save(self):
        """保存到文件"""
        os.makedirs(os.path.dirname(os.path.abspath(self.memory_file)), exist_ok=True)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def lookup(self, text_hash: str) -> Optional[str]:
        """查找翻译，返回译文或 None"""
        entry = self.data['entries'].get(text_hash)
        if entry:
            return entry.get('target')
        return None
    
    def add(self, text_hash: str, source: str, target: str, context: str = "", 
            auto_save: bool = False):
        """添加一条翻译记忆"""
        self.data['entries'][text_hash] = {
            'source': source,
            'target': target,
            'context': context,
        }
        if auto_save:
            self.save()
    
    def batch_add(self, entries: Dict[str, Tuple[str, str, str]], auto_save: bool = False):
        """批量添加
        entries: {hash: (source, target, context)}
        """
        for h, (src, tgt, ctx) in entries.items():
            self.data['entries'][h] = {
                'source': src,
                'target': tgt,
                'context': ctx,
            }
        if auto_save:
            self.save()
    
    def stats(self) -> dict:
        """返回统计信息"""
        return {
            'total_entries': len(self.data['entries']),
        }
