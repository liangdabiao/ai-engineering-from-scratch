"""
术语表管理
"""

import json
import os
import re
from typing import Dict, List, Tuple


class Glossary:
    def __init__(self, glossary_file: str):
        self.glossary_file = glossary_file
        self.terms: Dict[str, str] = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.glossary_file):
            try:
                with open(self.glossary_file, 'r', encoding='utf-8') as f:
                    self.terms = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
    
    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.glossary_file)), exist_ok=True)
        with open(self.glossary_file, 'w', encoding='utf-8') as f:
            json.dump(self.terms, f, ensure_ascii=False, indent=2)
    
    def get_translation(self, term: str) -> str:
        """获取术语的译文，找不到返回原词"""
        return self.terms.get(term, term)
    
    def add_term(self, english: str, chinese: str):
        """添加术语"""
        self.terms[english] = chinese
    
    def build_prompt_terms(self) -> str:
        """生成用于翻译 prompt 的术语表字符串"""
        lines = []
        for en, zh in self.terms.items():
            lines.append(f"  - {en} → {zh}")
        return "\n".join(lines)
    
    def find_terms_in_text(self, text: str) -> List[Tuple[str, str]]:
        """在文本中找到所有出现的术语，返回 [(英文, 中文), ...]"""
        found = []
        for en, zh in self.terms.items():
            # 按单词边界匹配，避免部分匹配
            pattern = r'\b' + re.escape(en) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found.append((en, zh))
        return found
    
    def count(self) -> int:
        return len(self.terms)
