"""
HTML 解析工具：提取可翻译的文本节点，翻译后写回
"""

import re
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
from typing import List, Dict, Tuple, Optional
import hashlib


def is_mathjax(text: str) -> bool:
    """判断文本是否是 MathJax 公式"""
    return bool(re.search(r'\$\$\$.*?\$\$\$', text, re.DOTALL))


def is_translatable_text(text: str) -> bool:
    """判断一段文本是否需要翻译"""
    if not text or not text.strip():
        return False
    stripped = text.strip()
    # 纯数字或符号不翻译
    if re.match(r'^[\d\s\.\,\;\:\!\?\-\—\–\(\)\[\]\{\}\/\\]+$', stripped):
        return False
    # 纯URL不翻译
    if re.match(r'^https?://', stripped):
        return False
    # 太短的（只有1-2个字符且是符号）可能不需要翻
    if len(stripped) <= 2 and not re.search(r'[a-zA-Z]', stripped):
        return False
    # 包含英文才需要翻译（纯中文已经是翻译好的了）
    if not re.search(r'[a-zA-Z]', stripped):
        return False
    return True


def text_hash(text: str) -> str:
    """生成文本的哈希值，作为翻译记忆的 key"""
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()[:12]


def extract_translatable_nodes(html_content: str, config: dict) -> List[Dict]:
    """
    从 HTML 中提取所有可翻译的文本节点
    返回列表，每个元素包含：
    - hash: 文本哈希
    - text: 原始文本
    - path: 节点路径（用于写回定位）
    - type: 'text' 或 'attr'
    - attr_name: 属性名（如果是属性）
    - context: 上下文标签名
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    nodes = []
    
    translatable_tags = set(config.get('translatable_tags', []))
    translatable_attrs = set(config.get('translatable_attrs', []))
    skip_tags = set(config.get('skip_tags', []))
    mathjax_pattern = config.get('mathjax_pattern', r'\$\$\$[\s\S]*?\$\$\$')
    
    def should_skip(element):
        """检查元素是否在应该跳过的标签内，或者是注释"""
        # 注释直接跳过
        if isinstance(element, Comment):
            return True
        
        parent = element.parent
        while parent:
            if parent.name in skip_tags:
                return True
            # 检查父节点是不是注释
            if isinstance(parent, Comment):
                return True
            parent = parent.parent
        return False
    
    def process_text_node(element, text_content, context_tag):
        """处理一个文本节点"""
        if should_skip(element):
            return
        
        text = text_content.strip()
        if not is_translatable_text(text):
            return
        
        # 跳过包含大量 MathJax 的
        if is_mathjax(text) and len(re.findall(r'[a-zA-Z]{3,}', text)) < 5:
            return
        
        nodes.append({
            'hash': text_hash(text),
            'text': text,
            'element_ref': element,
            'type': 'text',
            'context': context_tag,
        })
    
    # 遍历所有元素
    for tag_name in translatable_tags:
        for element in soup.find_all(tag_name):
            # 处理元素的直接文本子节点
            for child in element.children:
                if isinstance(child, NavigableString):
                    process_text_node(child, str(child), tag_name)
            
            # 处理可翻译的属性
            for attr_name in translatable_attrs:
                if attr_name in element.attrs:
                    attr_val = element[attr_name]
                    if isinstance(attr_val, str) and is_translatable_text(attr_val):
                        nodes.append({
                            'hash': text_hash(attr_val),
                            'text': attr_val,
                            'element_ref': element,
                            'type': 'attr',
                            'attr_name': attr_name,
                            'context': tag_name,
                        })
    
    # 去重：文本节点和属性节点分别去重
    # 因为即使文本相同，它们也是不同的元素，都需要写回
    seen_text = set()
    seen_attr = set()
    unique_nodes = []
    for node in nodes:
        key = node['hash']
        if node['type'] == 'text':
            if key not in seen_text:
                seen_text.add(key)
                unique_nodes.append(node)
        elif node['type'] == 'attr':
            # 属性节点也要去重，但和文本节点分开
            attr_key = (key, node['attr_name'])
            if attr_key not in seen_attr:
                seen_attr.add(attr_key)
                unique_nodes.append(node)
        else:
            unique_nodes.append(node)
    
    return unique_nodes, soup


def apply_translations(soup, translations: Dict[str, str], extracted_nodes: List[Dict]) -> str:
    """
    将翻译结果写回 BeautifulSoup 对象
    translations: {hash: translated_text}
    extracted_nodes: extract_translatable_nodes 返回的节点列表（用于参考类型）
    注意：我们会遍历整个 soup，替换所有匹配的文本和属性，
         而不仅仅是 extracted_nodes 中的引用，因为去重可能导致引用不完整
    """
    # 首先，收集所有需要替换的文本（从翻译表中）
    # 我们需要区分 text 替换和 attr 替换
    
    # 1. 替换所有文本节点
    # 收集所有翻译过的文本 hash -> 译文
    text_translations = {}
    attr_translations = {}  # (hash, attr_name) -> translated
    
    for node in extracted_nodes:
        h = node['hash']
        if h not in translations:
            continue
        translated = translations[h]
        if node['type'] == 'text':
            text_translations[h] = translated
        elif node['type'] == 'attr':
            attr_translations[(h, node['attr_name'])] = translated
    
    # 2. 遍历所有文本节点，查找并替换
    skip_tags = {'script', 'style', 'noscript', 'code', 'pre'}
    
    def is_inside_skip_tag(element):
        parent = element.parent
        while parent:
            if parent.name in skip_tags:
                return True
            if isinstance(parent, Comment):
                return True
            parent = parent.parent
        return False
    
    for element in soup.find_all(string=True):
        if isinstance(element, Comment):
            continue
        if is_inside_skip_tag(element):
            continue
        
        text = str(element).strip()
        if not text:
            continue
        
        h = text_hash(text)
        if h in text_translations:
            translated = text_translations[h]
            # 保留原始前后的空白
            original = str(element)
            stripped = original.strip()
            if stripped == original:
                element.replace_with(translated)
            else:
                # 有前后空白，保留
                leading = original[:len(original) - len(original.lstrip())]
                trailing = original[len(original.rstrip()):]
                element.replace_with(leading + translated + trailing)
    
    # 3. 遍历所有元素，替换属性
    translatable_attrs = {'alt', 'title', 'placeholder'}
    for element in soup.find_all(True):
        for attr_name in translatable_attrs:
            if attr_name not in element.attrs:
                continue
            attr_val = element[attr_name]
            if not isinstance(attr_val, str):
                continue
            if not attr_val.strip():
                continue
            h = text_hash(attr_val.strip())
            if (h, attr_name) in attr_translations:
                element[attr_name] = attr_translations[(h, attr_name)]
    
    return str(soup)


def set_lang_attribute(html_content: str, lang: str) -> str:
    """设置 HTML 的 lang 属性"""
    soup = BeautifulSoup(html_content, 'html.parser')
    if soup.html:
        soup.html['lang'] = lang
    return str(soup)
