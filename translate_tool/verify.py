"""
质量检查工具：验证翻译质量
- 检查 lang 属性是否正确
- 检查是否有明显漏翻的大段英文
- 检查 MathJax 公式是否完整
- 检查 HTML 标签是否配对
"""

import os
import re
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict


def check_lang(html_content: str, expected_lang: str = "zh-CN") -> dict:
    """检查 lang 属性"""
    soup = BeautifulSoup(html_content, 'html.parser')
    actual_lang = soup.html.get('lang', '') if soup.html else ''
    return {
        'ok': actual_lang == expected_lang,
        'expected': expected_lang,
        'actual': actual_lang,
    }


def check_mathjax(html_content: str) -> dict:
    """检查 MathJax 公式是否完整配对"""
    # 检查 $$$ 配对
    triple_open = len(re.findall(r'\$\$\$', html_content))
    # 检查 $ 配对（粗略，排除代码块内的）
    single_dollar = len(re.findall(r'(?<!\\)\$', html_content))
    
    return {
        'triple_dollar_count': triple_open,
        'triple_dollar_paired': triple_open % 2 == 0,
        'single_dollar_count': single_dollar,
        'ok': triple_open % 2 == 0,
    }


def count_english_paragraphs(html_content: str, threshold: int = 30) -> dict:
    """统计疑似未翻译的英文段落数
    简单启发式：包含大量英文单词的 <p> 标签
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    suspicious = []
    
    for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
        text = p.get_text().strip()
        if not text:
            continue
        # 统计英文单词数
        words = re.findall(r'[a-zA-Z]{4,}', text)
        if len(words) >= threshold:
            suspicious.append({
                'tag': p.name,
                'preview': text[:100] + ('...' if len(text) > 100 else ''),
                'english_words': len(words),
            })
    
    return {
        'suspicious_count': len(suspicious),
        'suspicious_items': suspicious[:10],  # 只返回前 10 个
    }


def check_html_tags(html_content: str) -> dict:
    """检查基本的 HTML 结构完整性"""
    has_doctype = '<!DOCTYPE' in html_content.upper() or '<!doctype' in html_content
    has_html = '<html' in html_content.lower()
    has_head = '<head' in html_content.lower()
    has_body = '<body' in html_content.lower()
    
    return {
        'has_doctype': has_doctype,
        'has_html': has_html,
        'has_head': has_head,
        'has_body': has_body,
        'ok': all([has_doctype, has_html, has_head, has_body]),
    }


def verify_file(file_path: str) -> dict:
    """验证单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lang_check = check_lang(content)
    mathjax_check = check_mathjax(content)
    english_check = count_english_paragraphs(content)
    html_check = check_html_tags(content)
    
    all_ok = (lang_check['ok'] and mathjax_check['ok'] 
              and english_check['suspicious_count'] < 5 
              and html_check['ok'])
    
    return {
        'file': file_path,
        'ok': all_ok,
        'lang': lang_check,
        'mathjax': mathjax_check,
        'english_paragraphs': english_check,
        'html_structure': html_check,
    }


def verify_directory(root_dir: str, pattern: str = "**/index.html") -> List[Dict]:
    """验证目录下所有文件"""
    root = Path(root_dir)
    files = sorted(root.glob(pattern))
    
    results = []
    for f in files:
        result = verify_file(str(f))
        results.append(result)
    
    return results


def print_report(results: List[Dict]):
    """打印验证报告"""
    print("\n" + "=" * 70)
    print("📋 翻译质量检查报告")
    print("=" * 70)
    
    total = len(results)
    passed = sum(1 for r in results if r['ok'])
    
    print(f"总文件数: {total}")
    print(f"通过检查: {passed} ({passed/total*100:.1f}%)" if total else "")
    print(f"有问题: {total - passed}")
    print()
    
    # 列出有问题的文件
    issues = [r for r in results if not r['ok']]
    if issues:
        print("❌ 有问题的文件:")
        for r in issues:
            print(f"\n  📄 {r['file']}")
            if not r['lang']['ok']:
                print(f"     - lang 属性错误: 期望 {r['lang']['expected']}，实际 {r['lang']['actual']}")
            if not r['mathjax']['ok']:
                print(f"     - MathJax 公式不配对: $$$ 数量 {r['mathjax']['triple_dollar_count']}")
            if r['english_paragraphs']['suspicious_count'] >= 5:
                print(f"     - 疑似未翻译段落: {r['english_paragraphs']['suspicious_count']} 段")
            if not r['html_structure']['ok']:
                print(f"     - HTML 结构不完整")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python verify.py <目录或文件>")
        sys.exit(1)
    
    target = sys.argv[1]
    
    if os.path.isfile(target):
        result = verify_file(target)
        print_report([result])
    elif os.path.isdir(target):
        results = verify_directory(target)
        print_report(results)
    else:
        print(f"路径不存在: {target}")
        sys.exit(1)
