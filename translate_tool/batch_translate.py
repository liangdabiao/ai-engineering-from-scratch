"""
批量翻译主脚本
用法：python batch_translate.py <目标目录> [--dry-run] [--pattern="**/index.html"]
"""

import os
import sys
import glob
import argparse
from pathlib import Path

# 导入模块
from html_utils import extract_translatable_nodes, apply_translations, set_lang_attribute
from translation_memory import TranslationMemory
from glossary import Glossary
from translator import Translator


def find_html_files(root_dir: str, pattern: str = "**/index.html",
                     exclude_dirs: list = None) -> list:
    """查找所有需要翻译的 HTML 文件"""
    if exclude_dirs is None:
        exclude_dirs = ['feed', 'wp-content', 'wp-includes', 'wp-json', 'xmlrpc.php']
    
    root = Path(root_dir)
    files = []
    
    for html_file in root.glob(pattern):
        # 检查路径中是否包含排除目录
        path_parts = set(html_file.parts)
        if any(d in path_parts for d in exclude_dirs):
            continue
        files.append(str(html_file))
    
    return sorted(files)


def translate_file(file_path: str, translator: Translator, 
                   tm: TranslationMemory, glossary: Glossary,
                   html_config: dict, batch_size: int = 30,
                   dry_run: bool = False) -> dict:
    """翻译单个 HTML 文件"""
    print(f"\n📄 处理: {file_path}")
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 提取可翻译节点
    nodes, soup = extract_translatable_nodes(html_content, html_config)
    print(f"   提取到 {len(nodes)} 个独特文本节点")
    
    # 查翻译记忆，分出已翻译和待翻译的
    cached = {}
    to_translate = []
    to_translate_hashes = []
    
    for node in nodes:
        h = node['hash']
        cached_trans = tm.lookup(h)
        if cached_trans:
            cached[h] = cached_trans
        else:
            to_translate.append(node['text'])
            to_translate_hashes.append(h)
    
    print(f"   翻译记忆命中: {len(cached)} 条，待翻译: {len(to_translate)} 条")
    
    # 如果有待翻译的，调用 AI 翻译
    translations = dict(cached)  # 从已缓存的开始
    
    if to_translate:
        # 构建术语表 prompt
        glossary_text = glossary.build_prompt_terms()
        
        # 分批翻译（避免一次送太多）
        all_results = []
        
        for i in range(0, len(to_translate), batch_size):
            batch = to_translate[i:i+batch_size]
            print(f"   翻译批次 {i//batch_size + 1}/{(len(to_translate)+batch_size-1)//batch_size} ({len(batch)} 条)...")
            
            batch_results = translator.translate_batch(batch, glossary_text)
            all_results.extend(batch_results)
        
        # 存入翻译结果和记忆
        # - tgt 为 None：API 完全失败，跳过
        # - tgt == src：可能是已翻译的中文内容，接受但不入记忆库
        # - tgt != src：正常翻译，入记忆库
        success_count = 0
        skip_count = 0
        fail_count = 0
        for h, src, tgt in zip(to_translate_hashes, to_translate, all_results):
            if tgt is None:
                fail_count += 1
            elif tgt.strip() == src.strip():
                # 译文=原文，接受但不入记忆库
                translations[h] = tgt
                skip_count += 1
            else:
                translations[h] = tgt
                tm.add(h, src, tgt, context="article")
                success_count += 1
        
        if fail_count > 0 or skip_count > 0:
            print(f"   📊 翻译结果：新翻译 {success_count} 条，已翻译 {skip_count} 条，失败 {fail_count} 条")
    
    # 写回翻译
    if not dry_run:
        new_html = apply_translations(soup, translations, nodes)
        
        # 设置 lang 属性
        new_html = set_lang_attribute(new_html, "zh-CN")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_html)
        
        print(f"   ✅ 已写入翻译结果")
    else:
        print(f"   [dry-run] 将写入 {len(translations)} 条翻译")
    
    return {
        'file': file_path,
        'total_nodes': len(nodes),
        'cached': len(cached),
        'translated': len(to_translate),
    }


def main():
    parser = argparse.ArgumentParser(description='批量翻译 HTML 文件')
    parser.add_argument('target_dir', nargs='?', default='../articles',
                        help='目标目录（默认: ../articles）')
    parser.add_argument('--pattern', default='**/index.html',
                        help='文件匹配模式（默认: **/index.html）')
    parser.add_argument('--dry-run', action='store_true',
                        help='试运行，不实际修改文件')
    parser.add_argument('--provider', default='tokenhub',
                        help='翻译提供商（默认: tokenhub）')
    parser.add_argument('--model', default='hy3',
                        help='翻译模型（默认: hy3）')
    parser.add_argument('--batch-size', type=int, default=30,
                        help='每批翻译的文本数量（默认: 30）')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='请求间隔秒数（默认: 2.0，避免限流）')
    parser.add_argument('--config', default='config.yaml',
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置（简单起见直接用默认值，可扩展为读 yaml）
    html_config = {
        'translatable_tags': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'li', 
                               'span', 'div', 'td', 'th', 'strong', 'em', 'b', 'i',
                               'label', 'option', 'button', 'title'],
        'translatable_attrs': ['alt', 'title', 'placeholder'],
        'skip_tags': ['script', 'style', 'noscript', 'code', 'pre'],
        'mathjax_pattern': r'\$\$\$[\s\S]*?\$\$\$',
    }
    
    # 初始化组件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    tm = TranslationMemory(os.path.join(script_dir, 'translation_memory.json'))
    glossary = Glossary(os.path.join(script_dir, 'glossary.json'))
    translator = Translator(provider=args.provider, model=args.model, 
                            request_delay=args.delay)
    
    print("=" * 60)
    print("📚 BetterExplained 批量翻译工具")
    print("=" * 60)
    print(f"目标目录: {args.target_dir}")
    print(f"文件模式: {args.pattern}")
    print(f"翻译模型: {args.model}")
    print(f"术语表: {glossary.count()} 条")
    print(f"翻译记忆: {tm.stats()['total_entries']} 条")
    print(f"试运行: {'是' if args.dry_run else '否'}")
    print("=" * 60)
    
    # 查找文件
    files = find_html_files(args.target_dir, args.pattern)
    print(f"\n找到 {len(files)} 个文件待处理")
    
    if not files:
        print("没有找到文件，退出。")
        return
    
    # 逐个翻译
    stats = []
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]", end="")
        result = translate_file(f, translator, tm, glossary, html_config, 
                                args.batch_size, args.dry_run)
        stats.append(result)
    
    # 保存翻译记忆
    if not args.dry_run:
        tm.save()
        print(f"\n💾 翻译记忆已保存（共 {tm.stats()['total_entries']} 条）")
    
    # 统计
    total_nodes = sum(s['total_nodes'] for s in stats)
    total_cached = sum(s['cached'] for s in stats)
    total_translated = sum(s['translated'] for s in stats)
    
    print("\n" + "=" * 60)
    print("📊 处理完成")
    print("=" * 60)
    print(f"处理文件数: {len(files)}")
    print(f"总文本节点: {total_nodes}")
    print(f"翻译记忆命中: {total_cached} ({total_cached/total_nodes*100:.1f}%)" if total_nodes else "")
    print(f"新翻译内容: {total_translated}")
    print("=" * 60)


if __name__ == '__main__':
    main()
