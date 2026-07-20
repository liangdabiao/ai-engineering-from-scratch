"""
全量翻译脚本：支持断点续传
每次运行会跳过已翻译完成的文章，只翻译未完成的
进度保存在 progress.json 中
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_translate import translate_file
from translation_memory import TranslationMemory
from glossary import Glossary
from translator import Translator


def get_article_list(articles_dir: str) -> list:
    """获取所有文章目录列表"""
    articles = []
    for entry in os.listdir(articles_dir):
        full_path = os.path.join(articles_dir, entry)
        if os.path.isdir(full_path):
            index_html = os.path.join(full_path, "index.html")
            if os.path.exists(index_html):
                # 跳过一些特殊目录
                if entry in {'category', 'feed', 'archives', 'tags'}:
                    continue
                articles.append(entry)
    return sorted(articles)


def is_already_translated(filepath: str) -> bool:
    """判断文章是否已经翻译过（中文占比 > 30%）"""
    import re
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return False
    
    # 提取 body 内容
    body_match = re.search(r'<body[^>]*>(.*?)</body>', content, re.DOTALL | re.IGNORECASE)
    body = body_match.group(1) if body_match else content
    body = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r'<[^>]+>', '', body)
    body = re.sub(r'\$\$\$[\s\S]*?\$\$\$', '', body)
    
    chinese = len(re.findall(r'[\u4e00-\u9fff]', body))
    total = len(body.strip())
    if total < 100:
        return False
    return chinese / total > 0.3


def load_progress(progress_file: str) -> dict:
    """加载进度"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress_file: str, progress: dict):
    """保存进度"""
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def main():
    if len(sys.argv) < 2:
        print("用法: python translate_all.py <articles_dir> [--resume]")
        print("  --resume: 从中断处继续（默认行为）")
        print("  --restart: 重新开始所有翻译")
        sys.exit(1)
    
    articles_dir = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    progress_file = os.path.join(script_dir, 'progress.json')
    
    # 模式
    restart = '--restart' in sys.argv
    
    html_config = {
        'translatable_tags': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'li', 
                               'span', 'div', 'td', 'th', 'strong', 'em', 'b', 'i',
                               'label', 'option', 'button', 'title'],
        'translatable_attrs': ['alt', 'title', 'placeholder'],
        'skip_tags': ['script', 'style', 'noscript', 'code', 'pre'],
        'mathjax_pattern': r'\$\$\$[\s\S]*?\$\$\$',
    }
    
    tm = TranslationMemory(os.path.join(script_dir, 'translation_memory.json'))
    glossary = Glossary(os.path.join(script_dir, 'glossary.json'))
    api_key = os.environ.get("TOKENHUB_API_KEY", "")
    translator = Translator(
        provider="tokenhub", 
        model="deepseek-v4-flash-202605", 
        api_key=api_key,
        request_delay=3.0
    )
    
    # 获取文章列表
    all_articles = get_article_list(articles_dir)
    print(f"找到 {len(all_articles)} 篇文章")
    
    # 加载进度
    progress = load_progress(progress_file)
    if restart:
        progress = {"completed": [], "failed": []}
        print("重新开始翻译")
    else:
        print(f"已完成: {len(progress['completed'])} 篇")
        print(f"失败: {len(progress['failed'])} 篇")
    
    # 待翻译的文章
    completed_set = set(progress['completed'])
    todo = [a for a in all_articles if a not in completed_set]
    
    if not todo:
        print("所有文章都已翻译完成！")
        return
    
    print(f"待翻译: {len(todo)} 篇")
    print("=" * 60)
    
    total_nodes = 0
    total_cached = 0
    total_translated = 0
    success_count = 0
    fail_count = 0
    
    for i, article in enumerate(todo, 1):
        filepath = os.path.join(articles_dir, article, "index.html")
        
        print(f"\n[{i}/{len(todo)}] {article}")
        
        # 跳过已翻译的文章
        if is_already_translated(filepath):
            print(f"   ⏭️  已翻译，跳过")
            progress['completed'].append(article)
            success_count += 1
            continue
        
        try:
            result = translate_file(
                filepath, translator, tm, glossary,
                html_config, batch_size=20  # 小批次，更稳定
            )
            
            total_nodes += result['total_nodes']
            total_cached += result['cached']
            total_translated += result['translated']
            success_count += 1
            progress['completed'].append(article)
            
            cache_rate = result['cached'] / result['total_nodes'] * 100 if result['total_nodes'] else 0
            print(f"   ✅ 完成 - 命中: {result['cached']}/{result['total_nodes']} ({cache_rate:.0f}%)")
            
        except Exception as e:
            fail_count += 1
            progress['failed'].append(article)
            print(f"   ❌ 失败: {e}")
        
        # 每 5 篇保存一次进度和记忆库
        if i % 5 == 0:
            save_progress(progress_file, progress)
            tm.save()
            print(f"\n   💾 进度已保存（{success_count} 成功, {fail_count} 失败）")
        
        # 不是最后一篇就等待
        if i < len(todo):
            time.sleep(3)
    
    # 最终保存
    save_progress(progress_file, progress)
    tm.save()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 翻译完成")
    print("=" * 60)
    print(f"本次处理: {len(todo)} 篇")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总文本节点: {total_nodes}")
    if total_nodes:
        print(f"记忆命中: {total_cached} ({total_cached/total_nodes*100:.1f}%)")
    print(f"新翻译: {total_translated}")
    print(f"记忆库总量: {tm.stats()['total_entries']} 条")
    print("=" * 60)
    
    if progress['failed']:
        print(f"\n失败的文章（共 {len(progress['failed'])} 篇）:")
        for a in progress['failed'][:10]:
            print(f"  - {a}")
        if len(progress['failed']) > 10:
            print(f"  ... 还有 {len(progress['failed']) - 10} 篇")
        print("\n运行以下命令重试失败的:")
        print("  python translate_all.py <articles_dir> --retry-failed")


if __name__ == '__main__':
    main()
