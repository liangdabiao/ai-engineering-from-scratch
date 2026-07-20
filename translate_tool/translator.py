"""
AI 翻译器模块：支持多种翻译提供商
支持：OpenAI API、腾讯云 TokenHub 等 OpenAI 兼容接口
可扩展：DeepL、Google、本地 LLM 等
"""

import os
import json
import re
import time
from typing import List, Dict, Optional
import openai


class Translator:
    def __init__(self, provider: str = "tokenhub", model: str = "hy3", 
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 temperature: float = 0.3, request_delay: float = 1.0):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.request_delay = request_delay  # 请求间隔（秒），避免限流
        
        # 根据 provider 设置默认值
        if provider == "tokenhub":
            default_base_url = "https://tokenhub.tencentmaas.com/v1"
            default_model = "hy3"
            env_key = "TOKENHUB_API_KEY"
            self.model = model if model != "gpt-4o" else default_model
        elif provider == "openai":
            default_base_url = None
            default_model = "gpt-4o"
            env_key = "OPENAI_API_KEY"
            self.model = model
        else:
            # 自定义 provider，需要传入 base_url
            default_base_url = base_url
            env_key = "CUSTOM_API_KEY"
            self.model = model
        
        self.api_key = api_key or os.environ.get(env_key, "")
        
        if not self.api_key:
            print(f"⚠️  警告：未设置 {env_key}，将使用模拟模式")
        else:
            kwargs = {"api_key": self.api_key}
            if default_base_url or base_url:
                kwargs["base_url"] = base_url or default_base_url
            self.client = openai.OpenAI(**kwargs)
    
    def build_translation_prompt(self, texts: List[str], glossary_terms: str = "",
                                  source_lang: str = "English", 
                                  target_lang: str = "Simplified Chinese") -> str:
        """构建翻译 prompt"""
        texts_json = json.dumps(texts, ensure_ascii=False, indent=2)
        
        prompt = f"""你是一个专业的技术翻译专家，擅长将{source_lang}技术文章翻译成{target_lang}。

## 翻译规则

1. **准确翻译技术术语**：数学、编程等专业术语要翻译准确
2. **保留格式**：不要修改任何 Markdown、HTML 标签或特殊格式
3. **保留 MathJax 公式**：用 $$$...$$$ 或 $...$ 包裹的数学公式完全保留，不要翻译
4. **保留代码**：代码块和行内代码完全保留
5. **语气自然**：翻译后的中文要自然流畅，符合中文表达习惯，不要生硬直译
6. **专有名词**：人名、地名、品牌名保留原文（如 Archimedes、BetterExplained、WordPress 等）
7. **首次术语对照**：对于重要的数学/技术术语，首次出现时在中文译名后加括号标注原文，例如：导数(Derivative)、积分(Integral)
"""
        
        if glossary_terms:
            prompt += f"""
## 术语表（请严格遵循）
{glossary_terms}
"""
        
        prompt += f"""
## 待翻译文本列表（JSON 格式）

以下是需要翻译的文本数组，请翻译后返回相同结构的 JSON 数组：

{texts_json}

## 要求

- 返回纯 JSON，不要任何解释文字
- 数组长度必须和输入一致
- 第 N 个元素是输入中第 N 个元素的翻译
- 保持原文的标点符号和格式
"""
        return prompt
    
    def translate_batch(self, texts: List[str], glossary_terms: str = "",
                         max_retries: int = 3) -> List[str]:
        """批量翻译一组文本
        返回长度相同的列表，翻译失败的条目为 None（而不是原文）
        """
        if not texts:
            return []
        
        # 模拟模式（没有 API Key 时）
        if not hasattr(self, 'client'):
            print(f"  [模拟模式] 将翻译 {len(texts)} 条文本...")
            return [f"[TRANSLATED] {t}" for t in texts]
        
        result = [None] * len(texts)  # None 表示未翻译/失败
        pending_indices = list(range(len(texts)))  # 待翻译的索引
        
        for attempt in range(max_retries):
            if not pending_indices:
                break
            
            pending_texts = [texts[i] for i in pending_indices]
            prompt = self.build_translation_prompt(pending_texts, glossary_terms)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": "You are a professional translator."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                result_text = response.choices[0].message.content.strip()
                
                try:
                    parsed = self._parse_json_robust(result_text)
                    translations = self._extract_translations(parsed, len(pending_texts))
                    
                    if translations is not None:
                        # 逐条检查结果
                        for idx_in_pending, original_idx in enumerate(pending_indices):
                            original_text = texts[original_idx]
                            translated = translations[idx_in_pending]
                            
                            if translated:
                                # 翻译成功（即使和原文相同也接受——可能是已翻译的中文内容）
                                result[original_idx] = translated
                            else:
                                # 空翻译，保持 None
                                pass
                        
                        # 所有条目都已处理，退出重试循环
                        pending_indices = []
                        continue
                    
                    print(f"  ⚠️  返回格式不匹配，重试 ({attempt+1}/{max_retries})...")
                except Exception as e:
                    print(f"  ⚠️  JSON 解析失败: {e}，重试 ({attempt+1}/{max_retries})...")
                
            except Exception as e:
                print(f"  ⚠️  翻译请求失败: {e}，重试 ({attempt+1}/{max_retries})...")
                # 429 限流：指数退避，更长等待
                if '429' in str(e) or 'rate_limit' in str(e):
                    wait_time = 15 * (2 ** attempt)  # 15s, 30s, 60s
                    print(f"  ⏳ 限流等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                else:
                    time.sleep(2 ** attempt)
        
        # 结束：失败的条目保持 None
        failed_count = sum(1 for r in result if r is None)
        if failed_count > 0:
            print(f"  ❌  {failed_count}/{len(texts)} 条最终翻译失败")
        
        return result
    
    def _parse_json_robust(self, text: str) -> dict:
        """更健壮的 JSON 解析，尝试修复常见问题"""
        # 1. 先尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 2. 尝试提取第一个 JSON 对象/数组
        # 找第一个 { 或 [
        obj_match = re.search(r'[\{\[]', text)
        if obj_match:
            start = obj_match.start()
            # 尝试从这里开始解析
            for end in range(len(text), start, -1):
                try:
                    candidate = text[start:end]
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
        
        # 3. 尝试修复常见的 JSON 问题（多余的尾随逗号等）
        # 去掉多余的尾随逗号
        fixed = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # 4. 最后手段：用正则提取所有字符串值（不推荐，但能救急）
        raise ValueError(f"无法解析 JSON: {text[:200]}")
    
    def _extract_translations(self, result, expected_len: int) -> Optional[List[str]]:
        """从各种可能的 JSON 结构中提取翻译数组"""
        # 本身就是数组
        if isinstance(result, list) and len(result) == expected_len:
            return result
        
        # 字典，找常见的 key
        if isinstance(result, dict):
            # 常见的 key
            for key in ['translations', 'result', 'texts', 'data', 'translation',
                        'output', 'response', 'translated_text', 'translated']:
                if key in result:
                    val = result[key]
                    if isinstance(val, list) and len(val) == expected_len:
                        return val
                    if isinstance(val, str):
                        # 可能是单条翻译
                        if expected_len == 1:
                            return [val]
            
            # 遍历所有值，找第一个长度匹配的数组
            for v in result.values():
                if isinstance(v, list) and len(v) == expected_len:
                    return v
            
            # 嵌套字典，递归找
            for v in result.values():
                if isinstance(v, dict):
                    found = self._extract_translations(v, expected_len)
                    if found is not None:
                        return found
        
        return None
