"""
文本摘要器：长文档截断+压缩
"""
import re
from typing import Dict, Any, List, Optional
from config import Config


class SummarizerTool:
    """文本摘要工具"""
    
    def __init__(self):
        """初始化摘要工具"""
        self.enabled = Config.ENABLE_SUMMARIZER
        self.max_length = Config.MAX_CONTEXT_LENGTH
        self.planner = None
        self.llm_client = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """初始化LLM客户端用于摘要"""
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                base_url=Config.DEEPSEEK_BASE_URL
            )
        except ImportError as e:
            print(f"⚠️ 警告: 无法初始化LLM客户端 - {e}")
            print("将使用基础摘要功能")
    
    def summarize(self, text: str, max_length: Optional[int] = None, 
                       style: str = "general") -> Dict[str, Any]:
        """
        对文本进行摘要
        
        Args:
            text: 待摘要的文本
            max_length: 最大摘要长度
            style: 摘要风格（general/academic/news/bullet_points）
            
        Returns:
            摘要结果字典
        """
        if not text or not text.strip():
            return {
                'summary': '',
                'original_length': 0,
                'summary_length': 0,
                'compression_ratio': 0,
                'method': 'empty',
                'error': '输入文本为空'
            }
        
        try:
            print(f"📝 开始摘要，原文长度: {len(text)} 字符")
            
            max_len = max_length
            
            # 如果文本已经足够短，直接返回
            if len(text) <= max_len:
                return {
                    'summary': text,
                    'original_length': len(text),
                    'summary_length': len(text),
                    'compression_ratio': 1.0,
                    'method': 'no_compression',
                    'error': None
                }
            
            # 根据是否有LLM选择摘要方法
            if self.llm_client:
                summary = self._llm_summarize(text, max_len, style)
                method = 'llm'
            else:
                summary = self._extractive_summarize(text, max_len)
                method = 'extractive'
            
            compression_ratio = len(summary) / len(text)
            
            print(f"✅ 摘要完成，压缩比: {compression_ratio:.2f}")
            
            return {
                'summary': summary,
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': compression_ratio,
                'method': method,
                'error': None
            }
            
        except Exception as e:
            print(f"❌ 摘要生成出错: {str(e)}")
            # 出错时返回截断版本
            truncated = text[:max_len] + "..." if len(text) > max_len else text
            return {
                'summary': truncated,
                'original_length': len(text),
                'summary_length': len(truncated),
                'compression_ratio': len(truncated) / len(text),
                'method': 'truncation',
                'error': f'摘要生成失败，返回截断版本: {str(e)}'
            }

    def _llm_summarize(self, query: str, text: str, max_length: int,
                       style: str) -> str:
        """
        使用LLM生成摘要
        
        Args:
            query: 查询
            text: 原文
            max_length: 最大长度
            style: 摘要风格
            
        Returns:
            LLM生成的摘要
        """
        style_prompts = {
            'general': '请对以下文本进行摘要，保留主要信息和关键点：',
            'academic': '请对以下学术文本进行摘要，重点保留研究方法、发现和结论：',
            'news': '请对以下新闻文本进行摘要，突出关键事实和重要细节：',
            'bullet_points': '请用要点形式总结以下文本的主要内容：'
        }
        
        prompt = style_prompts.get(style, style_prompts['general'])
        prompt += f"\n\n查询内容：{query}\n\n原文：\n{text}\n\n要求：\n1. 重点总结与查询内容「{query}」最相关的信息\n2. 优先提取能回答查询的关键内容和细节\n3. 严格控制摘要长度不超过{max_length}字符\n4. 保持相关信息的完整性和准确性\n5. 语言简洁清晰，直接生成摘要内容，不要生成其他没有用的内容"
        
        messages = [
            {"role": "system", "content": "你是一个专业的文本摘要专家。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat.completions.create(
                model='deepseek-chat',  # 使用v3模型
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                stream=False,
            )
            summary = response.choices[0].message.content.strip()
            # print(f"摘要内容:\n{summary[:200]}...")  # 只打印前200字符
            
            # 如果摘要仍然过长，进行截断
            if len(summary) > max_length:
                summary = summary[:max_length].rsplit('。', 1)[0] + '。'
            
            return summary
        except Exception as e:
            print(f"❌ LLM摘要失败: {str(e)}，回退到抽取式摘要")
            # LLM失败时回退到抽取式摘要
            return self._extractive_summarize(text, max_length)
    
    def _extractive_summarize(self, text: str, max_length: int) -> str:
        """
        抽取式摘要（基于规则）
        
        Args:
            text: 原文
            max_length: 最大长度
            
        Returns:
            抽取式摘要
        """
        # 分割成句子
        sentences = self._split_sentences(text)
        
        if not sentences:
            return text[:max_length]
        
        # 计算句子重要性分数
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = self._calculate_sentence_score(sentence, i, len(sentences))
            scored_sentences.append((sentence, score, i))
        
        # 按分数排序
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最重要的句子，直到达到长度限制
        selected_sentences = []
        current_length = 0
        
        for sentence, score, original_index in scored_sentences:
            if current_length + len(sentence) <= max_length:
                selected_sentences.append((sentence, original_index))
                current_length += len(sentence)
            else:
                break
        
        # 按原始顺序排序
        selected_sentences.sort(key=lambda x: x[1])
        
        # 组合摘要
        summary = ''.join([sentence for sentence, _ in selected_sentences])
        
        return summary.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        分割文本为句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 简单的句子分割（基于标点符号）
        sentences = re.split(r'[。！？；\n]+', text)
        
        # 清理和过滤
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 过滤过短的句子
                cleaned_sentences.append(sentence + '。')
        
        return cleaned_sentences
    
    def _calculate_sentence_score(self, sentence: str, position: int, 
                                total_sentences: int) -> float:
        """
        计算句子重要性分数
        
        Args:
            sentence: 句子文本
            position: 句子在文档中的位置
            total_sentences: 总句子数
            
        Returns:
            重要性分数
        """
        score = 0.0
        
        # 位置特征：开头和结尾的句子更重要
        if position == 0:
            score += 0.3
        elif position == total_sentences - 1:
            score += 0.2
        elif position < total_sentences * 0.2:
            score += 0.1
        
        # 长度特征：适中长度的句子更重要
        sentence_len = len(sentence)
        if 20 <= sentence_len <= 100:
            score += 0.2
        elif sentence_len > 100:
            score += 0.1
        
        # 关键词特征
        important_words = [
            '重要', '关键', '主要', '核心', '基本', '显著', '明显',
            '研究', '发现', '结果', '结论', '方法', '分析',
            '因此', '所以', '总之', '综上', '可见'
        ]
        
        for word in important_words:
            if word in sentence:
                score += 0.1
        
        # 数字和统计数据
        if re.search(r'\d+', sentence):
            score += 0.05
        
        # 引用和专有名词
        if re.search(r'[A-Z][a-z]+|《[^》]+》', sentence):
            score += 0.05
        
        return score
    
    def batch_summarize(self, query: str, text: str, 
                       chunk_size: int = 5000, 
                       chunk_summary_length: int = 300,
                       final_summary_length: int = 1000,
                       style: str = "general") -> Dict[str, Any]:
        """
        分批总结长文档
        
        Args:
            query: 查询内容
            text: 待总结的文本
            chunk_size: 每批处理的字符数 (默认5000)
            chunk_summary_length: 每批总结的长度 (默认300字符)
            final_summary_length: 最终总结的长度 (默认1000字符)
            style: 总结风格
            
        Returns:
            总结结果字典
        """
        if not text or not text.strip():
            return {
                'summary': '',
                'original_length': 0,
                'summary_length': 0,
                'compression_ratio': 0,
                'method': 'empty',
                'chunks_processed': 0,
                'error': '输入文本为空'
            }
        
        try:
            print(f"📝 开始分批总结，原文长度: {len(text)} 字符")
            print(f"📦 分批参数: 块大小={chunk_size}, 块总结长度={chunk_summary_length}, 最终长度={final_summary_length}")
            
            # 分割文本为块
            chunks = self._split_text_into_chunks(text, chunk_size)
            print(f"📦 文本分割为 {len(chunks)} 块")
            
            # 对每个块进行总结
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"📝 正在总结第 {i+1}/{len(chunks)} 块 (长度: {len(chunk)} 字符)")
                
                if self.llm_client:
                    chunk_summary = self._llm_summarize(query, chunk, chunk_summary_length, style)
                else:
                    chunk_summary = self._extractive_summarize(chunk, chunk_summary_length)
                
                chunk_summaries.append(chunk_summary)
                print(f"✅ 第 {i+1} 块总结完成 (长度: {len(chunk_summary)} 字符)")
            
            # 合并所有块的总结
            combined_summary = "\n\n".join(chunk_summaries)
            print(f"🔗 合并所有块总结，总长度: {len(combined_summary)} 字符")
            
            # 对合并后的总结进行最终总结
            if len(combined_summary) <= final_summary_length:
                print("📝 合并总结已符合长度要求，无需再次总结")
                final_summary = combined_summary
            else:
                print("📝 对合并总结进行最终总结")
                if self.llm_client:
                    final_summary = self._llm_summarize(query, combined_summary, final_summary_length, style)
                else:
                    final_summary = self._extractive_summarize(combined_summary, final_summary_length)
            
            compression_ratio = len(final_summary) / len(text)
            
            print(f"✅ 分批总结完成，最终压缩比: {compression_ratio:.3f}")
            
            return final_summary
            
        except Exception as e:
            print(f"❌ 分批总结出错: {str(e)}")
            # 出错时回退到常规总结
            print("🔄 回退到常规总结")
            return self.summarize(text, final_summary_length, style)
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        将文本分割为指定大小的块，尽量在句子边界分割
        
        Args:
            text: 原文本
            chunk_size: 每块的目标大小
            
        Returns:
            文本块列表
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # 先尝试按段落分割
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # 如果当前块加上这个段落不会超出大小限制
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # 如果当前块不为空，先保存
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 如果段落本身就超过块大小，需要进一步分割
                if len(paragraph) > chunk_size:
                    # 按句子分割段落
                    sentences = self._split_sentences(paragraph)
                    current_sentence_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_sentence_chunk) + len(sentence) <= chunk_size:
                            current_sentence_chunk += sentence
                        else:
                            if current_sentence_chunk:
                                chunks.append(current_sentence_chunk)
                            current_sentence_chunk = sentence
                    
                    if current_sentence_chunk:
                        current_chunk = current_sentence_chunk
                else:
                    current_chunk = paragraph
        
        # 保存最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
