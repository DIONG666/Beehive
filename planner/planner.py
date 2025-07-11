"""
DeepSeek-R1调用器 + ReAct Prompt封装
"""
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from config import Config
from .prompt_templates import (
    QUERY_DECOMPOSITION_PROMPT,
    REFLECTION_PROMPT,
    FINAL_ANSWER_PROMPT,
)


class DeepSeekPlanner:
    """DeepSeek模型调用器，实现ReAct推理规划"""
    
    def __init__(self):
        """初始化DeepSeek客户端"""
        self.client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        self.model = Config.DEEPSEEK_MODEL
        self.temperature = Config.TEMPERATURE
    
    def generate_response(self, messages: List[Dict[str, str]], 
                              temperature: Optional[float] = None) -> str:
        """
        生成模型响应
        
        Args:
            messages: 对话消息列表
            temperature: 温度参数
            
        Returns:
            模型生成的响应
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=8192,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"DeepSeek API调用失败: {str(e)}")
    
    def decompose_query(self, query: str) -> List[str]:
        """
        分解复杂查询为子问题或提取web链接
        
        Args:
            query: 原始查询
            
        Returns:
            子查询或web链接列表
        """

        messages = [
            {"role": "system", "content": "你是一个专业的查询分析师。"},
            {"role": "user", "content": QUERY_DECOMPOSITION_PROMPT.format(query=query)}
        ]
        
        response = self.generate_response(messages)
        print(f"🔍 分解查询: \n{response}")

        # 提取链接和子查询
        links = self._extract_tag_content(response, "link")
        subqueries = self._extract_tag_content(response, "subquery")
        
        # 优先返回链接，如果没有链接则返回子查询
        result = links if links else subqueries
        
        return result if result else [query]
    
    
    def reflect_on_progress(self, query: str, current_info: str) -> Dict[str, str]:
        """
        反思当前进展
        
        Args:
            query: 原始查询
            current_info: 当前信息
        Returns:
            反思结果
        """
        messages = [
            {"role": "system", "content": "你是一个专业的研究助手，负责评估研究进展。"},
            {"role": "user", "content": REFLECTION_PROMPT.format(
                query=query,
                current_info=current_info
            )}
        ]

        response = self.generate_response(messages)
        print(f"📝 反思进展: \n{response}")
        
        # 使用标签提取结果
        judgment = self._extract_single_tag_content(response, "judgment")
        answer = self._extract_single_tag_content(response, "answer")
        reasoning = self._extract_single_tag_content(response, "reasoning")
        citations_str = self._extract_single_tag_content(response, "citations")
        suggestions_str = self._extract_single_tag_content(response, "suggestions")
        
        # 处理引用和建议
        citations = []
        if citations_str and citations_str != "无":
            citations = [c.strip() for c in citations_str.split(';') if c.strip()]
        
        suggested_queries = []
        if suggestions_str and suggestions_str != "无":
            suggested_queries = [q.strip() for q in suggestions_str.split(';') if q.strip()]
        
        can_answer = "是" in judgment
        
        return {
            'can_answer': can_answer,
            'answer': answer if can_answer else "",
            'reasoning_trace': reasoning,
            'citations': citations,
            'suggested_queries': suggested_queries
        }
    
    def generate_final_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        生成最终答案
        
        Args:
            query: 原始查询
            context: 完整上下文
            
        Returns:
            最终答案字典
        """
        
        messages = [
            {"role": "system", "content": "你是一个专业的研究助手，负责整合信息生成最终答案。"},
            {"role": "user", "content": FINAL_ANSWER_PROMPT.format(
                query=query,
                context=context
            )}
        ]
        
        response = self.generate_response(messages)
        print(f"📋 生成最终答案: \n{response}")
        
        # 使用标签提取结果
        answer = self._extract_single_tag_content(response, "answer")
        reasoning = self._extract_single_tag_content(response, "reasoning")
        citations_str = self._extract_single_tag_content(response, "citations")
        
        # 处理引用
        citations = []
        if citations_str and citations_str != "无":
            citations = [c.strip() for c in citations_str.split(';') if c.strip()]

        print(f"回答内容：answer='{answer}', reasoning='{reasoning}', citations={citations}")

        return {
            'answer': answer,
            'reasoning_trace': reasoning,
            'citations': citations,
        }
    
    def _extract_tag_content(self, text: str, tag: str) -> List[str]:
        """
        提取标签内容
        
        Args:
            text: 包含标签的文本
            tag: 标签名（不包含尖括号）
            
        Returns:
            标签内容列表
        """
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        return [match.strip() for match in matches if match.strip()]
    
    def _extract_single_tag_content(self, text: str, tag: str) -> str:
        """
        提取单个标签内容
        
        Args:
            text: 包含标签的文本
            tag: 标签名（不包含尖括号）
            
        Returns:
            标签内容字符串，如果未找到返回空字符串
        """
        contents = self._extract_tag_content(text, tag)
        return contents[0] if contents else ""

