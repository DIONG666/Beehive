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
        self.max_context_length = Config.MAX_CONTEXT_LENGTH
    
    async def generate_response(self, messages: List[Dict[str, str]], 
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
                max_tokens=1024,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"DeepSeek API调用失败: {str(e)}")
    
    async def decompose_query(self, query: str) -> List[str]:
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
        
        response = await self.generate_response(messages)
        print(f"🔍 分解查询: {query}\n响应: \n{response}")
        
        # 解析子问题
        sub_queries = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('子问题' in line or '链接' in line):
                # 提取实际的查询内容
                if ':' in line:
                    query_content = line.split(':', 1)[1].strip()
                    sub_queries.append(query_content)
        
        print(f"🔍 分解结果: {sub_queries}")
        return sub_queries if sub_queries else [query]
    
    
    async def reflect_on_progress(self, query: str, current_info: str) -> Dict[str, str]:
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

        response = await self.generate_response(messages)
        print(f"📝 反思进展: {query}\n响应: \n{response}")
        
        # 解析反思结果     
        lines = response.split('\n')
        can_answer = False
        answer = ""
        reasoning_trace = ""
        citations = []
        suggested_queries = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('判断:'):
                can_answer = '是' in line
            elif line.startswith('答案:'):
                answer = line.split(':', 1)[1].strip()
            elif line.startswith('推理过程:'):
                reasoning_trace = line.split(':', 1)[1].strip()
            elif line.startswith('参考链接:'):
                citations_str = line.split(':', 1)[1].strip()
                if citations_str and citations_str != "无":
                    citations = [c.strip() for c in citations_str.split(';') if c.strip()]
            elif line.startswith('建议查询:'):
                queries_str = line.split(':', 1)[1].strip()
                if queries_str and queries_str != "无":
                    suggested_queries = [q.strip() for q in queries_str.split(';') if q.strip()]
        
        print(f"反思结果: can_answer={can_answer}, answer='{answer}', "
              f"reasoning_trace='{reasoning_trace}', citations={citations}, "
              f"suggested_queries={suggested_queries}")
        
        return {
            'can_answer': can_answer,
            'answer': answer if can_answer else "",
            'reasoning_trace': reasoning_trace,
            'citations': citations,
            'suggested_queries': suggested_queries
        }
    
    async def generate_final_answer(self, query: str, context: str) -> Dict[str, Any]:
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
        
        response = await self.generate_response(messages)
        print(f"📋 生成最终答案: {query}\n响应: \n{response}")
        
        # 解析最终答案
        lines = response.split('\n')
        answer = ""
        reasoning_trace = ""
        citations = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('答案:'):
                answer = line.split(':', 1)[1].strip()
            elif line.startswith('推理过程:'):
                reasoning_trace = line.split(':', 1)[1].strip()
            elif line.startswith('参考链接:'):
                citations_str = line.split(':', 1)[1].strip()
                if citations_str and citations_str != "无":
                    citations = [c.strip() for c in citations_str.split(';') if c.strip()]

        print(f"回答内容：answer='{answer}', reasoning_trace='{reasoning_trace}', citations={citations}")

        return {
            'answer': answer,
            'reasoning_trace': reasoning_trace,
            'citations': citations,
        }
    
