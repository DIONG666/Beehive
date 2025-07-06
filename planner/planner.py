"""
DeepSeek-R1调用器 + ReAct Prompt封装
"""
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from config import Config
from .prompt_templates import (
    REACT_SYSTEM_PROMPT,
    QUERY_DECOMPOSITION_PROMPT,
    REFLECTION_PROMPT,
    FINAL_ANSWER_PROMPT,
    TOOL_SELECTION_PROMPT,
    ERROR_HANDLING_PROMPT
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
        # print(f"🔍 分解查询: {query}\n响应: {response}")
        
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
        
        return sub_queries if sub_queries else [query]
    
    
    async def plan_next_action(self, query: str, context: str, 
                             available_tools: List[str]) -> Dict[str, Any]:
        """
        使用ReAct规划下一步行动
        
        Args:
            query: 当前查询
            context: 当前上下文
            available_tools: 可用工具列表
            
        Returns:
            包含行动计划的字典
        """
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"查询: {query}\n\n上下文: {context}"}
        ]
        
        response = await self.generate_response(messages)
        
        # 解析ReAct响应
        return self._parse_react_response(response, available_tools)
    
    def _parse_react_response(self, response: str, available_tools: List[str]) -> Dict[str, Any]:
        """
        解析ReAct格式的响应
        
        Args:
            response: 模型响应
            available_tools: 可用工具列表
            
        Returns:
            解析后的行动计划
        """
        lines = response.split('\n')
        thought = ""
        action = ""
        action_input = ""
        final_answer = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Thought:'):
                current_section = 'thought'
                thought = line.replace('Thought:', '').strip()
            elif line.startswith('Action:'):
                current_section = 'action'
                action = line.replace('Action:', '').strip()
            elif line.startswith('Action Input:'):
                current_section = 'action_input'
                action_input = line.replace('Action Input:', '').strip()
            elif line.startswith('Final Answer:'):
                current_section = 'final_answer'
                final_answer = line.replace('Final Answer:', '').strip()
            elif current_section and line:
                # 继续当前section的内容
                if current_section == 'thought':
                    thought += ' ' + line
                elif current_section == 'action_input':
                    action_input += ' ' + line
                elif current_section == 'final_answer':
                    final_answer += ' ' + line
        
        # 验证action是否在可用工具中
        if action and action not in available_tools:
            action = ""
            action_input = ""
        
        return {
            'thought': thought,
            'action': action,
            'action_input': action_input,
            'final_answer': final_answer,
            'needs_action': bool(action and action_input),
            'is_complete': bool(final_answer)
        }
    
    async def reflect_on_progress(self, query: str, current_info: str, 
                                reasoning_trace: List[str]) -> Dict[str, str]:
        """
        反思当前进展
        
        Args:
            query: 原始查询
            current_info: 当前信息
            reasoning_trace: 推理轨迹
            
        Returns:
            反思结果
        """
        trace_str = '\n'.join([f"{i+1}. {step}" for i, step in enumerate(reasoning_trace)])
        
        messages = [
            {"role": "system", "content": "你是一个专业的研究助手，负责评估研究进展。"},
            {"role": "user", "content": REFLECTION_PROMPT.format(
                query=query,
                current_info=current_info,
                reasoning_trace=trace_str
            )}
        ]
        
        response = await self.generate_response(messages)
        
        # 解析反思结果
        evaluation = ""
        missing_info = ""
        reasoning_evaluation = ""
        suggested_action = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('评估结果:'):
                current_section = 'evaluation'
                evaluation = line.replace('评估结果:', '').strip()
            elif line.startswith('缺失信息:'):
                current_section = 'missing'
                missing_info = line.replace('缺失信息:', '').strip()
            elif line.startswith('推理评价:'):
                current_section = 'reasoning'
                reasoning_evaluation = line.replace('推理评价:', '').strip()
            elif line.startswith('建议行动:'):
                current_section = 'action'
                suggested_action = line.replace('建议行动:', '').strip()
            elif current_section and line:
                if current_section == 'evaluation':
                    evaluation += ' ' + line
                elif current_section == 'missing':
                    missing_info += ' ' + line
                elif current_section == 'reasoning':
                    reasoning_evaluation += ' ' + line
                elif current_section == 'action':
                    suggested_action += ' ' + line
        
        return {
            'evaluation': evaluation,
            'missing_info': missing_info,
            'reasoning_evaluation': reasoning_evaluation,
            'suggested_action': suggested_action,
            'is_sufficient': '足够' in evaluation
        }
    
    async def generate_final_answer(self, query: str, search_results: List[Dict[str, Any]], 
                                  reasoning_trace: List[str]) -> Dict[str, Any]:
        """
        生成最终答案
        
        Args:
            query: 原始查询
            search_results: 搜索结果列表
            reasoning_trace: 推理轨迹
            
        Returns:
            最终答案字典
        """
        # 格式化搜索结果
        results_str = ""
        for i, result in enumerate(search_results, 1):
            results_str += f"{i}. {result.get('content', '')}\n"
            if result.get('source'):
                results_str += f"   来源: {result['source']}\n"
            results_str += "\n"
        
        trace_str = '\n'.join([f"{i+1}. {step}" for i, step in enumerate(reasoning_trace)])
        
        messages = [
            {"role": "system", "content": "你是一个专业的研究助手，负责整合信息生成最终答案。"},
            {"role": "user", "content": FINAL_ANSWER_PROMPT.format(
                query=query,
                search_results=results_str,
                reasoning_trace=trace_str
            )}
        ]
        
        response = await self.generate_response(messages)
        
        # 解析最终答案
        answer = ""
        explanation = ""
        citations = []
        confidence = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('答案:'):
                current_section = 'answer'
                answer = line.replace('答案:', '').strip()
            elif line.startswith('解释:'):
                current_section = 'explanation'
                explanation = line.replace('解释:', '').strip()
            elif line.startswith('引用:'):
                current_section = 'citations'
                citation = line.replace('引用:', '').strip()
                if citation:
                    citations.append(citation)
            elif line.startswith('置信度:'):
                current_section = 'confidence'
                confidence = line.replace('置信度:', '').strip()
            elif current_section and line:
                if current_section == 'answer':
                    answer += ' ' + line
                elif current_section == 'explanation':
                    explanation += ' ' + line
                elif current_section == 'citations':
                    citations.append(line)
        
        return {
            'answer': answer,
            'explanation': explanation,
            'citations': citations,
            'confidence': confidence,
            'search_results': search_results,
            'reasoning_trace': reasoning_trace
        }
    
    async def select_tool(self, task: str, available_tools: List[str]) -> Tuple[str, str]:
        """
        选择合适的工具
        
        Args:
            task: 当前任务描述
            available_tools: 可用工具列表
            
        Returns:
            (工具名称, 输入内容)
        """
        tools_str = ', '.join(available_tools)
        
        messages = [
            {"role": "system", "content": "你是一个工具选择专家。"},
            {"role": "user", "content": TOOL_SELECTION_PROMPT.format(
                task=task,
                available_tools=tools_str
            )}
        ]
        
        response = await self.generate_response(messages)
        
        # 解析工具选择
        tool_name = ""
        tool_input = ""
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('选择工具:'):
                tool_name = line.replace('选择工具:', '').strip()
            elif line.startswith('输入内容:'):
                tool_input = line.replace('输入内容:', '').strip()
        
        # 验证工具名称
        if tool_name not in available_tools:
            tool_name = available_tools[0] if available_tools else ""
        
        return tool_name, tool_input
    
    async def handle_error(self, error_type: str, error_message: str, 
                         context: str) -> Dict[str, str]:
        """
        处理错误并制定恢复策略
        
        Args:
            error_type: 错误类型
            error_message: 错误信息
            context: 当前上下文
            
        Returns:
            恢复策略
        """
        messages = [
            {"role": "system", "content": "你是一个错误恢复专家。"},
            {"role": "user", "content": ERROR_HANDLING_PROMPT.format(
                error_type=error_type,
                error_message=error_message,
                context=context
            )}
        ]
        
        response = await self.generate_response(messages)
        
        # 解析恢复策略
        strategy = ""
        explanation = ""
        modified_action = ""
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('选择策略:'):
                strategy = line.replace('选择策略:', '').strip()
            elif line.startswith('策略说明:'):
                explanation = line.replace('策略说明:', '').strip()
            elif line.startswith('修改后的操作:'):
                modified_action = line.replace('修改后的操作:', '').strip()
        
        return {
            'strategy': strategy,
            'explanation': explanation,
            'modified_action': modified_action
        }
