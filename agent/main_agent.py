"""
主Agent执行ReAct推理循环
"""
import asyncio
from typing import Dict, Any, List, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from planner.planner import DeepSeekPlanner
from tools.search_tool import KnowledgeBaseSearchTool
from tools.web_search_tool import WebSearchTool
from tools.calculator_tool import CalculatorTool
from tools.summarizer_tool import SummarizerTool
from memory.memory_manager import MemoryManager
from citation.citation_manager import CitationManager


class MainAgent:
    """主智能体，控制整个ReAct推理流程"""
    
    def __init__(self):
        """初始化主智能体"""
        self.config = Config()
        self.planner = DeepSeekPlanner()
        self.memory_manager = MemoryManager()
        self.citation_manager = CitationManager()
        
        # 初始化工具
        self.tools = {
            'search_knowledge_base': KnowledgeBaseSearchTool(),
            'web_search': WebSearchTool(),
            'calculator': CalculatorTool(),
            'summarize_text': SummarizerTool()
        }
        
        # 推理状态
        self.reasoning_trace = []
        self.search_results = []
        self.current_iteration = 0
        self.max_iterations = Config.MAX_ITERATIONS
    
    async def execute_reasoning(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        执行ReAct推理循环
        
        Args:
            query: 用户查询
            context: 可选的上下文信息
            
        Returns:
            推理结果
        """
        try:
            print(f"🧠 开始ReAct推理: {query}")
            
            # 初始化推理状态
            self._reset_reasoning_state()
            
            # 构建初始上下文
            full_context = await self._build_context(query, context)
            
            # 检查是否有相似的历史查询
            similar_queries = self.memory_manager.get_similar_queries(query)
            if similar_queries:
                print(f"💭 找到 {len(similar_queries)} 个相似历史查询")
                for sq in similar_queries[:2]:
                    self._add_reasoning_step("参考历史", f"相似查询: {sq['query']} -> {sq['answer']}")
            
            # 主推理循环
            final_result = await self._reasoning_loop(query, full_context)
            
            # 保存到内存
            memory_entry_id = self.memory_manager.add_memory_entry(
                query=query,
                context=full_context,
                reasoning_steps=self.reasoning_trace,
                search_results=self.search_results,
                final_answer=final_result.get('answer', ''),
                metadata={
                    'iterations': self.current_iteration,
                    'total_search_results': len(self.search_results),
                    'citations_count': len(self.citation_manager.get_all_citations())
                }
            )
            
            print(f"✅ 推理完成，记录ID: {memory_entry_id}")
            return final_result
            
        except Exception as e:
            print(f"❌ 推理执行失败: {str(e)}")
            return {
                'answer': f'推理过程中出现错误: {str(e)}',
                'citations': [],
                'reasoning_trace': self.reasoning_trace,
                'error': True
            }
    
    def _reset_reasoning_state(self):
        """重置推理状态"""
        self.reasoning_trace = []
        self.search_results = []
        self.current_iteration = 0
        self.citation_manager = CitationManager()  # 重新创建引用管理器
    
    async def _build_context(self, query: str, context: Optional[str] = None) -> str:
        """
        构建完整上下文
        
        Args:
            query: 查询
            context: 额外上下文
            
        Returns:
            完整上下文字符串
        """
        context_parts = [f"查询: {query}"]
        
        if context:
            context_parts.append(f"额外上下文: {context}")
        
        # 添加最近的对话历史
        recent_context = self.memory_manager.get_recent_context(3)
        if recent_context:
            context_parts.append(f"最近对话:\n{recent_context}")
        
        return "\n\n".join(context_parts)
    
    async def _reasoning_loop(self, query: str, context: str) -> Dict[str, Any]:
        """
        主推理循环
        
        Args:
            query: 查询
            context: 上下文
            
        Returns:
            推理结果
        """
        current_context = context
        
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            print(f"🔄 推理迭代 {self.current_iteration}/{self.max_iterations}")
            
            try:
                # 规划下一步行动
                available_tools = list(self.tools.keys())
                action_plan = await self.planner.plan_next_action(
                    query, current_context, available_tools
                )
                
                self._add_reasoning_step("思考", action_plan.get('thought', ''))
                
                # 检查是否完成
                if action_plan.get('is_complete'):
                    final_answer = action_plan.get('final_answer', '')
                    self._add_reasoning_step("完成", f"最终答案: {final_answer}")
                    
                    return await self._generate_final_result(query, final_answer)
                
                # 执行行动
                if action_plan.get('needs_action'):
                    action = action_plan.get('action')
                    action_input = action_plan.get('action_input')
                    
                    self._add_reasoning_step("行动", f"{action}: {action_input}")
                    
                    # 执行工具
                    observation = await self._execute_tool(action, action_input)
                    self._add_reasoning_step("观察", observation)
                    
                    # 更新上下文
                    current_context = self._update_context(current_context, action, observation)
                    
                    # 反思进展
                    if self.current_iteration % 3 == 0:  # 每3轮反思一次
                        reflection = await self._reflect_on_progress(query, current_context)
                        if reflection.get('is_sufficient'):
                            break
                else:
                    # 如果没有具体行动，尝试直接回答
                    print("⚠️ 未检测到具体行动，尝试生成答案")
                    break
                    
            except Exception as e:
                print(f"❌ 推理迭代 {self.current_iteration} 失败: {str(e)}")
                self._add_reasoning_step("错误", f"迭代失败: {str(e)}")
                
                # 尝试恢复
                if self.current_iteration < self.max_iterations - 1:
                    continue
                else:
                    break
        
        # 生成最终答案
        return await self._generate_final_answer_from_context(query, current_context)
    
    async def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            tool_input: 工具输入
            
        Returns:
            工具执行结果
        """
        try:
            if tool_name not in self.tools:
                return f"错误: 未知工具 {tool_name}"
            
            tool = self.tools[tool_name]
            
            if tool_name == 'search_knowledge_base':
                results = await tool.search(tool_input)
                self.search_results.extend(results)
                return self._format_search_results(results)
            
            elif tool_name == 'web_search':
                results = await tool.search(tool_input)
                self.search_results.extend(results)
                return self._format_search_results(results)
            
            elif tool_name == 'calculator':
                if '=' in tool_input:
                    # 方程求解
                    result = tool.solve_equation(tool_input)
                else:
                    # 普通计算
                    result = tool.calculate(tool_input)
                
                if result.get('error'):
                    return f"计算错误: {result['error']}"
                else:
                    return f"计算结果: {result['result']}"
            
            elif tool_name == 'summarize_text':
                result = await tool.summarize(tool_input)
                if result.get('error'):
                    return f"摘要错误: {result['error']}"
                else:
                    return f"摘要结果: {result['summary']}"
            
            else:
                return f"错误: 工具 {tool_name} 暂不支持"
                
        except Exception as e:
            error_msg = f"工具执行失败 {tool_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化搜索结果
        
        Args:
            results: 搜索结果列表
            
        Returns:
            格式化的结果字符串
        """
        if not results:
            return "未找到相关结果"
        
        formatted_results = []
        for i, result in enumerate(results[:5], 1):  # 只显示前5个结果
            content = result.get('content', '')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', '')
            
            formatted_result = f"结果{i}: {content}"
            if result.get('source'):
                formatted_result += f" (来源: {result['source']})"
            
            formatted_results.append(formatted_result)
        
        return "\n".join(formatted_results)
    
    def _update_context(self, current_context: str, action: str, observation: str) -> str:
        """
        更新推理上下文
        
        Args:
            current_context: 当前上下文
            action: 执行的行动
            observation: 观察结果
            
        Returns:
            更新后的上下文
        """
        new_info = f"\n执行了 {action}，结果: {observation}"
        
        # 限制上下文长度
        updated_context = current_context + new_info
        if len(updated_context) > self.config.MAX_CONTEXT_LENGTH:
            # 保留查询和最新信息，压缩中间部分
            lines = updated_context.split('\n')
            # 保留前几行（查询）和后几行（最新信息）
            truncated_context = '\n'.join(lines[:3] + ['...（中间内容省略）...'] + lines[-10:])
            return truncated_context
        
        return updated_context
    
    async def _reflect_on_progress(self, query: str, current_context: str) -> Dict[str, Any]:
        """
        反思当前进展
        
        Args:
            query: 原始查询
            current_context: 当前上下文
            
        Returns:
            反思结果
        """
        try:
            print("🤔 反思当前进展...")
            
            current_info = self._extract_key_information(current_context)
            reflection = await self.planner.reflect_on_progress(
                query, current_info, [step['content'] for step in self.reasoning_trace]
            )
            
            self._add_reasoning_step("反思", reflection.get('evaluation', ''))
            
            if not reflection.get('is_sufficient'):
                missing_info = reflection.get('missing_info', '')
                if missing_info and missing_info != '无':
                    self._add_reasoning_step("缺失信息", missing_info)
            
            return reflection
            
        except Exception as e:
            print(f"❌ 反思失败: {str(e)}")
            return {'is_sufficient': False, 'evaluation': '反思过程出错'}
    
    def _extract_key_information(self, context: str) -> str:
        """
        从上下文中提取关键信息
        
        Args:
            context: 上下文
            
        Returns:
            关键信息摘要
        """
        # 简单的关键信息提取
        lines = context.split('\n')
        key_lines = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ['结果', '答案', '发现', '显示', '表明']):
                key_lines.append(line)
        
        return '\n'.join(key_lines[-10:])  # 最多保留10条关键信息
    
    async def _generate_final_result(self, query: str, final_answer: str) -> Dict[str, Any]:
        """
        生成最终结果（当已有明确答案时）
        
        Args:
            query: 查询
            final_answer: 最终答案
            
        Returns:
            完整结果
        """
        # 生成引用
        citations = []
        if self.search_results:
            for result in self.search_results[:5]:  # 最多引用5个来源
                citation_id = self.citation_manager.add_citation(
                    source=result.get('source', ''),
                    content=result.get('content', '')[:200] + '...',
                    title=result.get('title', ''),
                    metadata=result.get('metadata', {})
                )
                citations.append(self.citation_manager.get_citation(citation_id)['citation_format'])
        
        return {
            'answer': final_answer,
            'citations': citations,
            'reasoning_trace': [step['content'] for step in self.reasoning_trace],
            'search_results': self.search_results,
            'metadata': {
                'iterations': self.current_iteration,
                'total_sources': len(self.search_results),
                'reasoning_steps': len(self.reasoning_trace)
            }
        }
    
    async def _generate_final_answer_from_context(self, query: str, context: str) -> Dict[str, Any]:
        """
        从上下文生成最终答案
        
        Args:
            query: 查询
            context: 上下文
            
        Returns:
            完整结果
        """
        try:
            print("📝 生成最终答案...")
            
            # 使用规划器生成最终答案
            final_result = await self.planner.generate_final_answer(
                query, self.search_results, [step['content'] for step in self.reasoning_trace]
            )
            
            # 在答案中插入引用标记
            answer_with_citations = self.citation_manager.insert_citations_in_text(
                final_result.get('answer', ''), self.search_results
            )
            
            # 生成引用列表
            citations_list = self.citation_manager.format_citations_list()
            
            return {
                'answer': answer_with_citations,
                'citations': citations_list.split('\n') if citations_list != "无引用文献。" else [],
                'reasoning_trace': [step['content'] for step in self.reasoning_trace],
                'search_results': self.search_results,
                'confidence': final_result.get('confidence', 'medium'),
                'explanation': final_result.get('explanation', ''),
                'metadata': {
                    'iterations': self.current_iteration,
                    'total_sources': len(self.search_results),
                    'reasoning_steps': len(self.reasoning_trace),
                    'citations_count': len(self.citation_manager.get_all_citations())
                }
            }
            
        except Exception as e:
            print(f"❌ 生成最终答案失败: {str(e)}")
            
            # 生成简单的回退答案
            return {
                'answer': f'基于收集的信息，无法生成完整答案。错误: {str(e)}',
                'citations': [],
                'reasoning_trace': [step['content'] for step in self.reasoning_trace],
                'search_results': self.search_results,
                'error': True
            }
    
    def _add_reasoning_step(self, step_type: str, content: str):
        """
        添加推理步骤
        
        Args:
            step_type: 步骤类型
            content: 步骤内容
        """
        step = {
            'iteration': self.current_iteration,
            'type': step_type,
            'content': content,
            'timestamp': self._get_current_timestamp()
        }
        
        self.reasoning_trace.append(step)
        
        # 同时记录到内存管理器
        self.memory_manager.add_reasoning_step(step)
        
        print(f"  📝 {step_type}: {content[:100]}{'...' if len(content) > 100 else ''}")
    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        return {
            'agent_type': 'ReAct Multi-Agent Research System',
            'available_tools': list(self.tools.keys()),
            'max_iterations': self.max_iterations,
            'current_session': self.memory_manager.get_session_summary(),
            'total_memory_entries': len(self.memory_manager.memory_entries),
            'reasoning_capabilities': [
                'ReAct推理循环',
                '多工具协调',
                '上下文管理',
                '记忆存储',
                '引用生成',
                '反思机制'
            ]
        }
    
    async def reset_session(self):
        """重置会话"""
        print("🔄 重置Agent会话...")
        self._reset_reasoning_state()
        self.memory_manager._initialize_session()
        print("✅ 会话重置完成")
