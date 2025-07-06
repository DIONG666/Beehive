"""
主Agent执行多智能体推理循环
"""
import os
import sys
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from planner.planner import DeepSeekPlanner
from tools.search_tool import KnowledgeBaseSearchTool
from tools.web_search_tool import WebSearchTool
from tools.summarizer_tool import SummarizerTool
from memory.memory_manager import MemoryManager
from citation.citation_manager import CitationManager


class MainAgent:
    """主智能体，控制整个多智能体推理流程"""
    
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
            'summarize_text': SummarizerTool()
        }
        
        # 推理状态
        self.reasoning_trace = []
        self.search_results = []
        self.current_iteration = 0
        self.max_iterations = Config.MAX_ITERATIONS
    
    async def execute_reasoning(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        执行推理循环流程
        
        Args:
            query: 用户查询
            context: 可选的上下文信息
            
        Returns:
            推理结果
        """
        try:
            print(f"🧠 开始多智能体推理: {query}")
            
            # 初始化推理状态
            self._reset_reasoning_state()
            
            # 构建初始上下文
            full_context = await self._build_context(query, context)
            
            # 主推理循环
            final_result = await self._reasoning_loop(query, full_context)
            
            # 保存到内存
            self.memory_manager.add_memory_entry(
                query=query,
                context=full_context,
                reasoning_steps=self.reasoning_trace,
                search_results=self.search_results,
                final_answer=final_result.get('answer', ''),
                metadata=final_result.get('metadata', {})
            )
            
            return final_result
            
        except Exception as e:
            error_result = {
                'answer': f'推理过程中出现错误: {str(e)}',
                'citations': [],
                'reasoning_trace': self.reasoning_trace,
                'search_results': self.search_results,
                'metadata': {'error': True, 'iterations': self.current_iteration}
            }
            return error_result
    
    async def _reasoning_loop(self, query: str, context: str) -> Dict[str, Any]:
        """
        新的推理循环流程
        
        Args:
            query: 查询
            context: 上下文
            
        Returns:
            推理结果
        """
        collected_summaries = []
        
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            print(f"\n🔄 推理迭代 {self.current_iteration}/{self.max_iterations}")
            
            # 步骤1: 使用planner分解查询或提取链接
            sub_queries = await self.planner.decompose_query(query)
            self._add_reasoning_step(
                "query_decomposition", 
                f"分解得到 {len(sub_queries)} 个子查询: {sub_queries}"
            )
            
            # 步骤2: 对每个子查询进行搜索和总结
            iteration_summaries = []
            for sub_query in sub_queries:
                summary = await self._process_sub_query(sub_query)
                if summary:
                    iteration_summaries.append(summary)
            
            # 将本次迭代的总结加入总体上下文
            collected_summaries.extend(iteration_summaries)
            
            # 步骤3: 使用planner判断是否能得出答案
            combined_context = context + "\n\n已收集的信息:\n" + "\n".join(collected_summaries)
            
            can_answer = await self._check_if_can_answer(query, combined_context)
            
            if can_answer['can_answer']:
                print("✅ 已收集足够信息，生成最终答案")
                return await self._generate_final_answer(query, combined_context, can_answer['answer'])
            else:
                print(f"📝 信息不足，需要继续搜索: {can_answer['missing_info']}")
                # 如果planner建议了新的查询方向，更新query用于下次迭代
                if can_answer.get('suggested_queries'):
                    query = can_answer['suggested_queries'][0]  # 使用建议的第一个查询
        
        # 达到最大迭代次数，强制生成答案
        print("⚠️ 达到最大迭代次数，强制生成答案")
        final_context = context + "\n\n已收集的信息:\n" + "\n".join(collected_summaries)
        return await self._generate_final_answer(query, final_context, forced=True)
    
    async def _process_sub_query(self, sub_query: str) -> Optional[str]:
        """
        处理单个子查询
        
        Args:
            sub_query: 子查询或链接
            
        Returns:
            总结后的内容
        """
        try:
            print(f"🔍 处理子查询: {sub_query}")
            
            # 检查是否是链接
            if sub_query.startswith('https://'):
                # 直接从链接获取内容
                document = await self.tools['web_search']._get_content_via_jina(sub_query)
            else:
                # 先搜索知识库
                kb_result = await self.tools['search_knowledge_base'].search(sub_query)
                
                if kb_result['use_knowledge_base']:
                    print("✅ 使用知识库结果")
                    document = kb_result['results'][0] if kb_result['results'] else None
                else:
                    print("🌐 知识库相关性不足，使用Wikipedia搜索")
                    # 使用web搜索（仅Wikipedia）
                    web_results = await self.tools['web_search'].search(sub_query, count=1)
                    document = await self.tools['web_search']._get_content_via_jina(web_results[0]) if web_results else None
            
            if not document or document.get('error'):
                print(f"❌ 无法获取文档: {sub_query}")
                return None
            
            # 使用summarizer总结文档内容
            content = document.get('content', '')
            if len(content) > 500:  # 只对长文档进行总结
                summary_result = await self.tools['summarize_text'].summarize(
                    content, max_length=300, style='general'
                )
                summary = summary_result.get('summary', content[:300])
            else:
                summary = content
            
            # 记录搜索结果
            self.search_results.append(document)
            
            print(f"📝 完成子查询处理，总结长度: {len(summary)}")
            return f"关于'{sub_query}'：{summary}"
            
        except Exception as e:
            print(f"❌ 处理子查询出错: {str(e)}")
            return None
        
    
    async def _check_if_can_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        检查是否能够回答问题
        
        Args:
            query: 原始查询
            context: 当前上下文
            
        Returns:
            判断结果
        """
        try:
            check_prompt = f"""
基于以下信息，判断是否能够回答用户问题：

原始问题: {query}

已收集的信息:
{context}

请回答：
1. 能否基于现有信息回答问题？(是/否)
2. 如果可以，请提供简洁的答案
3. 如果不能，还需要搜索什么信息？
4. 建议的下一步查询方向（如有）

格式：
判断: 是/否
答案: [如果能回答则提供答案，否则写"信息不足"]
缺失信息: [具体需要的信息]
建议查询: [建议的新查询，用分号分隔多个]
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的信息分析师。"},
                {"role": "user", "content": check_prompt}
            ]
            
            response = await self.planner.generate_response(messages)
            
            # 解析响应
            lines = response.split('\n')
            can_answer = False
            answer = ""
            missing_info = ""
            suggested_queries = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('判断:'):
                    can_answer = '是' in line
                elif line.startswith('答案:'):
                    answer = line.split(':', 1)[1].strip()
                elif line.startswith('缺失信息:'):
                    missing_info = line.split(':', 1)[1].strip()
                elif line.startswith('建议查询:'):
                    queries_str = line.split(':', 1)[1].strip()
                    if queries_str and queries_str != "无":
                        suggested_queries = [q.strip() for q in queries_str.split(';') if q.strip()]
            
            return {
                'can_answer': can_answer,
                'answer': answer if can_answer else "",
                'missing_info': missing_info,
                'suggested_queries': suggested_queries
            }
            
        except Exception as e:
            print(f"❌ 检查答案能力出错: {str(e)}")
            return {
                'can_answer': False,
                'answer': "",
                'missing_info': f"检查过程出错: {str(e)}",
                'suggested_queries': []
            }
    
    async def _generate_final_answer(self, query: str, context: str, answer: str = "", forced: bool = False) -> Dict[str, Any]:
        """
        生成最终答案
        
        Args:
            query: 原始查询
            context: 完整上下文
            answer: 已有的答案（如有）
            forced: 是否强制生成
            
        Returns:
            最终结果
        """
        try:
            if not answer or forced:
                # 使用planner生成最终答案
                final_result = await self.planner.generate_final_answer(
                    query, self.search_results, [step['content'] for step in self.reasoning_trace]
                )
                answer = final_result.get('answer', answer)
            
            # 生成引用
            citations = []
            if self.search_results:
                citations = self.citation_manager.generate_citations(self.search_results)
            
            return {
                'answer': answer,
                'citations': citations,
                'reasoning_trace': [step['content'] for step in self.reasoning_trace],
                'search_results': self.search_results,
                'metadata': {
                    'iterations': self.current_iteration,
                    'total_sources': len(self.search_results),
                    'reasoning_steps': len(self.reasoning_trace),
                    'forced_answer': forced
                }
            }
            
        except Exception as e:
            print(f"❌ 生成最终答案出错: {str(e)}")
            return {
                'answer': f'生成答案时出错: {str(e)}',
                'citations': [],
                'reasoning_trace': [step['content'] for step in self.reasoning_trace],
                'search_results': self.search_results,
                'metadata': {'error': True, 'iterations': self.current_iteration}
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
            context_parts.append(f"最近对话: {recent_context}")
        
        return "\n\n".join(context_parts)
    
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
    
    async def reset_session(self):
        """重置会话"""
        print("🔄 重置Agent会话...")
        self._reset_reasoning_state()
        self.memory_manager._initialize_session()
        print("✅ 会话重置完成")
