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


class MainAgent:
    """主智能体，控制整个多智能体推理流程"""
    
    def __init__(self):
        """初始化主智能体"""
        self.config = Config()
        self.planner = DeepSeekPlanner()
        self.memory_manager = MemoryManager()
        
        # 初始化工具
        self.tools = {
            'search_knowledge_base': KnowledgeBaseSearchTool(),
            'web_search': WebSearchTool(),
            'summarize_text': SummarizerTool()
        }
        
        # 推理状态
        self.current_iteration = 0
        self.max_iterations = Config.MAX_ITERATIONS
        self.recent_context_num =Config.RECENT_CONTEXT
    
    def execute_reasoning(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
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
            # context = await self._build_context(context)
            context = ""

            # 主推理循环
            final_result = self._reasoning_loop(query, context)
            
            # 保存到内存
            self.memory_manager.add_memory_entry(
                query=query,
                context=final_result.get('reasoning_trace', ''),
                final_answer=final_result.get('answer', ''),
            )
            
            return final_result
            
        except Exception as e:
            error_result = {
                'answer': f'推理过程中出现错误: {str(e)}',
                'citations': [],
            }
            return error_result
    
    def _reasoning_loop(self, query: str, context: str) -> Dict[str, Any]:
        """
        新的推理循环流程
        
        Args:
            query: 查询
            context: 上下文
            
        Returns:
            推理结果
        """
        links = []
        # 步骤1: 使用planner分解查询或提取链接
        sub_queries = self.planner.decompose_query(query)
        
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            print(f"\n🔄 推理迭代 {self.current_iteration}/{self.max_iterations}")
            
            # 步骤2: 对每个子查询进行搜索和总结
            for sub_query in sub_queries:
                if sub_query and f"关于'{sub_query}'的总结" not in context:
                    result = self._process_sub_query(query, sub_query, links)
                    if result:
                        context += f"\n\n关于'{sub_query}'的总结：{result['summary']}\n参考链接：{result['url']}"

            # 步骤3: 使用planner判断是否能得出答案
            reflection = self.planner.reflect_on_progress(query, context)

            if reflection['can_answer']:
                print("✅ 已收集足够信息，生成最终答案")
                return self._generate_final_answer(query, context, reflection['answer'], reflection['reasoning_trace'], reflection['citations'])
            else:
                print("📝 信息不足，需要继续搜索")
                # 如果planner建议了新的查询方向，更新query用于下次迭代
                if reflection['suggested_queries'] != []:
                    sub_queries = reflection['suggested_queries']
                else:
                    return self._generate_final_answer(query, context, forced=True)

        # 达到最大迭代次数，强制生成答案
        print("⚠️ 达到最大迭代次数，强制生成答案")
        return self._generate_final_answer(query, context, forced=True)

    def _process_sub_query(self, query: str, sub_query: str, links: List[str]) -> Optional[str]:
        """
        处理单个子查询
        
        Args:
            sub_query: 子查询或链接
            
        Returns:
            总结后的内容
        """
        try:
            print(f"\n🔍 处理子查询: {sub_query}")
            url = None
            
            # 检查是否是链接
            if sub_query.startswith('https://'):
                # 直接从链接获取内容
                document = self.tools['web_search']._get_content_via_jina(sub_query)
                url = sub_query
                links.append(url)
            else:
                # 先搜索知识库
                kb_result = self.tools['search_knowledge_base'].search(sub_query)
                
                if kb_result['use_knowledge_base']:
                    print("✅ 使用知识库结果")
                    document = kb_result['results'][0] if kb_result['results'] else None
                else:
                    print("🌐 知识库相关性不足，使用Web搜索")
                    # 使用web搜索
                    web_results = self.tools['web_search']._search_via_jina(sub_query, links, count=1)
                    document = self.tools['web_search']._get_content_via_jina(web_results[0]) if web_results else None
                    url = web_results[0] if web_results else None
                    links.append(url) if url else None
            
            if not document:
                print(f"❌ 无法获取文档: {sub_query}")
                return None
            
            # 使用summarizer总结文档内容
            if len(document) > 50000:  # 对长文档使用分批总结
                summary = self.tools['summarize_text'].batch_summarize(
                    query=query, 
                    text=document, 
                    chunk_size=50000,
                    chunk_summary_length=500,
                    final_summary_length=500,
                    style='general'
                )
            elif len(document) > 500:  # 中等长度文档使用常规总结
                summary = self.tools['summarize_text']._llm_summarize(
                    query, document, max_length=500, style='general'
                )
            else:
                summary = document
            
            
            print(f"📝 完成子查询处理，总结长度: {len(summary)}\n摘要内容: {summary[:100]}...")
            return {
                'summary': summary,
                'url': url
            }

        except Exception as e:
            print(f"❌ 处理子查询出错: {str(e)}")
            return None


    def _generate_final_answer(self, query: str, context: str, answer: str = "", reasoning_trace: str = "", citations: List[str] = [], forced: bool = False) -> Dict[str, Any]:
        """
        生成最终答案
        
        Args:
            query: 原始查询
            context: 完整上下文
            answer: 已有的答案（如有）
            reasoning_trace: 推理过程（如有）
            citations: 引用列表
            forced: 是否强制生成
            
        Returns:
            最终结果
        """
        try:
            if not answer or forced:
                # 使用planner生成最终答案
                final_result = self.planner.generate_final_answer(
                    query, context
                )
                answer = final_result.get('answer', answer)
                citations = final_result.get('citations', citations)
                reasoning_trace = final_result.get('reasoning_trace', reasoning_trace)

            # print(f"上下文内容:\n{context[:500]}...")  # 只打印前500字符
            return {
                'answer': answer,
                'citations': citations,
                'reasoning_trace': reasoning_trace,
            }
            
        except Exception as e:
            print(f"❌ 生成最终答案出错: {str(e)}")
            return {
                'answer': f'生成答案时出错: {str(e)}',
                'citations': [],
                'reasoning_trace': "",
            }
    
    def _reset_reasoning_state(self):
        """重置推理状态"""
        self.current_iteration = 0
    
    def _build_context(self, context: Optional[str] = None) -> str:
        """
        构建完整上下文
        
        Args:
            query: 查询
            context: 额外上下文
            
        Returns:
            完整上下文字符串
        """
        context_parts = []
        
        if context:
            context_parts.append(f"额外上下文: {context}")
        
        # 添加最近的对话历史
        recent_context = self.memory_manager.get_recent_context(self.recent_context_num)
        if recent_context:
            context_parts.append(f"最近对话: {recent_context}")
        
        return "\n\n".join(context_parts)

    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def reset_session(self):
        """重置会话"""
        print("🔄 重置Agent会话...")
        self._reset_reasoning_state()
        self.memory_manager._initialize_session()
        print("✅ 会话重置完成")
