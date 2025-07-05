"""
内部知识库搜索工具：调用 FAISS + reranker
"""
import asyncio
from typing import List, Dict, Any, Optional
from config import Config
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class KnowledgeBaseSearchTool:
    """知识库搜索工具"""
    
    def __init__(self):
        """初始化搜索工具"""
        self.retriever = None
        self.reranker = None
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化检索器和重排序器"""
        try:
            from retriever.retriever import VectorRetriever
            from reranker.reranker import JinaReranker
            
            self.retriever = VectorRetriever()
            self.reranker = JinaReranker()
        except ImportError as e:
            print(f"⚠️ 警告: 无法导入检索组件 - {e}")
            print("请确保已正确安装相关依赖")
    
    async def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        在知识库中搜索相关文档
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表，每个结果包含content, source, score等字段
        """
        if not self.retriever:
            return [{
                'content': '知识库搜索功能暂时不可用，请检查配置。',
                'source': 'system_error',
                'score': 0.0,
                'error': True
            }]
        
        try:
            top_k = top_k or Config.TOP_K
            
            # 第一阶段：向量检索
            print(f"🔍 在知识库中搜索: {query}")
            initial_results = await self.retriever.search(query, top_k)
            
            if not initial_results:
                return [{
                    'content': '未找到相关文档。',
                    'source': 'knowledge_base',
                    'score': 0.0
                }]
            
            # 第二阶段：重排序
            if self.reranker and len(initial_results) > 1:
                print(f"🔄 对{len(initial_results)}个结果进行重排序...")
                reranked_results = await self.reranker.rerank(query, initial_results)
                final_results = reranked_results[:Config.RERANK_TOP_K]
            else:
                final_results = initial_results[:Config.RERANK_TOP_K]
            
            print(f"✅ 找到 {len(final_results)} 个相关文档")
            return final_results
            
        except Exception as e:
            print(f"❌ 知识库搜索出错: {str(e)}")
            return [{
                'content': f'搜索过程中出现错误: {str(e)}',
                'source': 'search_error',
                'score': 0.0,
                'error': True
            }]
    
    async def search_by_keywords(self, keywords: List[str], 
                               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        根据关键词列表搜索
        
        Args:
            keywords: 关键词列表
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        # 将关键词组合成查询
        query = ' '.join(keywords)
        return await self.search(query, top_k)
    
    async def search_similar(self, document_id: str, 
                           top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        搜索与指定文档相似的文档
        
        Args:
            document_id: 文档ID
            top_k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        if not self.retriever:
            return []
        
        try:
            return await self.retriever.search_similar(document_id, top_k or Config.TOP_K)
        except Exception as e:
            print(f"❌ 相似文档搜索出错: {str(e)}")
            return []
    
    def get_tool_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            'name': 'search_knowledge_base',
            'description': '在内部知识库中搜索相关文档和信息',
            'parameters': {
                'query': '搜索查询字符串',
                'top_k': '返回结果数量（可选）'
            },
            'example_usage': 'search_knowledge_base("人工智能的发展历史")',
            'capabilities': [
                '向量语义搜索',
                '关键词匹配',
                '结果重排序',
                '相似文档推荐'
            ]
        }
    
    async def validate_query(self, query: str) -> bool:
        """
        验证查询是否有效
        
        Args:
            query: 待验证的查询
            
        Returns:
            查询是否有效
        """
        if not query or not query.strip():
            return False
        
        if len(query.strip()) < 2:
            return False
        
        # 检查是否包含有意义的内容
        meaningful_chars = sum(1 for c in query if c.isalnum() or c in '，。？！,.:;?!')
        if meaningful_chars < len(query) * 0.3:
            return False
        
        return True
    
    async def suggest_improvements(self, query: str, 
                                 results: List[Dict[str, Any]]) -> List[str]:
        """
        根据搜索结果质量建议查询改进
        
        Args:
            query: 原始查询
            results: 搜索结果
            
        Returns:
            改进建议列表
        """
        suggestions = []
        
        if not results or len(results) == 0:
            suggestions.append("尝试使用更通用的关键词")
            suggestions.append("检查拼写是否正确")
            suggestions.append("尝试使用同义词")
        
        elif len(results) == 1:
            suggestions.append("查询可能过于具体，尝试更广泛的术语")
        
        elif all(result.get('score', 0) < 0.5 for result in results):
            suggestions.append("结果相关性较低，尝试重新表述查询")
            suggestions.append("使用更具体的专业术语")
        
        return suggestions
