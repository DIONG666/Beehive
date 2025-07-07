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
    
    async def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        在知识库中搜索相关文档，返回最相关结果和相关性状态
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            包含results, max_score, use_knowledge_base的字典
        """
        if not self.retriever:
            return {
                'results': [{
                    'content': '知识库搜索功能暂时不可用，请检查配置。',
                    'source': 'system_error',
                    'score': 0.0,
                    'error': True
                }],
                'max_score': 0.0,
                'use_knowledge_base': False
            }
        
        try:
            print(f"🔍 在知识库中搜索: {query}")
            
            # 首先进行向量检索
            top_k = top_k or Config.TOP_K
            initial_results = await self.retriever.search(query, top_k)
            
            if not initial_results or len(initial_results) == 0:
                return {
                    'results': [{
                        'content': '知识库中未找到相关内容',
                        'source': 'empty_results',
                        'score': 0.0
                    }],
                    'max_score': 0.0,
                    'use_knowledge_base': False
                }
            
            # 检查最高分数
            max_score = max(result.get('score', 0.0) for result in initial_results)
            print(f"📊 知识库检索最高相关性分数: {max_score:.3f}")
            
            # 如果最高分数 >= 0.7，使用reranker进一步排序
            if max_score >= 0.7:
                print("✅ 相关性分数足够高，使用知识库结果并进行重排序")
                
                if self.reranker:
                    try:
                        reranked_results = await self.reranker.rerank(
                            query, initial_results, Config.RERANK_TOP_K
                        )
                        final_results = reranked_results[:top_k]
                    except Exception as e:
                        print(f"⚠️ 重排序失败，使用原始结果: {str(e)}")
                        final_results = initial_results[:top_k]
                else:
                    final_results = initial_results[:top_k]
                
                return {
                    'results': final_results,
                    'max_score': max_score,
                    'use_knowledge_base': True
                }
            else:
                print(f"⚠️ 相关性分数过低 ({max_score:.3f} < 0.7)，建议使用网络搜索")
                return {
                    'results': initial_results[:top_k],
                    'max_score': max_score,
                    'use_knowledge_base': False
                }
                
        except Exception as e:
            print(f"❌ 知识库搜索出错: {str(e)}")
            return {
                'results': [{
                    'content': f'搜索过程中出现错误: {str(e)}',
                    'source': 'search_error',
                    'score': 0.0,
                    'error': True
                }],
                'max_score': 0.0,
                'use_knowledge_base': False
            }
    
    async def add_document_to_knowledge_base(self, document: Dict[str, Any]) -> bool:
        """
        将新文档添加到知识库
        
        Args:
            document: 文档字典，包含content, source, url等字段
            
        Returns:
            是否添加成功
        """
        try:
            print(f"📚 添加文档到知识库: {document.get('title', 'Unknown')[:50]}...")
            
            # 检查文档是否已存在
            if self._document_exists(document):
                print("📄 文档已存在，跳过添加")
                return True
            
            # 这里应该调用索引构建器来添加文档
            # 为简化，我们先打印日志
            print("✅ 文档已添加到知识库（模拟）")
            return True
            
        except Exception as e:
            print(f"❌ 添加文档到知识库失败: {str(e)}")
            return False
    
    def _document_exists(self, document: Dict[str, Any]) -> bool:
        """
        检查文档是否已存在于知识库中
        
        Args:
            document: 文档字典
            
        Returns:
            是否已存在
        """
        # 简单检查：基于URL或标题
        url = document.get('url', '')
        title = document.get('title', '')
        
        if not self.retriever or not self.retriever.documents:
            return False
        
        for existing_doc in self.retriever.documents:
            existing_url = existing_doc.get('url', '')
            existing_title = existing_doc.get('title', '')
            
            if url and url == existing_url:
                return True
            if title and title == existing_title:
                return True
        
        return False
    
