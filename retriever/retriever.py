"""
实时向量检索器：query->document
"""
import os
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class VectorRetriever:
    """向量检索器"""
    
    def __init__(self):
        """初始化检索器"""
        self.config = Config()
        self.embedder = None
        self.index = None
        self.documents = []
        self.index_type = 'simple'
        self._initialize_components()
        self._load_index()
    
    def _initialize_components(self):
        """初始化组件"""
        try:
            from retriever.embedder import JinaEmbedder
            self.embedder = JinaEmbedder()
        except ImportError as e:
            print(f"⚠️ 警告: 无法导入嵌入器 - {e}")
            # 使用备用嵌入器
            try:
                from retriever.embedder import LocalEmbedder
                self.embedder = LocalEmbedder()
            except ImportError:
                print("❌ 无法导入任何嵌入器")
    
    def _load_index(self):
        """加载索引"""
        try:
            print("📂 加载索引...")
            
            # 加载文档
            docs_path = os.path.join(Config.INDEX_DIR, 'documents.pkl')
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"📄 加载了 {len(self.documents)} 个文档")
            else:
                print("⚠️ 文档文件不存在，将创建空索引")
                self.documents = []
                return
            
            # 尝试加载FAISS索引
            faiss_path = Config.FAISS_INDEX_PATH
            if os.path.exists(faiss_path):
                try:
                    import faiss
                    self.index = faiss.read_index(faiss_path)
                    self.index_type = 'faiss'
                    print("✅ FAISS索引加载成功")
                    return
                except ImportError:
                    print("⚠️ FAISS未安装，尝试加载简单索引")
                except Exception as e:
                    print(f"⚠️ FAISS索引加载失败: {str(e)}")
            
            # 尝试加载简单索引
            simple_path = os.path.join(Config.INDEX_DIR, 'simple_index.pkl')
            if os.path.exists(simple_path):
                with open(simple_path, 'rb') as f:
                    self.index = pickle.load(f)
                self.index_type = 'simple'
                print("✅ 简单索引加载成功")
            else:
                print("⚠️ 未找到任何索引文件")
                self.index = None
                
        except Exception as e:
            print(f"❌ 加载索引失败: {str(e)}")
            self.index = None
            self.documents = []
    
    async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self.embedder:
            return [{
                'content': '检索器未正确初始化',
                'source': 'retriever_error',
                'score': 0.0,
                'error': True
            }]
        
        if not self.documents:
            return [{
                'content': '知识库为空，请先构建索引',
                'source': 'empty_knowledge_base',
                'score': 0.0,
                'error': True
            }]
        
        try:
            print(f"🔍 搜索查询: {query}")
            
            # 生成查询嵌入
            query_embedding = await self.embedder.embed_single(query)
            
            if self.index_type == 'faiss' and self.index:
                results = await self._search_faiss(query_embedding, top_k)
            else:
                results = await self._search_simple(query_embedding, top_k)
            
            print(f"📋 找到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {str(e)}")
            return [{
                'content': f'搜索过程中出现错误: {str(e)}',
                'source': 'search_error',
                'score': 0.0,
                'error': True
            }]
    
    async def _search_faiss(self, query_embedding: List[float], 
                          top_k: int) -> List[Dict[str, Any]]:
        """
        使用FAISS搜索
        
        Args:
            query_embedding: 查询嵌入
            top_k: 返回数量
            
        Returns:
            搜索结果
        """
        try:
            import numpy as np
            
            # 转换查询嵌入
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # 搜索
            scores, indices = self.index.search(query_vector, min(top_k, len(self.documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    
                    # FAISS返回的是L2距离，转换为相似度分数
                    similarity = 1.0 / (1.0 + float(score))
                    
                    result = {
                        'content': doc.get('content', ''),
                        'title': doc.get('title', ''),
                        'source': doc.get('source', ''),
                        'score': similarity,
                        'metadata': doc.get('metadata', {}),
                        'id': doc.get('id', str(idx))
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ FAISS搜索失败: {str(e)}")
            return await self._search_simple(query_embedding, top_k)
    
    async def _search_simple(self, query_embedding: List[float], 
                           top_k: int) -> List[Dict[str, Any]]:
        """
        使用简单搜索
        
        Args:
            query_embedding: 查询嵌入
            top_k: 返回数量
            
        Returns:
            搜索结果
        """
        try:
            if self.index and 'embeddings' in self.index:
                doc_embeddings = self.index['embeddings']
            else:
                # 如果没有预计算的嵌入，现场计算
                texts = []
                for doc in self.documents:
                    text = doc.get('content', '')
                    if doc.get('title'):
                        text = doc['title'] + '\n' + text
                    texts.append(text)
                
                doc_embeddings = await self.embedder.embed_texts(texts)
            
            # 计算相似度
            similarities = self.embedder.batch_similarity(query_embedding, doc_embeddings)
            
            # 排序并获取top-k
            scored_docs = list(zip(self.documents, similarities, range(len(self.documents))))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for doc, score, idx in scored_docs[:top_k]:
                result = {
                    'content': doc.get('content', ''),
                    'title': doc.get('title', ''),
                    'source': doc.get('source', ''),
                    'score': float(score),
                    'metadata': doc.get('metadata', {}),
                    'id': doc.get('id', str(idx))
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ 简单搜索失败: {str(e)}")
            return []
    
    async def search_similar(self, document_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        搜索与指定文档相似的文档
        
        Args:
            document_id: 文档ID
            top_k: 返回数量
            
        Returns:
            相似文档列表
        """
        try:
            # 找到目标文档
            target_doc = None
            target_idx = None
            
            for i, doc in enumerate(self.documents):
                if doc.get('id') == document_id:
                    target_doc = doc
                    target_idx = i
                    break
            
            if not target_doc:
                return []
            
            # 获取目标文档的嵌入
            if self.index and 'embeddings' in self.index:
                doc_embeddings = self.index['embeddings']
                target_embedding = doc_embeddings[target_idx]
            else:
                target_text = target_doc.get('content', '')
                if target_doc.get('title'):
                    target_text = target_doc['title'] + '\n' + target_text
                target_embedding = await self.embedder.embed_single(target_text)
            
            # 使用目标文档的嵌入进行搜索
            results = await self._search_simple(target_embedding, top_k + 1)
            
            # 过滤掉目标文档本身
            filtered_results = [r for r in results if r.get('id') != document_id]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            print(f"❌ 相似文档搜索失败: {str(e)}")
            return []
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取文档
        
        Args:
            document_id: 文档ID
            
        Returns:
            文档字典
        """
        for doc in self.documents:
            if doc.get('id') == document_id:
                return doc
        return None
    
    def get_documents_by_source(self, source_pattern: str) -> List[Dict[str, Any]]:
        """
        根据来源模式获取文档
        
        Args:
            source_pattern: 来源模式
            
        Returns:
            匹配的文档列表
        """
        matching_docs = []
        for doc in self.documents:
            source = doc.get('source', '')
            if source_pattern in source:
                matching_docs.append(doc)
        return matching_docs
    
    async def hybrid_search(self, query: str, top_k: int = 10, 
                          keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        混合搜索（语义搜索 + 关键词搜索）
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            keyword_weight: 关键词权重
            
        Returns:
            混合搜索结果
        """
        try:
            # 语义搜索
            semantic_results = await self.search(query, top_k * 2)
            
            # 关键词搜索
            keyword_results = self._keyword_search(query, top_k * 2)
            
            # 合并和重新评分
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, keyword_weight
            )
            
            return combined_results[:top_k]
            
        except Exception as e:
            print(f"❌ 混合搜索失败: {str(e)}")
            return await self.search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        关键词搜索
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            
        Returns:
            关键词搜索结果
        """
        query_words = query.lower().split()
        scored_docs = []
        
        for i, doc in enumerate(self.documents):
            content = doc.get('content', '').lower()
            title = doc.get('title', '').lower()
            
            # 计算关键词匹配分数
            content_score = sum(content.count(word) for word in query_words)
            title_score = sum(title.count(word) * 2 for word in query_words)  # 标题权重更高
            
            total_score = content_score + title_score
            
            if total_score > 0:
                result = {
                    'content': doc.get('content', ''),
                    'title': doc.get('title', ''),
                    'source': doc.get('source', ''),
                    'score': float(total_score),
                    'metadata': doc.get('metadata', {}),
                    'id': doc.get('id', str(i))
                }
                scored_docs.append(result)
        
        # 排序
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # 归一化分数
        if scored_docs:
            max_score = scored_docs[0]['score']
            for result in scored_docs:
                result['score'] = result['score'] / max_score
        
        return scored_docs[:top_k]
    
    def _combine_search_results(self, semantic_results: List[Dict[str, Any]], 
                              keyword_results: List[Dict[str, Any]], 
                              keyword_weight: float) -> List[Dict[str, Any]]:
        """
        合并搜索结果
        
        Args:
            semantic_results: 语义搜索结果
            keyword_results: 关键词搜索结果
            keyword_weight: 关键词权重
            
        Returns:
            合并后的结果
        """
        # 创建文档ID到结果的映射
        semantic_map = {r['id']: r for r in semantic_results}
        keyword_map = {r['id']: r for r in keyword_results}
        
        all_doc_ids = set(semantic_map.keys()) | set(keyword_map.keys())
        
        combined_results = []
        for doc_id in all_doc_ids:
            semantic_score = semantic_map.get(doc_id, {}).get('score', 0.0)
            keyword_score = keyword_map.get(doc_id, {}).get('score', 0.0)
            
            # 加权合并分数
            combined_score = (1 - keyword_weight) * semantic_score + keyword_weight * keyword_score
            
            # 使用语义搜索的结果作为基础（通常质量更高）
            if doc_id in semantic_map:
                result = semantic_map[doc_id].copy()
            else:
                result = keyword_map[doc_id].copy()
            
            result['score'] = combined_score
            result['semantic_score'] = semantic_score
            result['keyword_score'] = keyword_score
            
            combined_results.append(result)
        
        # 按合并分数排序
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        return combined_results
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """获取检索器信息"""
        return {
            'num_documents': len(self.documents),
            'index_type': self.index_type,
            'embedder_available': self.embedder is not None,
            'index_loaded': self.index is not None,
            'capabilities': [
                '语义搜索',
                '关键词搜索',
                '混合搜索',
                '相似文档推荐',
                '多模式检索'
            ]
        }
