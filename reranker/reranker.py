"""
使用 Jina reranker 对召回结果重新打分排序
"""
import requests
import json
from typing import List, Dict, Any, Tuple
from config import Config


class JinaReranker:
    """Jina重排序器"""
    
    def __init__(self):
        """初始化重排序器"""
        self.api_key = Config.JINA_API_KEY
        self.model = Config.JINA_RERANKER_MODEL
        self.api_url = "https://api.jina.ai/v1/rerank"
        self.enabled = bool(self.api_key)
    
    async def rerank(self, query: str, documents: List[Dict[str, Any]], 
                    top_k: int = None) -> List[Dict[str, Any]]:
        """
        重新排序文档
        
        Args:
            query: 查询字符串
            documents: 文档列表
            top_k: 返回的文档数量
            
        Returns:
            重新排序后的文档列表
        """
        if not self.enabled:
            print("⚠️ Jina reranker API密钥未配置，跳过重排序")
            return documents[:top_k] if top_k else documents
        
        if not documents:
            return []
        
        try:
            print(f"🔄 对 {len(documents)} 个文档进行重排序...")
            
            # 准备文档文本
            doc_texts = []
            for doc in documents:
                text = doc.get('content', '')
                if doc.get('title'):
                    text = doc['title'] + '\n' + text
                # 限制文本长度以适应API限制
                text = text[:2000] if len(text) > 2000 else text
                doc_texts.append(text)
            
            # 调用Jina reranker API
            reranked_scores = await self._call_reranker_api(query, doc_texts)
            
            if not reranked_scores:
                print("⚠️ 重排序失败，返回原始顺序")
                return documents[:top_k] if top_k else documents
            
            # 重新排序文档
            reranked_documents = self._apply_reranking(documents, reranked_scores)
            
            result_count = len(reranked_documents)
            print(f"✅ 重排序完成，返回 {result_count} 个文档")
            
            return reranked_documents[:top_k] if top_k else reranked_documents
            
        except Exception as e:
            print(f"❌ 重排序过程失败: {str(e)}")
            return documents[:top_k] if top_k else documents
    
    async def _call_reranker_api(self, query: str, 
                               documents: List[str]) -> List[Tuple[int, float]]:
        """
        调用Jina reranker API
        
        Args:
            query: 查询字符串
            documents: 文档文本列表
            
        Returns:
            (索引, 分数) 元组列表
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            data = {
                'model': self.model,
                'query': query,
                'documents': documents,
                'top_k': len(documents)  # 返回所有文档的排序
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Reranker API请求失败: {response.status_code} - {response.text}")
                return []
            
            result = response.json()
            
            if 'results' not in result:
                print("❌ API响应格式错误")
                return []
            
            # 解析结果
            scored_indices = []
            for item in result['results']:
                index = item.get('index', -1)
                score = item.get('relevance_score', 0.0)
                if index >= 0:
                    scored_indices.append((index, score))
            
            return scored_indices
            
        except Exception as e:
            print(f"❌ Reranker API调用失败: {str(e)}")
            return []
    
    def _apply_reranking(self, documents: List[Dict[str, Any]], 
                        scored_indices: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """
        应用重排序结果
        
        Args:
            documents: 原始文档列表
            scored_indices: (索引, 分数) 列表
            
        Returns:
            重新排序的文档列表
        """
        reranked_docs = []
        
        # 按分数排序
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        for index, score in scored_indices:
            if 0 <= index < len(documents):
                doc = documents[index].copy()
                # 更新分数为reranker的分数
                doc['rerank_score'] = score
                doc['original_score'] = doc.get('score', 0.0)
                doc['score'] = score  # 使用reranker分数作为最终分数
                reranked_docs.append(doc)
        
        return reranked_docs
    
    async def batch_rerank(self, query_doc_pairs: List[Tuple[str, List[Dict[str, Any]]]]) -> List[List[Dict[str, Any]]]:
        """
        批量重排序
        
        Args:
            query_doc_pairs: (查询, 文档列表) 元组列表
            
        Returns:
            重排序后的文档列表的列表
        """
        results = []
        
        for query, documents in query_doc_pairs:
            reranked = await self.rerank(query, documents)
            results.append(reranked)
        
        return results
    
    def get_reranker_info(self) -> Dict[str, Any]:
        """获取重排序器信息"""
        return {
            'model': self.model,
            'api_enabled': self.enabled,
            'api_url': self.api_url,
            'capabilities': [
                '语义相关性重排序',
                '批量处理',
                '分数校准',
                '查询-文档匹配优化'
            ]
        }


class SimpleReranker:
    """简单重排序器（备用方案）"""
    
    def __init__(self):
        """初始化简单重排序器"""
        self.enabled = True
    
    async def rerank(self, query: str, documents: List[Dict[str, Any]], 
                    top_k: int = None) -> List[Dict[str, Any]]:
        """
        使用简单规则重排序
        
        Args:
            query: 查询字符串
            documents: 文档列表
            top_k: 返回数量
            
        Returns:
            重排序后的文档列表
        """
        try:
            print(f"🔄 使用简单规则重排序 {len(documents)} 个文档...")
            
            # 计算新的相关性分数
            reranked_docs = []
            query_words = set(query.lower().split())
            
            for doc in documents:
                new_score = self._calculate_relevance_score(doc, query_words)
                
                # 创建新的文档副本
                new_doc = doc.copy()
                new_doc['original_score'] = doc.get('score', 0.0)
                new_doc['rerank_score'] = new_score
                
                # 组合原始分数和新分数
                combined_score = 0.7 * new_score + 0.3 * doc.get('score', 0.0)
                new_doc['score'] = combined_score
                
                reranked_docs.append(new_doc)
            
            # 按新分数排序
            reranked_docs.sort(key=lambda x: x['score'], reverse=True)
            
            result = reranked_docs[:top_k] if top_k else reranked_docs
            print(f"✅ 简单重排序完成，返回 {len(result)} 个文档")
            
            return result
            
        except Exception as e:
            print(f"❌ 简单重排序失败: {str(e)}")
            return documents[:top_k] if top_k else documents
    
    def _calculate_relevance_score(self, document: Dict[str, Any], 
                                 query_words: set) -> float:
        """
        计算文档相关性分数
        
        Args:
            document: 文档
            query_words: 查询词集合
            
        Returns:
            相关性分数
        """
        content = document.get('content', '').lower()
        title = document.get('title', '').lower()
        
        # 基础分数组件
        title_matches = sum(1 for word in query_words if word in title)
        content_matches = sum(1 for word in query_words if word in content)
        
        # 位置权重：标题中的匹配权重更高
        title_score = title_matches * 2.0
        content_score = content_matches * 1.0
        
        # 长度惩罚：过短或过长的文档分数降低
        content_length = len(content)
        if content_length < 50:
            length_penalty = 0.5
        elif content_length > 2000:
            length_penalty = 0.8
        else:
            length_penalty = 1.0
        
        # 查询覆盖率：查询词在文档中的覆盖程度
        coverage = (title_matches + min(content_matches, len(query_words))) / len(query_words) if query_words else 0
        
        # 词频密度：查询词在文档中的密度
        word_count = len(content.split())
        density = content_matches / word_count if word_count > 0 else 0
        
        # 组合分数
        relevance_score = (
            0.4 * coverage +           # 查询覆盖率
            0.3 * (title_score + content_score) / (len(query_words) + 1) +  # 匹配分数
            0.2 * min(density, 0.1) * 10 +  # 词频密度（限制最大值）
            0.1 * length_penalty       # 长度惩罚
        )
        
        return min(relevance_score, 1.0)  # 限制最大值为1.0
    
    def get_reranker_info(self) -> Dict[str, Any]:
        """获取重排序器信息"""
        return {
            'type': 'simple',
            'enabled': self.enabled,
            'features': [
                '关键词匹配',
                '位置权重',
                '长度惩罚',
                '查询覆盖率',
                '词频密度'
            ]
        }


class HybridReranker:
    """混合重排序器"""
    
    def __init__(self):
        """初始化混合重排序器"""
        self.jina_reranker = JinaReranker()
        self.simple_reranker = SimpleReranker()
        self.use_jina = self.jina_reranker.enabled
    
    async def rerank(self, query: str, documents: List[Dict[str, Any]], 
                    top_k: int = None) -> List[Dict[str, Any]]:
        """
        混合重排序
        
        Args:
            query: 查询字符串
            documents: 文档列表
            top_k: 返回数量
            
        Returns:
            重排序后的文档列表
        """
        if self.use_jina:
            # 优先使用Jina reranker
            jina_results = await self.jina_reranker.rerank(query, documents, top_k)
            
            # 如果Jina失败，回退到简单重排序
            if not jina_results or len(jina_results) == 0:
                print("⚠️ Jina重排序失败，使用简单重排序")
                return await self.simple_reranker.rerank(query, documents, top_k)
            
            return jina_results
        else:
            # 直接使用简单重排序
            return await self.simple_reranker.rerank(query, documents, top_k)
    
    async def dual_rerank(self, query: str, documents: List[Dict[str, Any]], 
                         top_k: int = None, blend_ratio: float = 0.7) -> List[Dict[str, Any]]:
        """
        双重排序：结合Jina和简单重排序的结果
        
        Args:
            query: 查询字符串
            documents: 文档列表
            top_k: 返回数量
            blend_ratio: Jina分数的权重
            
        Returns:
            混合重排序结果
        """
        if not self.use_jina:
            return await self.simple_reranker.rerank(query, documents, top_k)
        
        try:
            # 获取两种重排序结果
            jina_results = await self.jina_reranker.rerank(query, documents.copy())
            simple_results = await self.simple_reranker.rerank(query, documents.copy())
            
            # 混合分数
            blended_results = self._blend_rankings(jina_results, simple_results, blend_ratio)
            
            return blended_results[:top_k] if top_k else blended_results
            
        except Exception as e:
            print(f"❌ 双重排序失败: {str(e)}")
            return await self.simple_reranker.rerank(query, documents, top_k)
    
    def _blend_rankings(self, jina_results: List[Dict[str, Any]], 
                       simple_results: List[Dict[str, Any]], 
                       blend_ratio: float) -> List[Dict[str, Any]]:
        """
        混合两种排序结果
        
        Args:
            jina_results: Jina重排序结果
            simple_results: 简单重排序结果
            blend_ratio: Jina分数权重
            
        Returns:
            混合结果
        """
        # 创建ID到结果的映射
        jina_map = {doc.get('id', str(i)): doc for i, doc in enumerate(jina_results)}
        simple_map = {doc.get('id', str(i)): doc for i, doc in enumerate(simple_results)}
        
        all_ids = set(jina_map.keys()) | set(simple_map.keys())
        
        blended_results = []
        
        for doc_id in all_ids:
            jina_doc = jina_map.get(doc_id)
            simple_doc = simple_map.get(doc_id)
            
            # 获取分数
            jina_score = jina_doc.get('score', 0.0) if jina_doc else 0.0
            simple_score = simple_doc.get('score', 0.0) if simple_doc else 0.0
            
            # 混合分数
            blended_score = blend_ratio * jina_score + (1 - blend_ratio) * simple_score
            
            # 使用Jina结果作为基础（如果存在）
            base_doc = jina_doc if jina_doc else simple_doc
            
            if base_doc:
                blended_doc = base_doc.copy()
                blended_doc['score'] = blended_score
                blended_doc['jina_score'] = jina_score
                blended_doc['simple_score'] = simple_score
                blended_results.append(blended_doc)
        
        # 按混合分数排序
        blended_results.sort(key=lambda x: x['score'], reverse=True)
        
        return blended_results
    
    def get_reranker_info(self) -> Dict[str, Any]:
        """获取混合重排序器信息"""
        return {
            'type': 'hybrid',
            'jina_enabled': self.use_jina,
            'components': [
                self.jina_reranker.get_reranker_info(),
                self.simple_reranker.get_reranker_info()
            ],
            'modes': [
                'jina_primary',
                'simple_fallback', 
                'dual_blend'
            ]
        }
