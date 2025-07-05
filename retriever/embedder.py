"""
使用 Jina embedding 进行文本嵌入
"""
import numpy as np
from typing import List, Dict, Any, Optional
import requests
import json
from config import Config


class JinaEmbedder:
    """Jina嵌入模型封装"""
    
    def __init__(self):
        """初始化Jina嵌入器"""
        self.api_key = Config.JINA_API_KEY
        self.model = Config.JINA_EMBEDDING_MODEL
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.embedding_dim = Config.EMBEDDING_DIM
        self.enabled = bool(self.api_key)
        
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        对文本列表进行嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        if not self.enabled:
            print("⚠️ Jina API密钥未配置，使用随机向量")
            return [self._generate_random_embedding() for _ in texts]
        
        try:
            print(f"🔄 正在嵌入 {len(texts)} 个文本...")
            
            # 分批处理，避免单次请求过大
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = await self._embed_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)
            
            print(f"✅ 嵌入完成，生成 {len(all_embeddings)} 个向量")
            return all_embeddings
            
        except Exception as e:
            print(f"❌ 嵌入失败: {str(e)}")
            print("回退到随机向量")
            return [self._generate_random_embedding() for _ in texts]
    
    async def embed_single(self, text: str) -> List[float]:
        """
        对单个文本进行嵌入
        
        Args:
            text: 单个文本
            
        Returns:
            嵌入向量
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else self._generate_random_embedding()
    
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文本
        
        Args:
            texts: 文本批次
            
        Returns:
            嵌入向量列表
        """
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            'model': self.model,
            'input': texts,
            'encoding_format': 'float'
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
            
            result = response.json()
            
            if 'data' not in result:
                raise Exception("API响应格式错误")
            
            embeddings = []
            for item in result['data']:
                if 'embedding' in item:
                    embeddings.append(item['embedding'])
                else:
                    embeddings.append(self._generate_random_embedding())
            
            return embeddings
            
        except Exception as e:
            print(f"❌ 批量嵌入请求失败: {str(e)}")
            return [self._generate_random_embedding() for _ in texts]
    
    def _generate_random_embedding(self) -> List[float]:
        """
        生成随机嵌入向量（作为备用方案）
        
        Returns:
            随机嵌入向量
        """
        return np.random.normal(0, 1, self.embedding_dim).tolist()
    
    def normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        归一化嵌入向量
        
        Args:
            embeddings: 原始嵌入向量列表
            
        Returns:
            归一化后的嵌入向量列表
        """
        normalized = []
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized.append((np.array(embedding) / norm).tolist())
            else:
                normalized.append(embedding)
        return normalized
    
    def calculate_similarity(self, embedding1: List[float], 
                           embedding2: List[float]) -> float:
        """
        计算两个嵌入向量的余弦相似度
        
        Args:
            embedding1: 嵌入向量1
            embedding2: 嵌入向量2
            
        Returns:
            余弦相似度分数
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"❌ 相似度计算失败: {str(e)}")
            return 0.0
    
    def batch_similarity(self, query_embedding: List[float], 
                        doc_embeddings: List[List[float]]) -> List[float]:
        """
        批量计算查询与文档的相似度
        
        Args:
            query_embedding: 查询嵌入向量
            doc_embeddings: 文档嵌入向量列表
            
        Returns:
            相似度分数列表
        """
        try:
            query_vec = np.array(query_embedding)
            doc_matrix = np.array(doc_embeddings)
            
            # 批量计算余弦相似度
            dot_products = np.dot(doc_matrix, query_vec)
            query_norm = np.linalg.norm(query_vec)
            doc_norms = np.linalg.norm(doc_matrix, axis=1)
            
            # 避免除零
            valid_mask = (doc_norms != 0) & (query_norm != 0)
            similarities = np.zeros(len(doc_embeddings))
            
            if query_norm != 0:
                similarities[valid_mask] = dot_products[valid_mask] / (doc_norms[valid_mask] * query_norm)
            
            return similarities.tolist()
            
        except Exception as e:
            print(f"❌ 批量相似度计算失败: {str(e)}")
            return [0.0] * len(doc_embeddings)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """获取嵌入模型信息"""
        return {
            'model': self.model,
            'embedding_dim': self.embedding_dim,
            'api_enabled': self.enabled,
            'api_url': self.api_url,
            'capabilities': [
                '文本向量化',
                '语义相似度计算', 
                '批量处理',
                '向量归一化'
            ]
        }


class LocalEmbedder:
    """本地嵌入模型（备用方案）"""
    
    def __init__(self):
        """初始化本地嵌入器"""
        self.embedding_dim = Config.EMBEDDING_DIM
        self.word_embeddings = {}
        self._load_simple_embeddings()
    
    def _load_simple_embeddings(self):
        """加载简单的词嵌入（基于词频和位置）"""
        # 这里实现一个简单的词嵌入，实际项目中可能使用预训练模型
        common_words = [
            '人工智能', '机器学习', '深度学习', '神经网络', '算法',
            '数据', '模型', '训练', '预测', '分类', '回归', '聚类',
            '特征', '参数', '优化', '损失函数', '梯度', '反向传播',
            '卷积', '池化', '激活函数', '过拟合', '正则化', '交叉验证'
        ]
        
        for i, word in enumerate(common_words):
            # 生成基于位置的简单嵌入
            embedding = np.zeros(self.embedding_dim)
            embedding[i % self.embedding_dim] = 1.0
            embedding[(i + 1) % self.embedding_dim] = 0.5
            self.word_embeddings[word] = embedding.tolist()
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        使用简单方法嵌入文本
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        embeddings = []
        for text in texts:
            embedding = self._embed_simple(text)
            embeddings.append(embedding)
        return embeddings
    
    def _embed_simple(self, text: str) -> List[float]:
        """
        简单的文本嵌入方法
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        # 基于词汇的简单嵌入
        embedding = np.zeros(self.embedding_dim)
        
        words = text.split()
        for word in words:
            if word in self.word_embeddings:
                word_emb = np.array(self.word_embeddings[word])
                embedding += word_emb
        
        # 归一化
        if len(words) > 0:
            embedding /= len(words)
        
        # 添加一些基于文本统计的特征
        embedding[0] = len(text) / 1000.0  # 文本长度
        embedding[1] = len(words) / 100.0  # 词数
        
        return embedding.tolist()
    
    async def embed_single(self, text: str) -> List[float]:
        """嵌入单个文本"""
        return self._embed_simple(text)
