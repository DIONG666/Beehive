"""
配置文件：包含模型路径、知识库路径、工具启用项等
"""
import os
from typing import Dict, Any

class Config:
    """系统配置类"""
    
    # 模型配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-33d0f52208a84f07812feccf3ede2f43")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DEEPSEEK_MODEL = "deepseek-reasoner"
    
    # Jina API配置
    JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_a5852c23a34549f5b5735cc313253719ejBZJXB9xF-RZi9Ht4Bn2NAhvkeC")
    JINA_EMBEDDING_MODEL = "jina-embeddings-v4"
    JINA_RERANKER_MODEL = "jina-reranker-m0"
    
    # 项目路径配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, "knowledge_base")
    INDEX_DIR = os.path.join(DATA_DIR, "index")
    MEMORY_CACHE_DIR = os.path.join(DATA_DIR, "memory_cache")
    
    # 检索配置
    FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
    EMBEDDING_DIM = 2048
    TOP_K = 20
    RERANK_TOP_K = 5
    
    # Agent配置
    MAX_ITERATIONS = 3
    MAX_CONTEXT_LENGTH = 8192
    TEMPERATURE = 0.7
    RECENT_CONTEXT = 1
    
    # 工具启用配置
    ENABLE_WEB_SEARCH = True
    ENABLE_SUMMARIZER = True
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            attr: getattr(cls, attr) 
            for attr in dir(cls) 
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }
    
    @classmethod
    def setup_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.DATA_DIR,
            cls.KNOWLEDGE_BASE_DIR,
            cls.INDEX_DIR,
            cls.MEMORY_CACHE_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# 初始化目录
Config.setup_directories()
