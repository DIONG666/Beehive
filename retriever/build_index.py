"""
构建 FAISS 索引脚本：从知识库加载
"""
import os
import json
import pickle
from typing import List, Dict, Any, Optional
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class IndexBuilder:
    """FAISS索引构建器"""
    
    def __init__(self):
        """初始化索引构建器"""
        self.config = Config()
        self.embedder = None
        self.documents = []
        self.embeddings = []
        self.index = None
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化组件"""
        try:
            from embedder import JinaEmbedder
            self.embedder = JinaEmbedder()
        except ImportError as e:
            print(f"⚠️ 警告: 无法导入嵌入器 - {e}")
            print("请确保已安装相关依赖")
    
    async def build_index_from_directory(self, data_dir: str, 
                                       file_patterns: List[str] = None) -> bool:
        """
        从目录构建索引
        
        Args:
            data_dir: 数据目录路径
            file_patterns: 文件模式列表
            
        Returns:
            构建是否成功
        """
        try:
            print(f"📁 开始从目录构建索引: {data_dir}")
            
            if not os.path.exists(data_dir):
                print(f"❌ 数据目录不存在: {data_dir}")
                return False
            
            # 加载文档
            documents = await self._load_documents_from_directory(data_dir, file_patterns)
            
            if not documents:
                print("❌ 未找到任何文档")
                return False
            
            # 构建索引
            return await self.build_index_from_documents(documents)
            
        except Exception as e:
            print(f"❌ 构建索引失败: {str(e)}")
            return False
    
    async def _load_documents_from_directory(self, data_dir: str, 
                                           file_patterns: List[str] = None) -> List[Dict[str, Any]]:
        """
        从目录加载文档
        
        Args:
            data_dir: 数据目录
            file_patterns: 文件模式
            
        Returns:
            文档列表
        """
        documents = []
        file_patterns = file_patterns or ['*.txt', '*.json', '*.md']
        
        print(f"🔍 扫描目录: {data_dir}")
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # 检查文件扩展名
                if any(file.endswith(pattern.replace('*', '')) for pattern in file_patterns):
                    try:
                        doc = await self._load_single_document(file_path)
                        if doc:
                            documents.append(doc)
                    except Exception as e:
                        print(f"⚠️ 加载文件失败 {file_path}: {str(e)}")
                        continue
        
        print(f"📄 加载了 {len(documents)} 个文档")
        return documents
    
    async def _load_single_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        加载单个文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return None
            
            # 根据文件类型处理
            if file_path.endswith('.json'):
                data = json.loads(content)
                if isinstance(data, dict):
                    return {
                        'id': data.get('id', os.path.basename(file_path)),
                        'content': data.get('content', ''),
                        'title': data.get('title', ''),
                        'source': file_path,
                        'metadata': data.get('metadata', {})
                    }
                else:
                    return {
                        'id': os.path.basename(file_path),
                        'content': json.dumps(data, ensure_ascii=False),
                        'title': os.path.basename(file_path),
                        'source': file_path,
                        'metadata': {}
                    }
            else:
                return {
                    'id': os.path.basename(file_path),
                    'content': content,
                    'title': os.path.basename(file_path),
                    'source': file_path,
                    'metadata': {'file_size': len(content)}
                }
                
        except Exception as e:
            print(f"❌ 读取文件失败 {file_path}: {str(e)}")
            return None
    
    async def build_index_from_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        从文档列表构建索引
        
        Args:
            documents: 文档列表
            
        Returns:
            构建是否成功
        """
        try:
            print(f"🔨 开始构建索引，文档数量: {len(documents)}")
            
            self.documents = documents
            
            # 准备文本用于嵌入
            texts = []
            for doc in documents:
                text = doc.get('content', '')
                if doc.get('title'):
                    text = doc['title'] + '\n' + text
                texts.append(text)
            
            # 生成嵌入
            if not self.embedder:
                print("❌ 嵌入器未初始化")
                return False
            
            print("🔄 生成文档嵌入...")
            embeddings = await self.embedder.embed_texts(texts)
            
            if not embeddings:
                print("❌ 嵌入生成失败")
                return False
            
            self.embeddings = embeddings
            
            # 构建FAISS索引
            success = self._build_faiss_index()
            
            if success:
                # 保存索引
                self._save_index()
                print("✅ 索引构建完成")
                return True
            else:
                print("❌ FAISS索引构建失败")
                return False
            
        except Exception as e:
            print(f"❌ 构建索引过程失败: {str(e)}")
            return False
    
    def _build_faiss_index(self) -> bool:
        """
        构建FAISS索引
        
        Returns:
            构建是否成功
        """
        try:
            # 尝试导入FAISS
            try:
                import faiss
                import numpy as np
            except ImportError:
                print("⚠️ FAISS未安装，使用简单索引")
                return self._build_simple_index()
            
            print("🔧 构建FAISS索引...")
            
            # 转换嵌入为numpy数组
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            
            # 创建FAISS索引
            dimension = embeddings_array.shape[1]
            
            # 使用L2距离的平面索引
            index = faiss.IndexFlatL2(dimension)
            
            # 添加向量到索引
            index.add(embeddings_array)
            
            self.index = index
            print(f"✅ FAISS索引构建完成，包含 {index.ntotal} 个向量")
            return True
            
        except Exception as e:
            print(f"❌ FAISS索引构建失败: {str(e)}")
            return self._build_simple_index()
    
    def _build_simple_index(self) -> bool:
        """
        构建简单索引（备用方案）
        
        Returns:
            构建是否成功
        """
        try:
            print("🔧 构建简单索引...")
            
            # 简单索引就是存储所有嵌入向量
            self.index = {
                'embeddings': self.embeddings,
                'type': 'simple'
            }
            
            print("✅ 简单索引构建完成")
            return True
            
        except Exception as e:
            print(f"❌ 简单索引构建失败: {str(e)}")
            return False
    
    def _save_index(self):
        """保存索引到文件"""
        try:
            print("💾 保存索引...")
            
            # 确保目录存在
            os.makedirs(Config.INDEX_DIR, exist_ok=True)
            
            # 保存文档元数据
            docs_path = os.path.join(Config.INDEX_DIR, 'documents.pkl')
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # 保存索引
            if hasattr(self.index, 'write_index'):
                # FAISS索引
                index_path = Config.FAISS_INDEX_PATH
                import faiss
                faiss.write_index(self.index, index_path)
            else:
                # 简单索引
                index_path = os.path.join(Config.INDEX_DIR, 'simple_index.pkl')
                with open(index_path, 'wb') as f:
                    pickle.dump(self.index, f)
            
            # 保存配置信息
            config_path = os.path.join(Config.INDEX_DIR, 'index_config.json')
            index_config = {
                'num_documents': len(self.documents),
                'embedding_dim': Config.EMBEDDING_DIM,
                'model': Config.JINA_EMBEDDING_MODEL,
                'index_type': 'faiss' if hasattr(self.index, 'write_index') else 'simple',
                'created_at': str(self._get_current_time())
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(index_config, f, ensure_ascii=False, indent=2)
            
            print("✅ 索引保存完成")
            
        except Exception as e:
            print(f"❌ 保存索引失败: {str(e)}")
    
    def _get_current_time(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now()
    
    async def update_index(self, new_documents: List[Dict[str, Any]]) -> bool:
        """
        更新索引（添加新文档）
        
        Args:
            new_documents: 新文档列表
            
        Returns:
            更新是否成功
        """
        try:
            print(f"🔄 更新索引，新增 {len(new_documents)} 个文档")
            
            # 生成新文档的嵌入
            new_texts = []
            for doc in new_documents:
                text = doc.get('content', '')
                if doc.get('title'):
                    text = doc['title'] + '\n' + text
                new_texts.append(text)
            
            new_embeddings = await self.embedder.embed_texts(new_texts)
            
            # 更新文档和嵌入列表
            self.documents.extend(new_documents)
            self.embeddings.extend(new_embeddings)
            
            # 重新构建索引
            success = self._build_faiss_index()
            
            if success:
                self._save_index()
                print("✅ 索引更新完成")
                return True
            else:
                print("❌ 索引更新失败")
                return False
                
        except Exception as e:
            print(f"❌ 更新索引失败: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            'num_documents': len(self.documents),
            'embedding_dim': len(self.embeddings[0]) if self.embeddings else 0,
            'index_type': 'faiss' if hasattr(self.index, 'ntotal') else 'simple',
            'total_vectors': self.index.ntotal if hasattr(self.index, 'ntotal') else len(self.embeddings)
        }


async def main():
    """主函数，用于命令行构建索引"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建FAISS索引")
    parser.add_argument("--data-dir", required=True, help="数据目录路径")
    parser.add_argument("--patterns", nargs="+", default=['*.txt', '*.json', '*.md'], 
                       help="文件模式")
    
    args = parser.parse_args()
    
    builder = IndexBuilder()
    success = await builder.build_index_from_directory(args.data_dir, args.patterns)
    
    if success:
        stats = builder.get_index_stats()
        print(f"\n📊 索引统计: {stats}")
        print("🎉 索引构建成功！")
    else:
        print("💥 索引构建失败！")
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
