"""
Web搜索工具：使用Jina API读取Web内容
"""
import asyncio
import aiohttp
import os
import json
from typing import List, Dict, Any, Optional
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class WebSearchTool:
    """Web搜索工具"""
    
    def __init__(self):
        """初始化Web搜索工具"""
        self.jina_api_key = Config.JINA_API_KEY
        self.enabled = Config.ENABLE_WEB_SEARCH and bool(self.jina_api_key)
        self.jina_reader_endpoint = "https://r.jina.ai/"
        self.knowledge_base_dir = Config.KNOWLEDGE_BASE_DIR

    async def search(self, query: str, count: int = 5) -> List[str]:
        """
        在Wikipedia上搜索内容并返回相关页面链接
        
        Args:
            query: 搜索查询
            count: 返回结果的数量（默认5个）
            
        Returns:
            Wikipedia页面链接列表
        """
        try:
            # 使用Wikipedia API搜索
            opensearch_url = "https://en.wikipedia.org/w/api.php"
            
            # 搜索参数
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': count,
                'format': 'json',
                'redirects': 'resolve'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(opensearch_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Wikipedia OpenSearch API返回格式：
                        # [query, [titles], [descriptions], [urls]]
                        if len(data) >= 4 and data[3]:
                            urls = data[3]
                            print(f"✅ 找到 {len(urls)} 个Wikipedia页面")
                            return urls[:count]
                        else:
                            print(f"⚠️ 未找到相关Wikipedia页面")
                            return []
                    else:
                        print(f"⚠️ Wikipedia API返回状态码: {response.status}")
                        return []
                        
        except Exception as e:
            print(f"❌ Wikipedia搜索失败: {str(e)}")
            return []
    
    
    async def _get_content_via_jina(self, url: str) -> str:
        """
        使用Jina API获取网页内容
        
        Args:
            url: 网页URL
            
        Returns:
            网页内容
        """
        try:
            jina_url = f"{self.jina_reader_endpoint}{url}"
            headers = {
                'Authorization': f'Bearer {self.jina_api_key}'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(jina_url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        print(f"✅ 通过Jina API获取内容，长度: {len(content)}")
                        title = url.split("/")[-1]
                        await self._save_to_knowledge_base(title, content)
                        return content
                    else:
                        print(f"⚠️ Jina API返回状态码: {response.status}")
                        return f"无法通过Jina API获取内容: HTTP {response.status}"
                        
        except Exception as e:
            print(f"❌ Jina API调用出错: {str(e)}")
            return f"Jina API调用失败: {str(e)}"
        
    
    async def _save_to_knowledge_base(self, title: str, content: str) -> Optional[str]:
        """
        将内容保存到知识库
        
        Args:
            title: 文档标题
            content: 文档内容
            
        Returns:
            保存的文件路径
        """
        try:
            # 确保知识库目录存在
            os.makedirs(self.knowledge_base_dir, exist_ok=True)
            
            filename = f"wikipedia_{title}.txt"
            filepath = os.path.join(self.knowledge_base_dir, filename)
            
            # 检查文件是否已存在
            if os.path.exists(filepath):
                print(f"📄 文件已存在: {filename}")
                return filepath
            
            # 准备文档内容（包含元数据）
            document_content = f"{content}"

            
            # 保存文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(document_content)
            
            print(f"💾 已保存到知识库: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ 保存到知识库失败: {str(e)}")
            return None
    

if __name__ == "__main__":
    # 测试工具
    web_search_tool = WebSearchTool()

    # 测试搜索功能
    asyncio.run(web_search_tool.search("Artificial Intelligence", count=3))
    
    # 测试获取内容
    asyncio.run(web_search_tool._get_content_via_jina("https://en.wikipedia.org/wiki/Artificial_intelligence"))