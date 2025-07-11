"""
Web搜索工具：使用Jina API读取Web内容
"""
import os
import json
import requests
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
        self.knowledge_base_dir = Config.KNOWLEDGE_BASE_DIR

    def _search_via_jina(self, query: str, count: int = 5) -> List[str]:
        """
        使用Jina API搜索Web内容
        
        Args:
            query: 搜索查询
            count: 返回结果的数量
            
        Returns:
            Web页面URL列表
        """
        try:
            # 构建Jina搜索URL
            search_url = f"https://s.jina.ai/?q={query.replace(' ', '+')}"
            headers = {
                "Authorization": f"Bearer {self.jina_api_key}",
                "X-Respond-With": "no-content",
                # "X-Site": "https://en.wikipedia.org/wiki/"
            }
            
            response = requests.get(search_url, headers=headers, timeout=30)
            if response.status_code == 200:
                text = response.text
                
                # 解析搜索结果，提取URL
                urls = []
                lines = text.split('\n')
                
                for line in lines:
                    if line.strip().startswith('[') and 'URL Source:' in line:
                        # 提取URL
                        url_start = line.find('https://')
                        if url_start != -1:
                            url = line[url_start:].strip()
                            urls.append(url)
                            
                            if len(urls) >= count:
                                break
                
                print(f"✅ 找到 {len(urls)} 个Web页面URL：{', '.join(urls)}")
                return urls
            else:
                print(f"⚠️ Jina搜索API返回状态码: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Jina搜索失败: {str(e)}")
            return []
       
    
    
    def _get_content_via_jina(self, url: str) -> str:
        """
        使用Jina API获取网页内容
        
        Args:
            url: 网页URL
            
        Returns:
            网页内容
        """
        try:
            # 如果url是英文wiki，转换为中文wiki/镜像wiki
            if "en.wikipedia.org" in url:
                # url = url.replace("en.wikipedia.org", "zh.wikipedia.org/wiki/")
                url = url.replace("en.wikipedia.org/wiki/", "encyclopedia.thefreedictionary.com/")
                print(f"🔄 转换为镜像Wikipedia: {url}")
            
            jina_url = f"https://r.jina.ai/{url}"
            headers = {
                # 'Authorization': f'Bearer {self.jina_api_key}'
            }
            
            response = requests.get(jina_url, headers=headers, timeout=30)
            if response.status_code == 200:
                content = response.text
                print(f"✅ 通过Jina API获取内容，长度: {len(content)}")
                # 提取URL中https://后面的内容，并替换特殊字符为下划线
                title = url.split("https://")[-1].replace(".", "_").replace("/", "_").replace(" ", "_")
                self._save_to_knowledge_base(title, content)
                return content
            else:
                print(f"⚠️ Jina API返回状态码: {response.status_code}")
                return f"无法通过Jina API获取内容: HTTP {response.status_code}"
                        
        except Exception as e:
            print(f"❌ Jina API调用出错: {str(e)}")
            return f"Jina API调用失败: {str(e)}"
        
    
    def _save_to_knowledge_base(self, title: str, content: str) -> Optional[str]:
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
            
            filename = f"{title}.txt"
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
    urls = web_search_tool._search_via_jina("Artificial Intelligence", count=3)
    print(f"搜索结果: {urls}")
    
    # 测试获取内容
    content = web_search_tool._get_content_via_jina("https://en.wikipedia.org/wiki/Artificial_intelligence")
    print(f"内容预览: {content[:200]}...")