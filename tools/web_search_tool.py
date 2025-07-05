"""
互联网搜索工具：可选接入 Bing API
"""
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from config import Config


class WebSearchTool:
    """网络搜索工具"""
    
    def __init__(self):
        """初始化网络搜索工具"""
        self.bing_api_key = Config.BING_API_KEY
        self.enabled = Config.ENABLE_WEB_SEARCH and bool(self.bing_api_key)
        self.bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    async def search(self, query: str, count: int = 10, 
                    market: str = "zh-CN") -> List[Dict[str, Any]]:
        """
        执行网络搜索
        
        Args:
            query: 搜索查询
            count: 返回结果数量
            market: 搜索市场（语言和地区）
            
        Returns:
            搜索结果列表
        """
        if not self.enabled:
            return [{
                'title': '网络搜索功能未启用',
                'content': '请配置BING_API_KEY以启用网络搜索功能',
                'url': '',
                'source': 'web_search_disabled',
                'error': True
            }]
        
        try:
            print(f"🌐 在网络上搜索: {query}")
            
            # 构建搜索参数
            params = {
                'q': query,
                'count': count,
                'mkt': market,
                'responseFilter': 'webpages',
                'textDecorations': False,
                'textFormat': 'Raw'
            }
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.bing_api_key,
                'User-Agent': 'Multi-Agent Research System/1.0'
            }
            
            # 发起搜索请求
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.bing_endpoint, 
                    params=params, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        raise Exception(f"Bing API返回错误状态码: {response.status}")
                    
                    data = await response.json()
                    return self._parse_bing_results(data)
                    
        except Exception as e:
            print(f"❌ 网络搜索出错: {str(e)}")
            return [{
                'title': '网络搜索错误',
                'content': f'搜索过程中出现错误: {str(e)}',
                'url': '',
                'source': 'web_search_error',
                'error': True
            }]
    
    def _parse_bing_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        解析Bing搜索结果
        
        Args:
            data: Bing API返回的原始数据
            
        Returns:
            格式化的搜索结果
        """
        results = []
        
        if 'webPages' not in data or 'value' not in data['webPages']:
            return results
        
        for item in data['webPages']['value']:
            result = {
                'title': item.get('name', ''),
                'content': item.get('snippet', ''),
                'url': item.get('url', ''),
                'source': 'web_search',
                'display_url': item.get('displayUrl', ''),
                'last_crawled': item.get('dateLastCrawled', ''),
                'score': 1.0  # Bing不提供相关性评分，统一设为1.0
            }
            results.append(result)
        
        print(f"✅ 找到 {len(results)} 个网络搜索结果")
        return results
    
    async def search_news(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        搜索新闻
        
        Args:
            query: 搜索查询
            count: 返回结果数量
            
        Returns:
            新闻搜索结果
        """
        if not self.enabled:
            return []
        
        try:
            news_endpoint = "https://api.bing.microsoft.com/v7.0/news/search"
            
            params = {
                'q': query,
                'count': count,
                'mkt': 'zh-CN',
                'sortBy': 'Date'
            }
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.bing_api_key,
                'User-Agent': 'Multi-Agent Research System/1.0'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    news_endpoint,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    return self._parse_news_results(data)
                    
        except Exception as e:
            print(f"❌ 新闻搜索出错: {str(e)}")
            return []
    
    def _parse_news_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析新闻搜索结果"""
        results = []
        
        if 'value' not in data:
            return results
        
        for item in data['value']:
            result = {
                'title': item.get('name', ''),
                'content': item.get('description', ''),
                'url': item.get('url', ''),
                'source': 'news_search',
                'published_time': item.get('datePublished', ''),
                'provider': item.get('provider', [{}])[0].get('name', '') if item.get('provider') else '',
                'score': 1.0
            }
            results.append(result)
        
        return results
    
    async def search_academic(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        搜索学术资源（通过特定的学术网站）
        
        Args:
            query: 搜索查询
            count: 返回结果数量
            
        Returns:
            学术搜索结果
        """
        # 添加学术相关的搜索词
        academic_query = f"{query} site:arxiv.org OR site:scholar.google.com OR site:researchgate.net OR site:ieee.org"
        
        results = await self.search(academic_query, count)
        
        # 标记为学术来源
        for result in results:
            result['source'] = 'academic_search'
        
        return results
    
    async def verify_url(self, url: str) -> bool:
        """
        验证URL是否可访问
        
        Args:
            url: 待验证的URL
            
        Returns:
            URL是否可访问
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except:
            return False
    
    def get_tool_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            'name': 'web_search',
            'description': '在互联网上搜索最新信息和资源',
            'parameters': {
                'query': '搜索查询字符串',
                'count': '返回结果数量（可选，默认10）',
                'search_type': '搜索类型：web/news/academic（可选）'
            },
            'example_usage': 'web_search("2024年人工智能最新进展")',
            'capabilities': [
                '网页搜索',
                '新闻搜索', 
                '学术资源搜索',
                'URL验证'
            ],
            'enabled': self.enabled
        }
    
    async def get_page_content(self, url: str, max_length: int = 2000) -> str:
        """
        获取网页内容（简单版本，实际项目中可能需要更复杂的解析）
        
        Args:
            url: 网页URL
            max_length: 最大内容长度
            
        Returns:
            网页文本内容
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15),
                    headers={
                        'User-Agent': 'Multi-Agent Research System/1.0'
                    }
                ) as response:
                    
                    if response.status != 200:
                        return ""
                    
                    content = await response.text()
                    
                    # 简单的HTML标签清理（实际项目中建议使用BeautifulSoup）
                    import re
                    text = re.sub(r'<[^>]+>', '', content)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    return text[:max_length] if len(text) > max_length else text
                    
        except Exception as e:
            print(f"❌ 获取网页内容出错: {str(e)}")
            return ""
