"""
引用文献跟踪与整理
"""
import re
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
from datetime import datetime


class CitationManager:
    """引用管理器"""
    
    def __init__(self):
        """初始化引用管理器"""
        self.citations = []
        self.citation_map = {}  # ID到引用的映射
        self.next_citation_id = 1
    
    def add_citation(self, source: str, content: str, title: str = "", 
                    metadata: Dict[str, Any] = None) -> str:
        """
        添加引用
        
        Args:
            source: 来源URL或文件路径
            content: 引用内容
            title: 标题
            metadata: 元数据
            
        Returns:
            引用ID
        """
        citation_id = f"cite_{self.next_citation_id}"
        self.next_citation_id += 1
        
        citation = {
            'id': citation_id,
            'source': source,
            'content': content,
            'title': title,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'citation_format': self._generate_citation_format(source, title, metadata)
        }
        
        self.citations.append(citation)
        self.citation_map[citation_id] = citation
        
        return citation_id
    
    def _generate_citation_format(self, source: str, title: str = "", 
                                metadata: Dict[str, Any] = None) -> str:
        """
        生成标准引用格式
        
        Args:
            source: 来源
            title: 标题
            metadata: 元数据
            
        Returns:
            格式化的引用
        """
        metadata = metadata or {}
        
        # 判断来源类型
        if self._is_url(source):
            return self._format_web_citation(source, title, metadata)
        elif source.endswith(('.pdf', '.doc', '.docx')):
            return self._format_document_citation(source, title, metadata)
        else:
            return self._format_generic_citation(source, title, metadata)
    
    def _is_url(self, source: str) -> bool:
        """判断是否为URL"""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _format_web_citation(self, url: str, title: str, metadata: Dict[str, Any]) -> str:
        """格式化网页引用"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # 基本格式：标题. 网站名称. 访问日期. URL
        citation_parts = []
        
        if title:
            citation_parts.append(f'"{title}"')
        
        citation_parts.append(domain)
        
        # 添加访问日期
        access_date = metadata.get('access_date', datetime.now().strftime('%Y-%m-%d'))
        citation_parts.append(f"访问日期: {access_date}")
        
        citation_parts.append(url)
        
        return '. '.join(citation_parts)
    
    def _format_document_citation(self, file_path: str, title: str, metadata: Dict[str, Any]) -> str:
        """格式化文档引用"""
        file_name = file_path.split('/')[-1] if '/' in file_path else file_path
        
        citation_parts = []
        
        if title:
            citation_parts.append(f'"{title}"')
        
        # 添加作者（如果有）
        author = metadata.get('author', '')
        if author:
            citation_parts.append(f"作者: {author}")
        
        citation_parts.append(f"文档: {file_name}")
        
        # 添加页码（如果有）
        page = metadata.get('page', '')
        if page:
            citation_parts.append(f"第{page}页")
        
        return '. '.join(citation_parts)
    
    def _format_generic_citation(self, source: str, title: str, metadata: Dict[str, Any]) -> str:
        """格式化通用引用"""
        citation_parts = []
        
        if title:
            citation_parts.append(f'"{title}"')
        
        citation_parts.append(f"来源: {source}")
        
        return '. '.join(citation_parts)
    
    def get_citation(self, citation_id: str) -> Optional[Dict[str, Any]]:
        """
        获取引用
        
        Args:
            citation_id: 引用ID
            
        Returns:
            引用字典
        """
        return self.citation_map.get(citation_id)
    
    def get_all_citations(self) -> List[Dict[str, Any]]:
        """
        获取所有引用
        
        Returns:
            引用列表
        """
        return self.citations.copy()
    
    def format_citations_list(self, style: str = 'apa') -> str:
        """
        格式化引用列表
        
        Args:
            style: 引用风格（apa/mla/chicago）
            
        Returns:
            格式化的引用列表
        """
        if not self.citations:
            return "无引用文献。"
        
        formatted_citations = []
        
        for i, citation in enumerate(self.citations, 1):
            if style.lower() == 'apa':
                formatted = self._format_apa_citation(citation, i)
            elif style.lower() == 'mla':
                formatted = self._format_mla_citation(citation, i)
            elif style.lower() == 'chicago':
                formatted = self._format_chicago_citation(citation, i)
            else:
                formatted = f"[{i}] {citation['citation_format']}"
            
            formatted_citations.append(formatted)
        
        return '\n'.join(formatted_citations)
    
    def _format_apa_citation(self, citation: Dict[str, Any], number: int) -> str:
        """APA格式引用"""
        title = citation.get('title', '')
        source = citation.get('source', '')
        metadata = citation.get('metadata', {})
        
        if self._is_url(source):
            # 网页 APA 格式
            author = metadata.get('author', '作者未知')
            year = metadata.get('year', datetime.now().year)
            return f"[{number}] {author}. ({year}). {title}. 检索自 {source}"
        else:
            # 文档 APA 格式
            return f"[{number}] {citation['citation_format']}"
    
    def _format_mla_citation(self, citation: Dict[str, Any], number: int) -> str:
        """MLA格式引用"""
        return f"[{number}] {citation['citation_format']}"
    
    def _format_chicago_citation(self, citation: Dict[str, Any], number: int) -> str:
        """芝加哥格式引用"""
        return f"[{number}] {citation['citation_format']}"
    
    def extract_citations_from_text(self, text: str) -> List[str]:
        """
        从文本中提取引用标记
        
        Args:
            text: 输入文本
            
        Returns:
            引用ID列表
        """
        # 查找形如 [cite_1] 或 [1] 的引用标记
        citation_pattern = r'\[(?:cite_)?(\d+)\]'
        matches = re.findall(citation_pattern, text)
        
        citation_ids = []
        for match in matches:
            citation_id = f"cite_{match}" if not match.startswith('cite_') else match
            citation_ids.append(citation_id)
        
        return citation_ids
    
    def insert_citations_in_text(self, text: str, search_results: List[Dict[str, Any]]) -> str:
        """
        在文本中插入引用标记
        
        Args:
            text: 原始文本
            search_results: 搜索结果（用于生成引用）
            
        Returns:
            包含引用标记的文本
        """
        # 为搜索结果生成引用
        result_citations = {}
        
        for i, result in enumerate(search_results):
            citation_id = self.add_citation(
                source=result.get('source', ''),
                content=result.get('content', '')[:200] + '...',
                title=result.get('title', ''),
                metadata=result.get('metadata', {})
            )
            result_citations[i] = citation_id
        
        # 在文本中智能插入引用标记
        enhanced_text = self._smart_insert_citations(text, search_results, result_citations)
        
        return enhanced_text
    
    def _smart_insert_citations(self, text: str, search_results: List[Dict[str, Any]], 
                              result_citations: Dict[int, str]) -> str:
        """
        智能地在文本中插入引用标记
        
        Args:
            text: 原始文本
            search_results: 搜索结果
            result_citations: 结果到引用ID的映射
            
        Returns:
            包含引用的文本
        """
        enhanced_text = text
        
        # 简单策略：在句子末尾添加最相关的引用
        sentences = re.split(r'[。！？]', text)
        enhanced_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # 找到最相关的搜索结果
            best_match_idx = self._find_best_citation_match(sentence, search_results)
            
            if best_match_idx is not None and best_match_idx in result_citations:
                citation_id = result_citations[best_match_idx]
                citation_number = citation_id.replace('cite_', '')
                enhanced_sentence = f"{sentence.strip()}[{citation_number}]。"
            else:
                enhanced_sentence = sentence.strip() + '。'
            
            enhanced_sentences.append(enhanced_sentence)
        
        return ''.join(enhanced_sentences)
    
    def _find_best_citation_match(self, sentence: str, search_results: List[Dict[str, Any]]) -> Optional[int]:
        """
        找到与句子最匹配的搜索结果
        
        Args:
            sentence: 句子
            search_results: 搜索结果列表
            
        Returns:
            最佳匹配的结果索引
        """
        sentence_words = set(sentence.lower().split())
        best_score = 0
        best_idx = None
        
        for i, result in enumerate(search_results):
            content = result.get('content', '').lower()
            title = result.get('title', '').lower()
            
            # 计算词汇重叠度
            content_words = set(content.split())
            title_words = set(title.split())
            
            content_overlap = len(sentence_words & content_words)
            title_overlap = len(sentence_words & title_words) * 2  # 标题权重更高
            
            score = content_overlap + title_overlap
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        # 只有当重叠度足够高时才返回匹配
        return best_idx if best_score >= 2 else None
    
    def validate_citations(self) -> Dict[str, List[str]]:
        """
        验证引用的有效性
        
        Returns:
            验证结果
        """
        results = {
            'valid': [],
            'invalid': [],
            'warnings': []
        }
        
        for citation in self.citations:
            citation_id = citation['id']
            source = citation['source']
            
            if self._is_url(source):
                # 验证URL格式
                if self._validate_url_format(source):
                    results['valid'].append(citation_id)
                else:
                    results['invalid'].append(citation_id)
            else:
                # 验证文件路径
                if source and len(source) > 0:
                    results['valid'].append(citation_id)
                else:
                    results['invalid'].append(citation_id)
            
            # 检查警告
            if not citation.get('title'):
                results['warnings'].append(f"{citation_id}: 缺少标题")
            
            if len(citation.get('content', '')) < 10:
                results['warnings'].append(f"{citation_id}: 引用内容过短")
        
        return results
    
    def _validate_url_format(self, url: str) -> bool:
        """验证URL格式"""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False
    
    def remove_duplicate_citations(self) -> int:
        """
        移除重复的引用
        
        Returns:
            移除的重复引用数量
        """
        seen_sources = set()
        unique_citations = []
        removed_count = 0
        
        for citation in self.citations:
            source = citation['source']
            if source not in seen_sources:
                seen_sources.add(source)
                unique_citations.append(citation)
            else:
                removed_count += 1
        
        # 重建引用映射
        self.citations = unique_citations
        self.citation_map = {c['id']: c for c in unique_citations}
        
        return removed_count
    
    def export_citations(self, format: str = 'bibtex') -> str:
        """
        导出引用为指定格式
        
        Args:
            format: 导出格式（bibtex/endnote/ris）
            
        Returns:
            格式化的引用数据
        """
        if format.lower() == 'bibtex':
            return self._export_bibtex()
        elif format.lower() == 'endnote':
            return self._export_endnote()
        elif format.lower() == 'ris':
            return self._export_ris()
        else:
            return self.format_citations_list()
    
    def _export_bibtex(self) -> str:
        """导出BibTeX格式"""
        bibtex_entries = []
        
        for citation in self.citations:
            entry_type = 'misc'
            if self._is_url(citation['source']):
                entry_type = 'online'
            
            bibtex_entry = f"""@{entry_type}{{{citation['id']},
    title = {{{citation.get('title', 'Untitled')}}},
    url = {{{citation['source']}}},
    note = {{访问时间: {citation['timestamp'].split('T')[0]}}}
}}"""
            bibtex_entries.append(bibtex_entry)
        
        return '\n\n'.join(bibtex_entries)
    
    def _export_endnote(self) -> str:
        """导出EndNote格式"""
        # 简化的EndNote格式
        endnote_entries = []
        
        for citation in self.citations:
            entry = f"""
%0 Web Page
%T {citation.get('title', 'Untitled')}
%U {citation['source']}
%D {citation['timestamp'].split('T')[0]}
"""
            endnote_entries.append(entry)
        
        return '\n'.join(endnote_entries)
    
    def _export_ris(self) -> str:
        """导出RIS格式"""
        ris_entries = []
        
        for citation in self.citations:
            entry = f"""TY  - ELEC
TI  - {citation.get('title', 'Untitled')}
UR  - {citation['source']}
DA  - {citation['timestamp'].split('T')[0]}
ER  - 
"""
            ris_entries.append(entry)
        
        return '\n'.join(ris_entries)
    
    def get_citation_stats(self) -> Dict[str, Any]:
        """
        获取引用统计信息
        
        Returns:
            统计信息
        """
        total_citations = len(self.citations)
        
        if total_citations == 0:
            return {
                'total_citations': 0,
                'url_citations': 0,
                'document_citations': 0,
                'citations_with_titles': 0,
                'average_content_length': 0
            }
        
        url_count = sum(1 for c in self.citations if self._is_url(c['source']))
        doc_count = total_citations - url_count
        titled_count = sum(1 for c in self.citations if c.get('title'))
        total_content_length = sum(len(c.get('content', '')) for c in self.citations)
        
        return {
            'total_citations': total_citations,
            'url_citations': url_count,
            'document_citations': doc_count,
            'citations_with_titles': titled_count,
            'average_content_length': total_content_length / total_citations
        }
