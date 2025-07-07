"""
存储每轮对话与中间检索结果
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from config import Config


@dataclass
class MemoryEntry:
    """内存条目"""
    id: str
    timestamp: str
    query: str
    context: str
    final_answer: str


class MemoryManager:
    """内存管理器"""
    
    def __init__(self):
        """初始化内存管理器"""
        self.config = Config()
        self.memory_file = os.path.join(Config.MEMORY_CACHE_DIR, 'memory.json')
        self.session_file = os.path.join(Config.MEMORY_CACHE_DIR, 'current_session.json')
        self.memory_entries = []
        self.current_session = {}
        self._load_memory()
        self._initialize_session()
    
    def _load_memory(self):
        """加载历史内存"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memory_entries = [MemoryEntry(**entry) for entry in data]
                print(f"📚 加载了 {len(self.memory_entries)} 条历史记录")
            else:
                print("📚 未找到历史记录文件，创建新的内存存储")
                self.memory_entries = []
        except Exception as e:
            print(f"❌ 加载内存失败: {str(e)}")
            self.memory_entries = []
    
    def _initialize_session(self):
        """初始化当前会话"""
        session_id = self._generate_session_id()
        self.current_session = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'queries': [],
            'total_queries': 0
        }
        self._save_session()
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def add_memory_entry(self, query: str, context: str, final_answer: str) -> str:
        """
        添加内存条目
        
        Args:
            query: 查询
            context: 上下文
            final_answer: 最终答案
            
        Returns:
            内存条目ID
        """
        try:
            entry_id = self._generate_entry_id()
            timestamp = datetime.now().isoformat()
            
            entry = MemoryEntry(
                id=entry_id,
                timestamp=timestamp,
                query=query,
                context=context,
                final_answer=final_answer
            )
            
            self.memory_entries.append(entry)
            
            # 更新当前会话
            self.current_session['queries'].append({
                'entry_id': entry_id,
                'query': query,
                'timestamp': timestamp
            })
            self.current_session['total_queries'] += 1
            
            # 保存内存
            self._save_memory()
            self._save_session()
            
            print(f"💾 保存内存条目: {entry_id}")
            return entry_id
            
        except Exception as e:
            print(f"❌ 保存内存条目失败: {str(e)}")
            return ""
    
    def _generate_entry_id(self) -> str:
        """生成条目ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"entry_{timestamp}"
    
    def get_memory_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        获取内存条目
        
        Args:
            entry_id: 条目ID
            
        Returns:
            内存条目
        """
        for entry in self.memory_entries:
            if entry.id == entry_id:
                return entry
        return None
    
    def search_memory(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """
        搜索相关的历史记录
        
        Args:
            query: 查询字符串
            limit: 返回数量限制
            
        Returns:
            相关的内存条目列表
        """
        try:
            query_words = set(query.lower().split())
            scored_entries = []
            
            for entry in self.memory_entries:
                score = self._calculate_memory_relevance(entry, query_words)
                if score > 0:
                    scored_entries.append((entry, score))
            
            # 按相关性排序
            scored_entries.sort(key=lambda x: x[1], reverse=True)
            
            return [entry for entry, score in scored_entries[:limit]]
            
        except Exception as e:
            print(f"❌ 搜索内存失败: {str(e)}")
            return []
    
    def _calculate_memory_relevance(self, entry: MemoryEntry, query_words: set) -> float:
        """
        计算内存条目与查询的相关性
        
        Args:
            entry: 内存条目
            query_words: 查询词集合
            
        Returns:
            相关性分数
        """
        score = 0.0
        
        # 查询匹配
        entry_query_words = set(entry.query.lower().split())
        query_match = len(query_words & entry_query_words) / len(query_words) if query_words else 0
        score += query_match * 0.4
        
        # 答案匹配
        answer_words = set(entry.final_answer.lower().split())
        answer_match = len(query_words & answer_words) / len(query_words) if query_words else 0
        score += answer_match * 0.3
        
        # 时间衰减（最近的记录权重更高）
        try:
            entry_time = datetime.fromisoformat(entry.timestamp)
            time_diff = (datetime.now() - entry_time).days
            time_weight = max(0.1, 1.0 - time_diff / 30)  # 30天后权重降至0.1
            score *= time_weight
        except:
            pass
        
        # 成功性权重（有最终答案的记录权重更高）
        if entry.final_answer and len(entry.final_answer) > 20:
            score *= 1.2
        
        return score
    
    def get_recent_context(self, limit: int = 5) -> str:
        """
        获取最近的对话上下文
        
        Args:
            limit: 返回的最近条目数量
            
        Returns:
            格式化的上下文字符串
        """
        try:
            recent_entries = self.memory_entries[-limit:] if self.memory_entries else []
            
            context_parts = []
            for entry in recent_entries:
                context_part = f"Q: {entry.query}\nA: {entry.final_answer[:200]}..."
                context_parts.append(context_part)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"❌ 获取最近上下文失败: {str(e)}")
            return ""
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        获取当前会话摘要
        
        Returns:
            会话摘要
        """
        try:
            return {
                'session_id': self.current_session['session_id'],
                'duration': self._calculate_session_duration(),
                'total_queries': self.current_session['total_queries'],
                'queries': self.current_session['queries']
            }
        except Exception as e:
            print(f"❌ 获取会话摘要失败: {str(e)}")
            return {}
    
    def _calculate_session_duration(self) -> str:
        """计算会话持续时间"""
        try:
            start_time = datetime.fromisoformat(self.current_session['start_time'])
            duration = datetime.now() - start_time
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            if hours > 0:
                return f"{hours}小时{minutes}分钟"
            else:
                return f"{minutes}分钟"
        except:
            return "未知"
    
    def clear_old_memory(self, days: int = 30):
        """
        清理旧的内存记录
        
        Args:
            days: 保留天数
        """
        try:
            cutoff_date = datetime.now().replace(day=datetime.now().day - days)
            
            original_count = len(self.memory_entries)
            self.memory_entries = [
                entry for entry in self.memory_entries
                if datetime.fromisoformat(entry.timestamp) > cutoff_date
            ]
            
            cleaned_count = original_count - len(self.memory_entries)
            
            if cleaned_count > 0:
                self._save_memory()
                print(f"🧹 清理了 {cleaned_count} 条旧记录")
            
        except Exception as e:
            print(f"❌ 清理内存失败: {str(e)}")
    
    def export_memory(self, file_path: str, format: str = 'json'):
        """
        导出内存记录
        
        Args:
            file_path: 导出文件路径
            format: 导出格式（json/csv）
        """
        try:
            if format.lower() == 'json':
                data = [asdict(entry) for entry in self.memory_entries]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'csv':
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Timestamp', 'Query', 'Final Answer'])
                    for entry in self.memory_entries:
                        writer.writerow([entry.id, entry.timestamp, entry.query, entry.final_answer])
            
            print(f"📤 内存记录已导出到: {file_path}")
            
        except Exception as e:
            print(f"❌ 导出内存失败: {str(e)}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息
        
        Returns:
            统计信息字典
        """
        try:
            total_entries = len(self.memory_entries)
            
            if total_entries == 0:            return {
                'total_entries': 0,
                'avg_query_length': 0,
                'avg_answer_length': 0,
                'most_active_day': 'N/A'
            }
            
            # 基础统计
            total_query_chars = sum(len(entry.query) for entry in self.memory_entries)
            total_answer_chars = sum(len(entry.final_answer) for entry in self.memory_entries)
            
            # 按日期统计
            date_counts = {}
            for entry in self.memory_entries:
                try:
                    date = entry.timestamp.split('T')[0]
                    date_counts[date] = date_counts.get(date, 0) + 1
                except:
                    pass
            
            most_active_day = max(date_counts.items(), key=lambda x: x[1])[0] if date_counts else 'N/A'
            
            return {
                'total_entries': total_entries,
                'avg_query_length': total_query_chars / total_entries,
                'avg_answer_length': total_answer_chars / total_entries,
                'most_active_day': most_active_day,
                'date_distribution': date_counts
            }
            
        except Exception as e:
            print(f"❌ 获取内存统计失败: {str(e)}")
            return {}
    
    def _save_memory(self):
        """保存内存到文件"""
        try:
            os.makedirs(Config.MEMORY_CACHE_DIR, exist_ok=True)
            
            data = [asdict(entry) for entry in self.memory_entries]
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"❌ 保存内存失败: {str(e)}")
    
    def _save_session(self):
        """保存当前会话"""
        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存会话失败: {str(e)}")
    
    
    def get_similar_queries(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取相似的历史查询
        
        Args:
            query: 当前查询
            limit: 返回数量
            
        Returns:
            相似查询列表
        """
        try:
            similar_entries = self.search_memory(query, limit * 2)
            
            results = []
            for entry in similar_entries[:limit]:
                results.append({
                    'id': entry.id,
                    'query': entry.query,
                    'answer': entry.final_answer[:100] + '...' if len(entry.final_answer) > 100 else entry.final_answer,
                    'timestamp': entry.timestamp
                })
            
            return results
            
        except Exception as e:
            print(f"❌ 获取相似查询失败: {str(e)}")
            return []
