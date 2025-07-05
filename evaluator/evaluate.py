"""
评测脚本：兼容FRAMES评测代码格式
"""
import json
import os
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class FramesEvaluator:
    """FRAMES基准评测器"""
    
    def __init__(self, research_system):
        """
        初始化评测器
        
        Args:
            research_system: 研究系统实例
        """
        self.research_system = research_system
        self.config = Config()
        self.results = []
        self.metrics = {
            'accuracy': 0.0,
            'exact_match': 0.0,
            'f1_score': 0.0,
            'citation_accuracy': 0.0,
            'average_response_time': 0.0
        }
    
    async def evaluate(self, dataset_path: str) -> Dict[str, float]:
        """
        评测系统在FRAMES数据集上的表现
        
        Args:
            dataset_path: FRAMES数据集路径
            
        Returns:
            评测结果字典
        """
        try:
            print(f"🧪 开始FRAMES基准评测...")
            print(f"📁 数据集路径: {dataset_path}")
            
            # 加载数据集
            dataset = await self._load_frames_dataset(dataset_path)
            
            if not dataset:
                print("❌ 数据集加载失败")
                return {}
            
            print(f"📊 数据集包含 {len(dataset)} 个问题")
            
            # 执行评测
            self.results = []
            
            for i, item in enumerate(dataset):
                print(f"\n🔍 处理问题 {i+1}/{len(dataset)}")
                result = await self._evaluate_single_item(item, i+1)
                self.results.append(result)
                
                # 每10个问题输出一次进度
                if (i + 1) % 10 == 0:
                    current_accuracy = self._calculate_current_accuracy()
                    print(f"📈 当前准确率: {current_accuracy:.2%}")
            
            # 计算最终指标
            final_metrics = self._calculate_metrics()
            
            # 保存结果
            await self._save_evaluation_results(dataset_path, final_metrics)
            
            print(f"\n🎯 评测完成!")
            self._print_final_results(final_metrics)
            
            return final_metrics
            
        except Exception as e:
            print(f"❌ 评测过程失败: {str(e)}")
            return {}
    
    async def _load_frames_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        加载FRAMES数据集
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            数据集列表
        """
        try:
            if os.path.isfile(dataset_path):
                # 单个JSON文件
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'questions' in data:
                    return data['questions']
                else:
                    print("❌ 不支持的数据集格式")
                    return []
            
            elif os.path.isdir(dataset_path):
                # 目录中的多个文件
                dataset = []
                for file_name in os.listdir(dataset_path):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(dataset_path, file_name)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_data = json.load(f)
                            if isinstance(file_data, list):
                                dataset.extend(file_data)
                            elif isinstance(file_data, dict):
                                dataset.append(file_data)
                
                return dataset
            
            else:
                print(f"❌ 数据集路径不存在: {dataset_path}")
                return []
                
        except Exception as e:
            print(f"❌ 加载数据集失败: {str(e)}")
            return []
    
    async def _evaluate_single_item(self, item: Dict[str, Any], item_number: int) -> Dict[str, Any]:
        """
        评测单个问题
        
        Args:
            item: 数据集项目
            item_number: 项目编号
            
        Returns:
            评测结果
        """
        try:
            # 提取问题和答案
            question = item.get('question', item.get('query', ''))
            expected_answer = item.get('answer', item.get('expected_answer', ''))
            expected_citations = item.get('citations', item.get('references', []))
            context = item.get('context', '')
            
            if not question:
                return {
                    'item_number': item_number,
                    'question': '',
                    'error': 'Question not found in item',
                    'accuracy': 0.0,
                    'exact_match': 0.0,
                    'f1_score': 0.0,
                    'citation_accuracy': 0.0,
                    'response_time': 0.0
                }
            
            print(f"❓ 问题: {question[:100]}...")
            
            # 记录开始时间
            start_time = asyncio.get_event_loop().time()
            
            # 调用研究系统
            try:
                system_result = await self.research_system.research_query(question, context)
            except Exception as e:
                print(f"❌ 系统调用失败: {str(e)}")
                return {
                    'item_number': item_number,
                    'question': question,
                    'error': f'System error: {str(e)}',
                    'accuracy': 0.0,
                    'exact_match': 0.0,
                    'f1_score': 0.0,
                    'citation_accuracy': 0.0,
                    'response_time': 0.0
                }
            
            # 记录结束时间
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            # 提取系统回答和引用
            system_answer = system_result.get('answer', '')
            system_citations = system_result.get('citations', [])
            
            print(f"💡 系统答案: {system_answer[:100]}...")
            print(f"⏱️ 响应时间: {response_time:.2f}秒")
            
            # 计算各项指标
            accuracy = self._calculate_answer_accuracy(system_answer, expected_answer)
            exact_match = self._calculate_exact_match(system_answer, expected_answer)
            f1_score = self._calculate_f1_score(system_answer, expected_answer)
            citation_accuracy = self._calculate_citation_accuracy(system_citations, expected_citations)
            
            return {
                'item_number': item_number,
                'question': question,
                'expected_answer': expected_answer,
                'system_answer': system_answer,
                'expected_citations': expected_citations,
                'system_citations': system_citations,
                'accuracy': accuracy,
                'exact_match': exact_match,
                'f1_score': f1_score,
                'citation_accuracy': citation_accuracy,
                'response_time': response_time,
                'reasoning_trace': system_result.get('reasoning_trace', []),
                'error': None
            }
            
        except Exception as e:
            print(f"❌ 评测单项失败: {str(e)}")
            return {
                'item_number': item_number,
                'question': item.get('question', ''),
                'error': str(e),
                'accuracy': 0.0,
                'exact_match': 0.0,
                'f1_score': 0.0,
                'citation_accuracy': 0.0,
                'response_time': 0.0
            }
    
    def _calculate_answer_accuracy(self, system_answer: str, expected_answer: str) -> float:
        """
        计算答案准确率（基于关键词重叠）
        
        Args:
            system_answer: 系统答案
            expected_answer: 期望答案
            
        Returns:
            准确率分数 (0-1)
        """
        if not expected_answer or not system_answer:
            return 0.0
        
        # 简单的词汇重叠计算
        expected_words = set(expected_answer.lower().split())
        system_words = set(system_answer.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = expected_words & system_words
        accuracy = len(overlap) / len(expected_words)
        
        return min(accuracy, 1.0)
    
    def _calculate_exact_match(self, system_answer: str, expected_answer: str) -> float:
        """
        计算精确匹配分数
        
        Args:
            system_answer: 系统答案
            expected_answer: 期望答案
            
        Returns:
            精确匹配分数 (0 或 1)
        """
        if not expected_answer or not system_answer:
            return 0.0
        
        # 标准化文本进行比较
        normalized_system = ' '.join(system_answer.lower().split())
        normalized_expected = ' '.join(expected_answer.lower().split())
        
        return 1.0 if normalized_system == normalized_expected else 0.0
    
    def _calculate_f1_score(self, system_answer: str, expected_answer: str) -> float:
        """
        计算F1分数
        
        Args:
            system_answer: 系统答案
            expected_answer: 期望答案
            
        Returns:
            F1分数
        """
        if not expected_answer or not system_answer:
            return 0.0
        
        expected_words = set(expected_answer.lower().split())
        system_words = set(system_answer.lower().split())
        
        if not expected_words or not system_words:
            return 0.0
        
        overlap = expected_words & system_words
        
        precision = len(overlap) / len(system_words) if system_words else 0.0
        recall = len(overlap) / len(expected_words) if expected_words else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _calculate_citation_accuracy(self, system_citations: List[str], 
                                   expected_citations: List[str]) -> float:
        """
        计算引用准确率
        
        Args:
            system_citations: 系统引用
            expected_citations: 期望引用
            
        Returns:
            引用准确率
        """
        if not expected_citations:
            return 1.0 if not system_citations else 0.5  # 如果不需要引用，没有引用得满分
        
        if not system_citations:
            return 0.0
        
        # 简单的引用匹配（基于URL或关键词）
        matches = 0
        total_expected = len(expected_citations)
        
        for expected_cite in expected_citations:
            for system_cite in system_citations:
                if self._citations_match(expected_cite, system_cite):
                    matches += 1
                    break
        
        return matches / total_expected if total_expected > 0 else 0.0
    
    def _citations_match(self, expected: str, system: str) -> bool:
        """
        判断两个引用是否匹配
        
        Args:
            expected: 期望引用
            system: 系统引用
            
        Returns:
            是否匹配
        """
        # 简单的匹配逻辑，可以根据需要改进
        expected_lower = expected.lower()
        system_lower = system.lower()
        
        # 检查URL匹配
        if 'http' in expected_lower and 'http' in system_lower:
            # 提取域名进行比较
            expected_domain = self._extract_domain(expected)
            system_domain = self._extract_domain(system)
            return expected_domain == system_domain
        
        # 检查关键词匹配
        expected_words = set(expected_lower.split())
        system_words = set(system_lower.split())
        overlap = expected_words & system_words
        
        # 如果有超过一半的词汇重叠，认为匹配
        return len(overlap) >= min(len(expected_words), len(system_words)) * 0.5
    
    def _extract_domain(self, url: str) -> str:
        """从URL中提取域名"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return url.lower()
    
    def _calculate_current_accuracy(self) -> float:
        """计算当前平均准确率"""
        if not self.results:
            return 0.0
        
        total_accuracy = sum(result.get('accuracy', 0.0) for result in self.results)
        return total_accuracy / len(self.results)
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """
        计算所有评测指标
        
        Returns:
            指标字典
        """
        if not self.results:
            return self.metrics
        
        valid_results = [r for r in self.results if r.get('error') is None]
        
        if not valid_results:
            return self.metrics
        
        # 计算平均指标
        total_accuracy = sum(r.get('accuracy', 0.0) for r in valid_results)
        total_exact_match = sum(r.get('exact_match', 0.0) for r in valid_results)
        total_f1 = sum(r.get('f1_score', 0.0) for r in valid_results)
        total_citation_accuracy = sum(r.get('citation_accuracy', 0.0) for r in valid_results)
        total_response_time = sum(r.get('response_time', 0.0) for r in valid_results)
        
        count = len(valid_results)
        
        return {
            'accuracy': total_accuracy / count,
            'exact_match': total_exact_match / count,
            'f1_score': total_f1 / count,
            'citation_accuracy': total_citation_accuracy / count,
            'average_response_time': total_response_time / count,
            'total_questions': len(self.results),
            'valid_questions': count,
            'error_rate': (len(self.results) - count) / len(self.results) if self.results else 0.0
        }
    
    async def _save_evaluation_results(self, dataset_path: str, metrics: Dict[str, float]):
        """
        保存评测结果
        
        Args:
            dataset_path: 数据集路径
            metrics: 评测指标
        """
        try:
            # 创建结果目录
            results_dir = os.path.join(Config.DATA_DIR, 'evaluation_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成结果文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(results_dir, f'frames_evaluation_{timestamp}.json')
            
            # 准备保存的数据
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'dataset_path': dataset_path,
                'system_info': {
                    'config': Config.get_config(),
                    'agent_info': self.research_system.main_agent.get_agent_info()
                },
                'metrics': metrics,
                'detailed_results': self.results
            }
            
            # 保存到文件
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 评测结果已保存到: {results_file}")
            
            # 也保存一份简化的摘要
            summary_file = os.path.join(results_dir, f'frames_summary_{timestamp}.json')
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'dataset_path': dataset_path,
                'metrics': metrics
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"❌ 保存评测结果失败: {str(e)}")
    
    def _print_final_results(self, metrics: Dict[str, float]):
        """
        打印最终评测结果
        
        Args:
            metrics: 评测指标
        """
        print("\n" + "="*60)
        print("📊 FRAMES基准评测结果")
        print("="*60)
        
        print(f"总问题数: {metrics.get('total_questions', 0)}")
        print(f"有效问题数: {metrics.get('valid_questions', 0)}")
        print(f"错误率: {metrics.get('error_rate', 0.0):.2%}")
        print()
        
        print("📈 核心指标:")
        print(f"  准确率 (Accuracy): {metrics.get('accuracy', 0.0):.2%}")
        print(f"  精确匹配 (Exact Match): {metrics.get('exact_match', 0.0):.2%}")
        print(f"  F1分数: {metrics.get('f1_score', 0.0):.3f}")
        print(f"  引用准确率: {metrics.get('citation_accuracy', 0.0):.2%}")
        print(f"  平均响应时间: {metrics.get('average_response_time', 0.0):.2f}秒")
        
        print("\n" + "="*60)
    
    def generate_detailed_report(self) -> str:
        """
        生成详细的评测报告
        
        Returns:
            评测报告文本
        """
        if not self.results:
            return "无评测结果可生成报告。"
        
        report_lines = []
        report_lines.append("# FRAMES基准评测详细报告")
        report_lines.append(f"评测时间: {datetime.now().isoformat()}")
        report_lines.append(f"总问题数: {len(self.results)}")
        report_lines.append("")
        
        # 按准确率分组分析
        high_accuracy = [r for r in self.results if r.get('accuracy', 0) > 0.8]
        medium_accuracy = [r for r in self.results if 0.5 <= r.get('accuracy', 0) <= 0.8]
        low_accuracy = [r for r in self.results if r.get('accuracy', 0) < 0.5]
        
        report_lines.append("## 准确率分布")
        report_lines.append(f"高准确率 (>80%): {len(high_accuracy)} 题")
        report_lines.append(f"中等准确率 (50-80%): {len(medium_accuracy)} 题")
        report_lines.append(f"低准确率 (<50%): {len(low_accuracy)} 题")
        report_lines.append("")
        
        # 错误分析
        error_results = [r for r in self.results if r.get('error')]
        if error_results:
            report_lines.append("## 错误分析")
            for result in error_results[:5]:  # 只显示前5个错误
                report_lines.append(f"- 问题 {result['item_number']}: {result.get('error', '')}")
            report_lines.append("")
        
        # 性能分析
        response_times = [r.get('response_time', 0) for r in self.results if r.get('response_time')]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            report_lines.append("## 性能分析")
            report_lines.append(f"平均响应时间: {avg_time:.2f}秒")
            report_lines.append(f"最长响应时间: {max_time:.2f}秒")
            report_lines.append(f"最短响应时间: {min_time:.2f}秒")
            report_lines.append("")
        
        return "\n".join(report_lines)


# 命令行接口
async def main():
    """命令行评测入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FRAMES基准评测")
    parser.add_argument("--dataset", required=True, help="FRAMES数据集路径")
    parser.add_argument("--output", help="结果输出目录")
    
    args = parser.parse_args()
    
    # 这里需要导入研究系统，但避免循环导入
    try:
        from main import MultiAgentResearchSystem
        system = MultiAgentResearchSystem()
        
        evaluator = FramesEvaluator(system)
        results = await evaluator.evaluate(args.dataset)
        
        if args.output:
            # 保存到指定目录
            os.makedirs(args.output, exist_ok=True)
            output_file = os.path.join(args.output, 'evaluation_results.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_file}")
        
        return 0
        
    except ImportError as e:
        print(f"❌ 无法导入研究系统: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
