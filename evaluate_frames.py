"""
基于新系统流程的FRAMES评估脚本
"""
import argparse
import json
import os
import time
import asyncio
from typing import List, Dict, Any
from datetime import datetime

# 导入我们的研究系统
from agent.main_agent import MainAgent
from config import Config
from planner.planner import DeepSeekPlanner


class NewFramesEvaluator:
    """新的FRAMES评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.system = MainAgent()
        self.evaluator_planner = DeepSeekPlanner()
        self.results = []
    
    def load_existing_results(self, filename: str) -> List[Dict]:
        """加载现有结果"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_result(self, filename: str, result: Dict):
        """保存单个结果"""
        results = self.load_existing_results(filename)
        results.append(result)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def get_last_processed_index(self, results: List[Dict]) -> int:
        """获取最后处理的索引"""
        if not results:
            return -1
        return max(int(r.get('index', -1)) for r in results)
    
    def generate_research_prompt(self, prompt: str, wiki_links: List[str]) -> str:
        """生成研究提示"""
        if wiki_links:
            links_text = "\n".join(wiki_links)
            return f"根据以下Wikipedia资源回答问题:\n{links_text}\n\n问题: {prompt}"
        else:
            return f"问题: {prompt}"
    
    async def get_system_response(self, prompt: str) -> Dict[str, Any]:
        """获取系统响应"""
        try:
            result = await self.system.execute_reasoning(prompt)
            return {
                'answer': result.get('answer', ''),
                'citations': result.get('citations', []),
                'reasoning_steps': len(result.get('reasoning_trace', [])),
                'sources_used': len(result.get('search_results', [])),
                'iterations': result.get('metadata', {}).get('iterations', 0),
                'error': result.get('metadata', {}).get('error', False)
            }
        except Exception as e:
            return {
                'answer': f'系统错误: {str(e)}',
                'citations': [],
                'reasoning_steps': 0,
                'sources_used': 0,
                'iterations': 0,
                'error': True
            }
    
    async def evaluate_response(self, question: str, system_response: str, ground_truth: str) -> Dict[str, str]:
        """使用DeepSeek评估响应质量"""
        evaluation_prompt = f"""请评估以下LLM回答的质量：

问题: {question}

系统回答: {system_response}

参考答案: {ground_truth}

评估标准：
1. 回答是否包含参考答案的核心信息？
2. 回答是否准确、相关？
3. 回答是否有良好的逻辑性和完整性？

请给出评估结果：

评估结果: TRUE/FALSE (TRUE表示回答质量好，FALSE表示质量差)
解释: [详细解释评估理由]
分数: [0-100的数字分数]
"""
        
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的答案质量评估专家。"},
                {"role": "user", "content": evaluation_prompt}
            ]
            
            response = await self.evaluator_planner.generate_response(messages)
            
            # 解析评估结果
            lines = response.split('\n')
            decision = "FALSE"
            explanation = ""
            score = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith("评估结果:"):
                    decision = "TRUE" if "TRUE" in line else "FALSE"
                elif line.startswith("解释:"):
                    explanation = line.split(":", 1)[1].strip()
                elif line.startswith("分数:"):
                    try:
                        score = int(line.split(":", 1)[1].strip())
                    except:
                        score = 0
            
            return {
                "decision": decision,
                "explanation": explanation,
                "score": score
            }
            
        except Exception as e:
            return {
                "decision": "FALSE",
                "explanation": f"评估过程出错: {str(e)}",
                "score": 0
            }
    
    async def evaluate_dataset(self, dataset_path: str, output_file: str, max_samples: int = None):
        """评估数据集"""
        print(f"📊 开始评估数据集: {dataset_path}")
        
        # 加载数据集
        try:
            if dataset_path.endswith('.json'):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
            else:
                print(f"❌ 不支持的数据格式: {dataset_path}")
                return
        except Exception as e:
            print(f"❌ 加载数据集失败: {str(e)}")
            return
        
        # 检查已有结果
        existing_results = self.load_existing_results(output_file)
        last_processed = self.get_last_processed_index(existing_results)
        
        print(f"📋 数据集大小: {len(dataset)}")
        print(f"📋 已处理样本: {len(existing_results)}")
        print(f"📋 最后处理索引: {last_processed}")
        
        # 如果指定了最大样本数
        if max_samples:
            dataset = dataset[:max_samples]
            print(f"📋 限制样本数: {max_samples}")
        
        # 开始评估
        processed_count = 0
        for i, item in enumerate(dataset):
            # 跳过已处理的样本
            if i <= last_processed:
                continue
            
            try:
                print(f"\n🔍 处理样本 {i+1}/{len(dataset)}")
                
                # 构建研究提示
                prompt = self.generate_research_prompt(
                    item.get('Prompt', ''), 
                    item.get('wiki_links', [])
                )
                
                # 获取系统响应
                system_result = await self.get_system_response(prompt)
                
                # 评估响应质量
                evaluation = await self.evaluate_response(
                    item.get('Prompt', ''),
                    system_result['answer'],
                    item.get('Answer', '')
                )
                
                # 保存结果
                result = {
                    "index": i,
                    "prompt": item.get('Prompt', ''),
                    "ground_truth": item.get('Answer', ''),
                    "system_response": system_result['answer'],
                    "evaluation_decision": evaluation['decision'],
                    "evaluation_explanation": evaluation['explanation'],
                    "evaluation_score": evaluation['score'],
                    "reasoning_type": item.get('reasoning_types', 'unknown'),
                    "wiki_links": item.get('wiki_links', []),
                    "system_metadata": {
                        "citations": len(system_result['citations']),
                        "reasoning_steps": system_result['reasoning_steps'],
                        "sources_used": system_result['sources_used'],
                        "iterations": system_result['iterations'],
                        "error": system_result['error']
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                self.save_result(output_file, result)
                processed_count += 1
                
                print(f"✅ 样本 {i+1} 完成 - 决策: {evaluation['decision']}, 分数: {evaluation['score']}")
                
                # 可选：添加延迟以避免API限制
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ 处理样本 {i+1} 失败: {str(e)}")
                # 保存错误结果
                error_result = {
                    "index": i,
                    "prompt": item.get('Prompt', ''),
                    "ground_truth": item.get('Answer', ''),
                    "system_response": f"处理错误: {str(e)}",
                    "evaluation_decision": "FALSE",
                    "evaluation_explanation": f"处理过程中出错: {str(e)}",
                    "evaluation_score": 0,
                    "reasoning_type": item.get('reasoning_types', 'unknown'),
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                }
                self.save_result(output_file, error_result)
                continue
        
        print(f"\n✅ 评估完成！处理了 {processed_count} 个新样本")
        
        # 生成统计报告
        await self.generate_report(output_file)
    
    async def generate_report(self, results_file: str):
        """生成评估报告"""
        print("\n📊 生成评估报告...")
        
        results = self.load_existing_results(results_file)
        if not results:
            print("❌ 没有找到评估结果")
            return
        
        # 基本统计
        total_samples = len(results)
        correct_answers = sum(1 for r in results if r['evaluation_decision'] == 'TRUE')
        accuracy = correct_answers / total_samples if total_samples > 0 else 0
        
        # 分数统计
        scores = [r.get('evaluation_score', 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # 系统性能统计
        avg_iterations = sum(r.get('system_metadata', {}).get('iterations', 0) for r in results) / total_samples
        avg_sources = sum(r.get('system_metadata', {}).get('sources_used', 0) for r in results) / total_samples
        error_rate = sum(1 for r in results if r.get('system_metadata', {}).get('error', False)) / total_samples
        
        print(f"\n📈 评估报告")
        print(f"=" * 50)
        print(f"总样本数: {total_samples}")
        print(f"正确答案数: {correct_answers}")
        print(f"准确率: {accuracy:.2%}")
        print(f"平均分数: {avg_score:.1f}/100")
        print(f"平均迭代次数: {avg_iterations:.1f}")
        print(f"平均使用来源数: {avg_sources:.1f}")
        print(f"错误率: {error_rate:.2%}")
        
        # 按推理类型分析
        reasoning_types = {}
        for r in results:
            rt = r.get('reasoning_type', 'unknown')
            if rt not in reasoning_types:
                reasoning_types[rt] = {'total': 0, 'correct': 0, 'scores': []}
            reasoning_types[rt]['total'] += 1
            if r['evaluation_decision'] == 'TRUE':
                reasoning_types[rt]['correct'] += 1
            reasoning_types[rt]['scores'].append(r.get('evaluation_score', 0))
        
        print(f"\n📊 按推理类型分析:")
        for rt, stats in reasoning_types.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            print(f"  {rt}: {accuracy:.2%} ({stats['correct']}/{stats['total']}) - 平均分: {avg_score:.1f}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="新系统FRAMES评估")
    parser.add_argument("--dataset", required=True, help="数据集路径")
    parser.add_argument("--output", default="evaluation_results_new.json", help="结果输出文件")
    parser.add_argument("--max-samples", type=int, help="最大样本数限制")
    
    args = parser.parse_args()
    
    # 检查配置
    config = Config()
    if not config.DEEPSEEK_API_KEY:
        print("❌ 错误: DEEPSEEK_API_KEY 未设置")
        print("请设置环境变量: export DEEPSEEK_API_KEY='your_key'")
        return 1
    
    # 开始评估
    evaluator = NewFramesEvaluator()
    
    try:
        await evaluator.evaluate_dataset(
            args.dataset,
            args.output,
            args.max_samples
        )
        return 0
    except Exception as e:
        print(f"❌ 评估失败: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
