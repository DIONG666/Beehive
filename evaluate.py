"""
FRAMES基准评测脚本
使用多智能体系统回答问题，然后用DeepSeek-R1评估答案正确性
"""
import argparse
import json
import os
import time
import io
import sys
from typing import List, Dict, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

# 配置DeepSeek客户端用于评估
DEEPSEEK_CLIENT = None

def init_deepseek_client():
    """初始化DeepSeek客户端"""
    global DEEPSEEK_CLIENT
    api_key = Config.DEEPSEEK_API_KEY
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未设置，请在环境变量中配置")
    
    DEEPSEEK_CLIENT = OpenAI(
        api_key=api_key,
        base_url=Config.DEEPSEEK_BASE_URL
    )

def load_existing_results(filename: str) -> List[Dict]:
    """加载已有的评测结果"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_result(filename: str, result: Dict):
    """保存单个评测结果"""
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def get_last_processed_index(results: List[Dict]) -> int:
    """获取最后处理的索引"""
    if not results:
        return -1
    return max(int(r.get('index', -1)) for r in results)

def generate_research_prompt(prompt: str, wiki_links: List[str]) -> str:
    """生成研究提示"""
    if wiki_links:
        return f"根据以下Wikipedia资源回答问题:\n{wiki_links}\n\n问题: {prompt}"
    else:
        return f"问题: {prompt}"

def get_system_response(query: str, context: str = "") -> Dict[str, Any]:
    """
    使用多智能体系统获取回答
    
    Args:
        query: 查询问题
        context: 上下文信息
        
    Returns:
        系统回答结果
    """
    try:
        # 导入系统
        from main import MultiAgentResearchSystem
        system = MultiAgentResearchSystem()
        
        # 调用系统
        result =  system.research_query(query, context)
        return result
        
    except Exception as e:
        print(f"❌ 系统调用失败: {str(e)}")
        return {
            'answer': f"系统错误: {str(e)}",
            'citations': [],
            'reasoning_trace': "",
            'error': str(e)
        }

def evaluate_response_with_deepseek(question: str, system_response: str, ground_truth: str) -> Dict[str, str]:
    """
    使用DeepSeek-R1评估回答的正确性
    
    Args:
        question: 原始问题
        system_response: 系统回答
        ground_truth: 标准答案
        
    Returns:
        评估结果 (decision, explanation)
    """
    evaluation_prompt = f"""===任务===
我需要你帮助评估一个AI系统提供的答案与标准答案的匹配程度。你的任务是判断标准答案的内容是否在AI系统的回答中体现。

===指导原则===
1. 仔细比较"AI系统回答"与"标准答案"。
2. 关注答案的实质内容 - 寻找等价信息或正确答案。
3. 不要过分关注确切的措辞，除非确切措辞对意义至关重要。
4. 你的最终决定应基于"标准答案"的含义和重要事实是否在"AI系统回答"中体现。

===输入数据===
- 问题: {question}
- AI系统回答: {system_response}
- 标准答案: {ground_truth}

===输出格式===
严格按照以下输出格式输出，不要输出其它无关内容：
<explanation>(你是如何做出决定的?)</explanation>
<decision>("TRUE" 或 "FALSE")</decision>

请开始评估。"""

    try:
        evaluation_response = DEEPSEEK_CLIENT.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是一个客观公正的AI评估助手。"},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=4096,
            temperature=0.3,
        )
        
        evaluation_text = evaluation_response.choices[0].message.content.strip()
        print(f"📋 DeepSeek评估结果:\n{evaluation_text}")
        
        # 提取决定和解释
        lines = evaluation_text.split('\n')
        decision = "FALSE"
        explanation = ""
        
        for line in lines:
            line = line.strip()
            if "决定:" in line or "Decision:" in line:
                decision_part = line.split(":", 1)[1].strip().upper()
                if "TRUE" in decision_part:
                    decision = "TRUE"
                elif "FALSE" in decision_part:
                    decision = "FALSE"
            elif "解释:" in line or "Explanation:" in line:
                explanation = line.split(":", 1)[1].strip()
        
        return {"decision": decision, "explanation": explanation}
        
    except Exception as e:
        print(f"❌ DeepSeek评估失败: {str(e)}")
        return {"decision": "FALSE", "explanation": f"评估错误: {str(e)}"}

def process_single_item(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    处理单个评测项目
    
    Args:
        item: 数据集项目
        index: 项目索引
        
    Returns:
        处理结果
    """
    # 提取问题信息
    question = item.get('Prompt')
    ground_truth = item.get('Answer')
    reasoning_type = item.get('reasoning_types')
    wiki_links = item.get('wiki_links')
    prompt = generate_research_prompt(question, wiki_links)

    if not question:
        return {
            'index': index,
            'error': 'Question not found in item',
            'evaluation_decision': 'FALSE'
        }
    
    print(f"📝 处理问题 {index}: {question[:100]}...")
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用多智能体系统获取回答
    try:
        system_result =  get_system_response(prompt)
        system_answer = system_result.get('answer', '')
        system_citations = system_result.get('citations', [])
        reasoning_trace = system_result.get('reasoning_trace', "")
        
    except Exception as e:
        print(f"❌ 系统调用失败: {str(e)}")
        return {
            'index': index,
            'question': question,
            'ground_truth': ground_truth,
            'system_answer': f"系统错误: {str(e)}",
            'evaluation_decision': 'FALSE',
            'evaluation_explanation': f"系统错误: {str(e)}",
            'reasoning_type': reasoning_type,
            'response_time': 0,
            'error': str(e)
        }
    
    # 记录响应时间
    response_time = time.time() - start_time
    
    print(f"💡 系统回答: {system_answer[:100]}...")
    print(f"⏱️ 响应时间: {response_time:.2f}秒")
    
    # 使用DeepSeek评估答案
    print("🔍 正在评估答案...")
    evaluation = evaluate_response_with_deepseek(question, system_answer, ground_truth)
    
    print(f"✅ 评估结果: {evaluation['decision']}")
    
    return {
        'index': index,
        'question': question,
        'ground_truth': ground_truth,
        'system_answer': system_answer,
        'system_citations': system_citations,
        'reasoning_trace': reasoning_trace,
        'evaluation_decision': evaluation['decision'],
        'evaluation_explanation': evaluation['explanation'],
        'reasoning_type': reasoning_type,
        'response_time': response_time,
    }

def main():
    print("🚀 开始FRAMES基准评测")
    print("="*60)
    
    # 初始化DeepSeek客户端
    try:
        init_deepseek_client()
        print("✅ DeepSeek客户端初始化成功")
    except Exception as e:
        print(f"❌ DeepSeek客户端初始化失败: {str(e)}")
        return
    
    # 加载数据集
    print(f"📁 加载数据集")
    dataset = load_dataset("google/frames-benchmark", split="test")
    
    if not dataset:
        print("❌ 数据集加载失败")
        return
    
    print(f"📊 数据集包含 {len(dataset)} 个问题")
    
    # 准备结果文件
    filename = f"frames_evaluation_multi_agent.json"
    results_dir = os.path.join(Config.DATA_DIR, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    full_filename = os.path.join(results_dir, filename)
    
    # 检查已有结果
    existing_results = load_existing_results(full_filename)
    last_processed_index = get_last_processed_index(existing_results)
    
    print(f"📋 结果将保存到: {full_filename}")
    if existing_results:
        print(f"🔄 发现已有结果 {len(existing_results)} 个，从索引 {last_processed_index + 1} 开始")
    
    # 处理数据集
    processed_count = 0
    
    for i, item in enumerate(tqdm(dataset, desc="处理问题")):
        index = i
        
        # 跳过已处理的项目
        if index <= last_processed_index:
            continue
        
        try:
            result =  process_single_item(item, index)
            save_result(full_filename, result)
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 处理问题 {index} 失败: {str(e)}")
            continue
    
    # 计算并输出最终统计
    print("\n" + "="*60)
    print("📊 最终评测统计")
    print("="*60)
    
    results = load_existing_results(full_filename)
    total_samples = len(results)
    correct_answers = sum(1 for r in results if r.get('evaluation_decision') == 'TRUE')
    accuracy = correct_answers / total_samples if total_samples > 0 else 0
    
    print(f"总样本数: {total_samples}")
    print(f"正确答案: {correct_answers}")
    print(f"准确率: {accuracy:.2%}")
    
    # 按推理类型统计准确率
    reasoning_types = set(r.get('reasoning_type', 'unknown') for r in results)
    print(f"\n📈 按推理类型统计:")
    for rt in reasoning_types:
        rt_samples = [r for r in results if r.get('reasoning_type') == rt]
        if rt_samples:
            rt_correct = sum(1 for r in rt_samples if r.get('evaluation_decision') == 'TRUE')
            rt_accuracy = rt_correct / len(rt_samples)
            print(f"  {rt}: {rt_accuracy:.2%} ({rt_correct}/{len(rt_samples)})")
    
    # 计算平均响应时间
    response_times = [r.get('response_time', 0) for r in results if r.get('response_time')]
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        print(f"\n⏱️ 平均响应时间: {avg_time:.2f}秒")
    
    print(f"\n💾 完整结果已保存到: {full_filename}")

if __name__ == "__main__":
    # 运行评测
    main()
