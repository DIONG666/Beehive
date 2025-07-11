"""
FRAMES基准评测脚本 - 指定索引版本
支持指定特定的index列表进行评测，并覆盖原有结果

使用方法：
1. 代码内配置（推荐）：
   修改main()函数中的TARGET_INDICES列表，然后直接运行：
   python evaluate_specific.py

2. 命令行参数：
   python evaluate_specific.py --indices "0,1,2"
   python evaluate_specific.py --indices "0-5"
   python evaluate_specific.py --indices "0,2-4,6"

3. 重测FALSE结果：
   python evaluate_specific.py --retest-false --start-index 100

4. 重测子串匹配FALSE结果：
   python evaluate_specific.py --retest-substring-false --start-index 0

5. 查看分析报告：
   python evaluate_specific.py --show-false-summary
   python evaluate_specific.py --show-substring-details
   python evaluate_specific.py --check-substring-accuracy

示例配置：
TARGET_INDICES = [0, 1, 2]           # 评测索引0,1,2
TARGET_INDICES = list(range(0, 10))  # 评测索引0-9
TARGET_INDICES = [0, 5, 10, 15]      # 评测特定索引
RETEST_SUBSTRING_FALSE = True        # 重测包含答案但为FALSE的结果
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

def save_results(filename: str, results: List[Dict]):
    """保存完整的评测结果列表"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def update_result_by_index(results: List[Dict], new_result: Dict) -> List[Dict]:
    """
    根据index更新结果列表中的对应项目
    
    Args:
        results: 现有结果列表
        new_result: 新的结果项目
        
    Returns:
        更新后的结果列表
    """
    target_index = new_result.get('index')
    
    # 查找是否已存在该index的结果
    for i, result in enumerate(results):
        if result.get('index') == target_index:
            results[i] = new_result
            print(f"✅ 已覆盖索引 {target_index} 的结果")
            return results
    
    # 如果不存在，则添加新结果
    results.append(new_result)
    print(f"✅ 已添加索引 {target_index} 的新结果")
    return results

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
        result = system.research_query(query, context)
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
5. 如果AI系统回答中正确回答了标准答案中的关键信息或等价内容，则认为是正确的。比如询问"How old was the journalist who reviewed the first iPad for the Wall St Journal when the first iPad came out?"，尽管标准答案中还会提到记者的名字，但是只要系统回答了记者的年龄，就认为是正确的。

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
        
        # 使用正则表达式提取标签内容
        import re
        
        # 提取 <explanation> 标签内容
        explanation_match = re.search(r'<explanation>(.*?)</explanation>', evaluation_text, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        # 提取 <decision> 标签内容
        decision_match = re.search(r'<decision>(.*?)</decision>', evaluation_text, re.DOTALL | re.IGNORECASE)
        decision_text = decision_match.group(1).strip() if decision_match else ""
        
        # 处理决定结果
        decision = "FALSE"  # 默认值
        if decision_text:
            decision_upper = decision_text.upper()
            if "TRUE" in decision_upper:
                decision = "TRUE"
            elif "FALSE" in decision_upper:
                decision = "FALSE"
        
        # 如果标签提取失败，尝试备用方法（向后兼容）
        if not explanation and not decision_text:
            print("⚠️ 标签提取失败，尝试备用方法...")
            lines = evaluation_text.split('\n')
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
    # prompt = generate_research_prompt(question, wiki_links)
    prompt = question

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
        system_result = get_system_response(prompt)
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

def parse_indices(indices_str: str) -> List[int]:
    """
    解析索引字符串，支持多种格式
    
    Args:
        indices_str: 索引字符串，支持格式：
                    - "1,2,3" (逗号分隔)
                    - "1-5" (范围)
                    - "1,3-5,7" (混合)
                    
    Returns:
        索引列表
    """
    indices = set()
    
    # 按逗号分割
    parts = indices_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 处理范围
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            indices.update(range(start, end + 1))
        else:
            # 处理单个数字
            indices.add(int(part))
    
    return sorted(list(indices))

def get_false_indices(results_file: str, start_index: int = 0) -> List[int]:
    """
    从结果文件中提取评测结果为FALSE的索引
    
    Args:
        results_file: 结果文件路径
        start_index: 开始检测的索引位置（默认为0，即从头开始）
        
    Returns:
        评测结果为FALSE的索引列表（只包含 >= start_index 的索引）
    """
    try:
        results = load_existing_results(results_file)
        false_indices = []
        
        for result in results:
            if result.get('evaluation_decision') == 'FALSE':
                index = result.get('index')
                if index is not None and index >= start_index:
                    false_indices.append(index)
        
        return sorted(false_indices)
        
    except Exception as e:
        print(f"❌ 读取结果文件失败: {str(e)}")
        return []

def show_false_results_summary(results_file: str, start_index: int = 0):
    """
    显示FALSE结果的详细摘要
    
    Args:
        results_file: 结果文件路径
        start_index: 起始索引，只显示 >= start_index 的FALSE结果
    """
    try:
        results = load_existing_results(results_file)
        false_results = [r for r in results if r.get('evaluation_decision') == 'FALSE' and r.get('index', 0) >= start_index]
        
        if not false_results:
            print(f"✅ 没有找到索引 >= {start_index} 的FALSE结果")
            return
            
        print(f"\n📊 FALSE结果摘要（从索引 {start_index} 开始，共 {len(false_results)} 个）:")
        print("-" * 60)
        
        # 按推理类型分组
        from collections import defaultdict
        by_reasoning_type = defaultdict(list)
        
        for result in false_results:
            reasoning_type = result.get('reasoning_type', 'unknown')
            by_reasoning_type[reasoning_type].append(result.get('index'))
        
        for reasoning_type, indices in by_reasoning_type.items():
            print(f"📈 {reasoning_type}: {len(indices)} 个")
            print(f"   索引: {indices[:10]}{'...' if len(indices) > 10 else ''}")
        
        false_indices = [r.get('index') for r in false_results]
        print(f"\n🔢 所有FALSE索引（>= {start_index}）: {false_indices}")
        
    except Exception as e:
        print(f"❌ 生成摘要失败: {str(e)}")

def find_false_with_substring_match(results_file: str, start_index: int = 0, case_sensitive: bool = False) -> List[int]:
    """
    找出系统回答包含标准答案作为子串但被评为FALSE的索引
    
    Args:
        results_file: 结果文件路径
        start_index: 开始检测的索引位置
        case_sensitive: 是否区分大小写（默认不区分）
        
    Returns:
        符合条件的索引列表
    """
    try:
        results = load_existing_results(results_file)
        substring_false_indices = []
        
        for result in results:
            if (result.get('evaluation_decision') == 'FALSE' and 
                result.get('index', 0) >= start_index):
                
                system_answer = result.get('system_answer', '')
                ground_truth = result.get('ground_truth', '')
                
                if system_answer and ground_truth:
                    # 根据是否区分大小写进行比较
                    if case_sensitive:
                        contains_truth = system_answer in ground_truth
                    else:
                        contains_truth = system_answer.lower() in ground_truth.lower()

                    if contains_truth:
                        substring_false_indices.append(result.get('index'))
        
        return sorted(substring_false_indices)
        
    except Exception as e:
        print(f"❌ 查找子串匹配失败: {str(e)}")
        return []

def show_substring_false_details(results_file: str, start_index: int = 0, case_sensitive: bool = False, max_display: int = 10):
    """
    显示包含子串但被评为FALSE的详细信息
    
    Args:
        results_file: 结果文件路径
        start_index: 起始索引
        case_sensitive: 是否区分大小写
        max_display: 最大显示条数
    """
    try:
        results = load_existing_results(results_file)
        substring_false_results = []
        
        for result in results:
            if (result.get('evaluation_decision') == 'FALSE' and 
                result.get('index', 0) >= start_index):
                
                system_answer = result.get('system_answer', '')
                ground_truth = result.get('ground_truth', '')
                
                if system_answer and ground_truth:
                    # 根据是否区分大小写进行比较
                    if case_sensitive:
                        contains_truth = ground_truth in system_answer
                    else:
                        contains_truth = ground_truth.lower() in system_answer.lower()
                    
                    if contains_truth:
                        substring_false_results.append(result)
        
        if not substring_false_results:
            print(f"✅ 没有找到索引 >= {start_index} 的子串匹配FALSE结果")
            return
        
        print(f"\n📊 子串匹配FALSE结果详情（从索引 {start_index} 开始，共 {len(substring_false_results)} 个）:")
        print("=" * 80)
        
        # 显示前几个详细案例
        display_count = min(max_display, len(substring_false_results))
        for i, result in enumerate(substring_false_results[:display_count]):
            index = result.get('index')
            question = result.get('question', '')[:100] + '...' if len(result.get('question', '')) > 100 else result.get('question', '')
            system_answer = result.get('system_answer', '')
            ground_truth = result.get('ground_truth', '')
            evaluation_explanation = result.get('evaluation_explanation', '')
            
            print(f"\n📝 索引 {index}:")
            print(f"问题: {question}")
            print(f"标准答案: {ground_truth}")
            print(f"系统回答: {system_answer[:200]}{'...' if len(system_answer) > 200 else ''}")
            print(f"评估解释: {evaluation_explanation[:150]}{'...' if len(evaluation_explanation) > 150 else ''}")
            
            # 高亮显示匹配部分
            if case_sensitive:
                match_pos = system_answer.find(ground_truth)
            else:
                match_pos = system_answer.lower().find(ground_truth.lower())
            
            if match_pos >= 0:
                start_pos = max(0, match_pos - 50)
                end_pos = min(len(system_answer), match_pos + len(ground_truth) + 50)
                context = system_answer[start_pos:end_pos]
                print(f"🎯 匹配上下文: ...{context}...")
            
            print("-" * 60)
        
        if len(substring_false_results) > max_display:
            print(f"\n... 还有 {len(substring_false_results) - max_display} 个类似结果")
        
        # 显示所有索引
        all_indices = [r.get('index') for r in substring_false_results]
        print(f"\n🔢 所有子串匹配FALSE索引: {all_indices}")
        
        # 按推理类型分组统计
        from collections import defaultdict
        by_reasoning_type = defaultdict(list)
        
        for result in substring_false_results:
            reasoning_type = result.get('reasoning_type', 'unknown')
            by_reasoning_type[reasoning_type].append(result.get('index'))
        
        print(f"\n📈 按推理类型分组:")
        for reasoning_type, indices in by_reasoning_type.items():
            print(f"  {reasoning_type}: {len(indices)} 个 - {indices[:5]}{'...' if len(indices) > 5 else ''}")
        
    except Exception as e:
        print(f"❌ 显示详情失败: {str(e)}")

def check_substring_matching_accuracy(results_file: str, start_index: int = 0, case_sensitive: bool = False):
    """
    检查子串匹配的准确性统计
    
    Args:
        results_file: 结果文件路径  
        start_index: 起始索引
        case_sensitive: 是否区分大小写
    """
    try:
        results = load_existing_results(results_file)
        
        # 统计各种情况
        total_false = 0
        substring_match_false = 0  # 包含子串但为FALSE
        substring_match_true = 0   # 包含子串且为TRUE
        no_substring_false = 0     # 不包含子串且为FALSE
        no_substring_true = 0      # 不包含子串但为TRUE（异常情况）
        
        for result in results:
            if result.get('index', 0) < start_index:
                continue
                
            system_answer = result.get('system_answer', '')
            ground_truth = result.get('ground_truth', '')
            evaluation_decision = result.get('evaluation_decision', 'FALSE')
            
            if system_answer and ground_truth:
                # 检查是否包含子串
                if case_sensitive:
                    contains_truth = ground_truth in system_answer
                else:
                    contains_truth = ground_truth.lower() in system_answer.lower()
                
                if evaluation_decision == 'FALSE':
                    total_false += 1
                    if contains_truth:
                        substring_match_false += 1
                    else:
                        no_substring_false += 1
                else:  # TRUE
                    if contains_truth:
                        substring_match_true += 1
                    else:
                        no_substring_true += 1
        
        print(f"\n📊 子串匹配准确性分析（索引 >= {start_index}）:")
        print("=" * 50)
        print(f"📉 FALSE结果总数: {total_false}")
        print(f"🎯 包含标准答案但为FALSE: {substring_match_false} ({substring_match_false/total_false*100:.1f}%)")
        print(f"❌ 不包含标准答案且为FALSE: {no_substring_false} ({no_substring_false/total_false*100:.1f}%)")
        print(f"✅ 包含标准答案且为TRUE: {substring_match_true}")
        print(f"⚠️  不包含标准答案但为TRUE: {no_substring_true} (可能的异常)")
        
        if substring_match_false > 0:
            print(f"\n💡 建议: 有 {substring_match_false} 个结果可能被误判，建议重新评估")
        
    except Exception as e:
        print(f"❌ 统计分析失败: {str(e)}")

def main():
    # ===== 在这里配置要评测的索引和结果文件 =====
    # 直接在代码中定义要评测的索引列表
    TARGET_INDICES = [474,482,494,514,540,572,584,585,629,663,684,712,739,767,773,775]  # 可以修改这个列表来指定要评测的索引
    
    RESULTS_FILE = 'frames_evaluation_multi_agent.json'  # 结果文件名
    
    # 重测模式：是否重测FALSE结果
    RETEST_FALSE_ONLY = True  # 设置为True时，会自动提取FALSE结果进行重测
    
    # 子串匹配重测模式：重测那些包含标准答案但被评为FALSE的结果
    RETEST_SUBSTRING_FALSE = False  # 设置为True时，会重测包含子串但为FALSE的结果
    
    # FALSE重测的起始索引（只有当重测FALSE结果时才生效）
    FALSE_START_INDEX = 0  # 从这个索引开始检测FALSE结果，默认为0（从头开始）
    
    # 子串匹配是否区分大小写
    CASE_SENSITIVE = True  # 默认不区分大小写
    
    # 如果命令行参数存在，则使用命令行参数（向后兼容）
    parser = argparse.ArgumentParser(description='FRAMES基准评测 - 指定索引版本')
    parser.add_argument('--indices', type=str, required=False, 
                       help='要评测的索引列表，支持格式：1,2,3 或 1-5 或 1,3-5,7')
    parser.add_argument('--results-file', type=str, 
                       default=RESULTS_FILE,
                       help='结果文件名（默认：frames_evaluation_multi_agent.json）')
    parser.add_argument('--retest-false', action='store_true',
                       help='重测结果文件中评测为FALSE的项目')
    parser.add_argument('--retest-substring-false', action='store_true',
                       help='重测包含标准答案但被评为FALSE的项目')
    parser.add_argument('--start-index', type=int, default=FALSE_START_INDEX,
                       help='重测FALSE结果时的起始索引（默认：0）')
    parser.add_argument('--show-false-summary', action='store_true',
                       help='仅显示FALSE结果摘要，不进行评测')
    parser.add_argument('--show-substring-details', action='store_true',
                       help='显示子串匹配FALSE结果的详细信息')
    parser.add_argument('--check-substring-accuracy', action='store_true',
                       help='检查子串匹配的准确性统计')
    parser.add_argument('--case-sensitive', action='store_true',
                       help='子串匹配时区分大小写（默认不区分）')
    
    args = parser.parse_args()
    
    print("🚀 开始FRAMES基准评测 - 指定索引版本")
    print("="*60)
    
    # 准备结果文件路径
    results_file = args.results_file
    results_dir = os.path.join(Config.DATA_DIR, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    full_filename = os.path.join(results_dir, results_file)
    
    # 如果只是要显示摘要，则显示后退出
    if args.show_false_summary:
        start_index = args.start_index if hasattr(args, 'start_index') else FALSE_START_INDEX
        show_false_results_summary(full_filename, start_index)
        return
    
    # 如果只是要显示子串匹配详情，则显示后退出
    if args.show_substring_details:
        start_index = args.start_index if hasattr(args, 'start_index') else FALSE_START_INDEX
        case_sensitive = args.case_sensitive if hasattr(args, 'case_sensitive') else CASE_SENSITIVE
        show_substring_false_details(full_filename, start_index, case_sensitive)
        return
    
    # 如果只是要检查子串匹配准确性，则显示后退出
    if args.check_substring_accuracy:
        start_index = args.start_index if hasattr(args, 'start_index') else FALSE_START_INDEX
        case_sensitive = args.case_sensitive if hasattr(args, 'case_sensitive') else CASE_SENSITIVE
        check_substring_matching_accuracy(full_filename, start_index, case_sensitive)
        return
    
    # 确定使用哪种模式获取索引
    if args.retest_substring_false or RETEST_SUBSTRING_FALSE:
        # 重测子串匹配FALSE结果模式
        start_index = args.start_index if hasattr(args, 'start_index') else FALSE_START_INDEX
        case_sensitive = args.case_sensitive if hasattr(args, 'case_sensitive') else CASE_SENSITIVE
        print(f"🎯 子串匹配重测模式：重测包含标准答案但被评为FALSE的结果")
        print(f"📍 起始索引: {start_index}")
        print(f"🔤 区分大小写: {case_sensitive}")
        target_indices = find_false_with_substring_match(full_filename, start_index, case_sensitive)
        if not target_indices:
            print(f"❓ 未找到索引 >= {start_index} 的子串匹配FALSE结果")
            return
        print(f"📝 找到 {len(target_indices)} 个子串匹配FALSE结果待重测（索引范围: {min(target_indices)}-{max(target_indices)}）")
        
    elif args.retest_false or RETEST_FALSE_ONLY:
        # 重测FALSE结果模式
        start_index = args.start_index if hasattr(args, 'start_index') else FALSE_START_INDEX
        print(f"🔄 重测模式：从结果文件中提取FALSE结果进行重测")
        print(f"📍 起始索引: {start_index}")
        target_indices = get_false_indices(full_filename, start_index)
        if not target_indices:
            print(f"❓ 未找到索引 >= {start_index} 的FALSE结果")
            return
        print(f"📝 找到 {len(target_indices)} 个FALSE结果待重测（索引范围: {min(target_indices)}-{max(target_indices)}）")
        
    elif args.indices:
        # 命令行参数模式
        try:
            target_indices = parse_indices(args.indices)
            print(f"📝 使用命令行参数指定的索引")
        except Exception as e:
            print(f"❌ 命令行索引解析失败: {str(e)}")
            return
            
    else:
        # 代码定义模式
        target_indices = TARGET_INDICES
        print(f"📝 使用代码中定义的索引")
    
    print(f"🎯 目标索引: {target_indices}")
    print(f"📊 共 {len(target_indices)} 个问题待评测")
    print(f"📋 结果文件: {results_file}")
    
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
    
    # 检查索引有效性
    max_index = len(dataset) - 1
    invalid_indices = [idx for idx in target_indices if idx > max_index]
    if invalid_indices:
        print(f"❌ 无效索引: {invalid_indices} (数据集最大索引: {max_index})")
        return
    
    # 准备结果文件
    results_dir = os.path.join(Config.DATA_DIR, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    full_filename = os.path.join(results_dir, results_file)
    
    # 加载现有结果
    existing_results = load_existing_results(full_filename)
    print(f"📋 结果文件: {full_filename}")
    print(f"📝 现有结果数量: {len(existing_results)}")
    
    # 如果是重测模式，显示当前FALSE结果统计
    if args.retest_substring_false or RETEST_SUBSTRING_FALSE:
        false_count = len([r for r in existing_results if r.get('evaluation_decision') == 'FALSE'])
        total_count = len(existing_results)
        print(f"🔍 当前结果统计：FALSE: {false_count}, 总数: {total_count}")
        print(f"🎯 准备重测 {len(target_indices)} 个子串匹配FALSE结果")
    elif args.retest_false or RETEST_FALSE_ONLY:
        false_count = len([r for r in existing_results if r.get('evaluation_decision') == 'FALSE'])
        total_count = len(existing_results)
        print(f"🔍 当前结果统计：FALSE: {false_count}, 总数: {total_count}")
        print(f"🔄 准备重测 {len(target_indices)} 个FALSE结果")
    
    # 处理指定的索引
    processed_count = 0
    failed_count = 0
    
    for idx in tqdm(target_indices, desc="处理指定问题"):
        try:
            item = dataset[idx]
            result = process_single_item(item, idx)
            
            # 更新结果列表
            existing_results = update_result_by_index(existing_results, result)
            
            # 保存更新后的结果
            save_results(full_filename, existing_results)
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 处理问题 {idx} 失败: {str(e)}")
            failed_count += 1
            continue
    
    # 输出处理统计
    print("\n" + "="*60)
    print("📊 处理统计")
    print("="*60)
    print(f"目标问题数: {len(target_indices)}")
    print(f"处理成功: {processed_count}")
    print(f"处理失败: {failed_count}")
    
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
    
    # 显示本次处理的索引统计
    processed_results = [r for r in results if r.get('index') in target_indices]
    if processed_results:
        processed_correct = sum(1 for r in processed_results if r.get('evaluation_decision') == 'TRUE')
        processed_accuracy = processed_correct / len(processed_results) if processed_results else 0
        print(f"\n🎯 本次处理的索引统计:")
        print(f"  处理问题数: {len(processed_results)}")
        print(f"  正确答案: {processed_correct}")
        print(f"  准确率: {processed_accuracy:.2%}")
    
    print(f"\n💾 完整结果已保存到: {full_filename}")

if __name__ == "__main__":
    main()
