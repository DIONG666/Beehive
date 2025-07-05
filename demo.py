#!/usr/bin/env python3
"""
Multi-Agent DeepResearch 系统演示脚本
"""
import asyncio
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.main_agent import MainAgent
from config import Config


async def demo_basic_functionality():
    """演示基本功能"""
    print("🎭 Multi-Agent DeepResearch 系统演示")
    print("=" * 50)
    
    # 创建主智能体
    agent = MainAgent()
    
    # 演示查询列表
    demo_queries = [
        "什么是人工智能？",
        "深度学习有哪些主要优势？",
        "计算 15 * 37 的结果",
        "请简要介绍神经网络的基本原理"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 演示查询 {i}: {query}")
        print("-" * 40)
        
        try:
            # 执行查询
            result = await agent.execute_reasoning(query)
            
            # 显示结果
            print(f"💡 答案: {result.get('answer', '无答案')}")
            
            if result.get('citations'):
                print(f"📚 引用: {len(result['citations'])} 个来源")
            
            if result.get('reasoning_trace'):
                print(f"🧠 推理步骤: {len(result['reasoning_trace'])} 步")
            
            print(f"⏱️ 状态: {'成功' if not result.get('error') else '失败'}")
            
        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
        
        print()
    
    # 显示系统信息
    print("📊 系统信息:")
    agent_info = agent.get_agent_info()
    print(f"  可用工具: {', '.join(agent_info['available_tools'])}")
    print(f"  最大迭代: {agent_info['max_iterations']}")
    print(f"  记忆条目: {agent_info['total_memory_entries']}")


async def demo_tool_functionality():
    """演示工具功能"""
    print("\n🛠️ 工具功能演示")
    print("=" * 50)
    
    from tools.calculator_tool import CalculatorTool
    from tools.summarizer_tool import SummarizerTool
    
    # 演示计算器
    print("🧮 计算器演示:")
    calc = CalculatorTool()
    
    expressions = ["2 + 3 * 4", "sqrt(16)", "2^10"]
    for expr in expressions:
        result = calc.calculate(expr)
        print(f"  {expr} = {result.get('result', '错误')}")
    
    # 演示摘要器
    print("\n📝 摘要器演示:")
    summarizer = SummarizerTool()
    
    long_text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    旨在创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、
    问题解决、感知、语言理解等。人工智能技术已经在医疗诊断、自动驾驶、
    语音识别、图像识别、自然语言处理等多个领域取得了重要突破。
    随着深度学习和大型语言模型的发展，AI系统在理解和生成自然语言方面
    展现出了前所未有的能力。
    """
    
    try:
        summary_result = await summarizer.summarize(long_text, max_length=100)
        print(f"  原文长度: {summary_result.get('original_length', 0)} 字符")
        print(f"  摘要长度: {summary_result.get('summary_length', 0)} 字符")
        print(f"  压缩比: {summary_result.get('compression_ratio', 0):.2f}")
        print(f"  摘要: {summary_result.get('summary', '无摘要')[:100]}...")
    except Exception as e:
        print(f"  摘要失败: {str(e)}")


def check_environment():
    """检查环境配置"""
    print("🔧 环境检查")
    print("=" * 50)
    
    config = Config()
    
    # 检查API密钥
    api_keys = {
        'DeepSeek': config.DEEPSEEK_API_KEY,
        'Jina': config.JINA_API_KEY,
        'Bing': config.BING_API_KEY
    }
    
    for name, key in api_keys.items():
        status = "✅ 已配置" if key else "❌ 未配置"
        print(f"  {name} API: {status}")
    
    # 检查目录
    directories = [
        config.DATA_DIR,
        config.FRAMES_DATASET_DIR,
        config.INDEX_DIR,
        config.MEMORY_CACHE_DIR
    ]
    
    print("\n📁 目录检查:")
    for directory in directories:
        exists = os.path.exists(directory)
        status = "✅ 存在" if exists else "❌ 不存在"
        print(f"  {os.path.basename(directory)}: {status}")
    
    # 检查依赖
    print("\n📦 依赖检查:")
    dependencies = ['openai', 'numpy', 'requests', 'aiohttp']
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  {dep}: ✅ 已安装")
        except ImportError:
            print(f"  {dep}: ❌ 未安装")


async def main():
    """主演示函数"""
    print("🎬 Multi-Agent DeepResearch 系统演示开始")
    print("=" * 60)
    
    # 环境检查
    check_environment()
    
    print("\n" + "=" * 60)
    
    # 检查是否有API密钥
    config = Config()
    if not config.DEEPSEEK_API_KEY:
        print("⚠️ 警告: 未配置DeepSeek API密钥")
        print("这将影响LLM相关功能的正常使用")
        print("请设置环境变量: export DEEPSEEK_API_KEY='your_key'")
        print("\n继续演示基础功能...")
    
    # 工具功能演示
    await demo_tool_functionality()
    
    # 如果有API密钥，演示完整功能
    if config.DEEPSEEK_API_KEY:
        await demo_basic_functionality()
    else:
        print("\n🔒 由于缺少API密钥，跳过完整系统演示")
        print("配置API密钥后可运行完整功能")
    
    print("\n🎉 演示完成！")
    print("更多功能请参考 README.md 或运行:")
    print("  python main.py --mode interactive")


if __name__ == "__main__":
    asyncio.run(main())
