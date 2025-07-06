"""
新系统流程演示脚本
"""
import asyncio
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from agent.main_agent import MainAgent


async def test_new_system():
    """测试新的系统流程"""
    print("🚀 新多智能体研究系统演示")
    print("=" * 50)
    
    # 检查基本配置
    print("📋 检查系统配置...")
    config = Config()
    
    if not config.DEEPSEEK_API_KEY:
        print("⚠️ 警告: DEEPSEEK_API_KEY 未设置")
        print("请设置环境变量: export DEEPSEEK_API_KEY='your_key'")
        return
    
    print("✅ 配置检查完成")
    
    # 初始化主智能体
    print("\n🤖 初始化主智能体...")
    agent = MainAgent()
    
    # 测试查询分解（包含Wikipedia链接）
    print("\n📋 测试1: 包含Wikipedia链接的查询")
    query_with_links = """
    请告诉我关于人工智能的信息，可以参考以下资源：
    https://en.wikipedia.org/wiki/Artificial_intelligence
    https://en.wikipedia.org/wiki/Machine_learning
    """
    
    try:
        result1 = await agent.execute_reasoning(query_with_links)
        print("✅ 测试1完成")
        print(f"答案长度: {len(result1.get('answer', ''))}")
        print(f"引用数量: {len(result1.get('citations', []))}")
        print(f"推理步骤: {len(result1.get('reasoning_trace', []))}")
    except Exception as e:
        print(f"❌ 测试1失败: {str(e)}")
    
    # 测试查询分解（不包含链接）
    print("\n📋 测试2: 普通查询分解")
    normal_query = "What are the main types of machine learning algorithms and their applications?"
    
    try:
        result2 = await agent.execute_reasoning(normal_query)
        print("✅ 测试2完成")
        print(f"答案长度: {len(result2.get('answer', ''))}")
        print(f"引用数量: {len(result2.get('citations', []))}")
        print(f"推理步骤: {len(result2.get('reasoning_trace', []))}")
        
        # 显示推理轨迹
        print("\n📝 推理轨迹:")
        for i, step in enumerate(result2.get('reasoning_trace', [])[:3], 1):
            print(f"{i}. {step[:100]}...")
        
    except Exception as e:
        print(f"❌ 测试2失败: {str(e)}")
    
    # 显示智能体信息
    print("\n🔍 智能体信息:")
    agent_info = agent.get_agent_info()
    print(f"类型: {agent_info['agent_type']}")
    print(f"可用工具: {agent_info['available_tools']}")
    print(f"最大迭代次数: {agent_info['max_iterations']}")
    print(f"能力列表: {', '.join(agent_info['reasoning_capabilities'])}")


async def test_frames_like_query():
    """测试类似FRAMES格式的查询"""
    print("\n🎯 测试FRAMES风格查询")
    print("=" * 50)
    
    agent = MainAgent()
    
    # 模拟FRAMES数据集中的查询
    frames_query = """
    What is the relationship between deep learning and neural networks? 
    Please explain the key concepts and provide examples of applications.
    """
    
    try:
        result = await agent.execute_reasoning(frames_query)
        
        print("📊 查询结果:")
        print(f"答案: {result.get('answer', 'No answer')[:200]}...")
        print(f"来源数量: {len(result.get('search_results', []))}")
        print(f"迭代次数: {result.get('metadata', {}).get('iterations', 0)}")
        
        # 评估答案质量指标
        answer = result.get('answer', '')
        citations = result.get('citations', [])
        
        quality_metrics = {
            'has_answer': len(answer) > 50,
            'has_citations': len(citations) > 0,
            'comprehensive': len(answer) > 200,
            'structured': '。' in answer or '.' in answer,
            'evidence_based': len(result.get('search_results', [])) > 0
        }
        
        print("\n📈 答案质量评估:")
        for metric, value in quality_metrics.items():
            status = "✅" if value else "❌"
            print(f"{status} {metric}: {value}")
        
        overall_score = sum(quality_metrics.values()) / len(quality_metrics)
        print(f"\n🎯 总体质量分数: {overall_score:.2f}")
        
    except Exception as e:
        print(f"❌ FRAMES测试失败: {str(e)}")


def check_environment():
    """检查环境设置"""
    print("🔧 环境检查")
    print("=" * 30)
    
    required_vars = ['DEEPSEEK_API_KEY']
    optional_vars = ['JINA_API_KEY', 'BING_API_KEY']
    
    print("必需的环境变量:")
    for var in required_vars:
        value = os.getenv(var)
        status = "✅" if value else "❌"
        print(f"{status} {var}: {'已设置' if value else '未设置'}")
    
    print("\n可选的环境变量:")
    for var in optional_vars:
        value = os.getenv(var)
        status = "✅" if value else "⚠️"
        print(f"{status} {var}: {'已设置' if value else '未设置'}")
    
    print("\n📁 数据目录:")
    data_dir = "data/frames_dataset"
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"✅ {data_dir}: {len(files)} 个文件")
        for file in files[:3]:
            print(f"   - {file}")
        if len(files) > 3:
            print(f"   ... 还有 {len(files) - 3} 个文件")
    else:
        print(f"❌ {data_dir}: 目录不存在")


async def main():
    """主函数"""
    print("🎬 新多智能体研究系统测试")
    print("开始时间:", end=" ")
    from datetime import datetime
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # 环境检查
    check_environment()
    
    # 基本系统测试
    await test_new_system()
    
    # FRAMES风格测试
    await test_frames_like_query()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成！")
    print("💡 提示:")
    print("  - 如果出现API错误，请检查密钥设置")
    print("  - 如果没有找到文档，请运行 setup.sh 构建索引")
    print("  - 完整评测请使用 evaluate_frames.py")


if __name__ == "__main__":
    asyncio.run(main())
