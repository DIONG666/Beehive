# Multi-Agent DeepResearch 智能研究系统

一个基于多智能体架构的深度研究系统，使用DeepSeek推理模型，集成知识库检索、网络搜索、智能摘要等功能，专为复杂研究任务设计。

## 🎯 核心特性

- **🧠 智能推理**: 基于DeepSeek-R1模型的多轮推理循环
- **🔍 多源检索**: 知识库检索 + 网络搜索的混合检索策略  
- **📝 智能摘要**: 支持分块并行处理的长文档摘要
- **🔄 自适应路由**: 根据相关性自动选择最佳信息源
- **💾 记忆管理**: 持久化存储推理历史和上下文
- **📊 基准评测**: 支持FRAMES等研究基准测试

## 🏗️ 系统架构

```
Multi-Agent-DeepResearch/
├── main.py                    # 系统入口和主控制器
├── config.py                  # 系统配置文件
├── requirements.txt           # 依赖包列表
├── setup.sh                   # 一键安装脚本
├── evaluate.py               # FRAMES基准评测脚本
│
├── agent/                     # 主智能体模块
│   └── main_agent.py         # 核心推理循环控制器
│
├── planner/                   # 推理规划模块
│   ├── planner.py            # DeepSeek推理调用器
│   └── prompt_templates.py   # 推理提示模板
│
├── tools/                     # 工具模块
│   ├── __init__.py
│   ├── search_tool.py        # 知识库搜索工具
│   ├── web_search_tool.py    # 网络搜索工具
│   └── summarizer_tool.py    # 智能摘要工具
│
├── retriever/                 # 检索系统
│   ├── embedder.py           # 文本嵌入服务
│   ├── build_index.py        # 索引构建工具
│   └── retriever.py          # 向量检索器
│
├── reranker/                  # 重排序模块
│   └── reranker.py           # 结果重排序器
│
├── memory/                    # 记忆管理
│   └── memory_manager.py     # 对话历史管理器
│
└── data/                      # 数据目录
    ├── knowledge_base/        # 知识库文档
    ├── index/                 # FAISS向量索引
    ├── memory_cache/          # 记忆缓存
    └── evaluation_results/    # 评测结果
```

## ⚡ 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd Multi-Agent-DeepResearch

# 设置API密钥
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export JINA_API_KEY="your_jina_api_key"        # 可选，用于嵌入和重排序

# 一键安装
bash setup.sh
```

### 2. 基础使用

```bash
# 单次查询
python main.py --query "人工智能的最新发展趋势是什么？"

# 交互模式
python main.py --mode interactive

# 包含Wikipedia链接的查询
python main.py --query "请分析 https://en.wikipedia.org/wiki/Machine_learning 这篇文章的主要内容"
```

### 3. 系统评测

```bash
# 运行FRAMES基准评测
python evaluate.py

# 查看评测结果
ls data/evaluation_results/
```

## 🔧 系统配置

在 `config.py` 中可以调整以下参数：

### API配置
```python
DEEPSEEK_API_KEY = "your_api_key"      # DeepSeek API密钥 (必需)
JINA_API_KEY = "your_jina_key"         # Jina API密钥 (可选)
```

### 推理参数
```python
MAX_ITERATIONS = 3                     # 最大推理轮次
MAX_CONTEXT_LENGTH = 8192             # 最大上下文长度
TEMPERATURE = 0.7                      # 生成温度
```

### 检索参数
```python
TOP_K = 20                            # 检索文档数量
RERANK_TOP_K = 5                      # 重排序后保留数量
EMBEDDING_DIM = 2048                  # 嵌入维度
```

## 🧠 智能推理流程

系统采用多轮自适应推理策略：

### 1. 查询分析
- **链接检测**: 自动识别和提取Wikipedia链接
- **查询分解**: 将复杂问题分解为可处理的子任务

### 2. 信息检索
- **知识库优先**: 首先搜索本地知识库
- **相关性评估**: 基于向量相似度判断信息质量
- **动态切换**: 相关性不足时自动切换到网络搜索

### 3. 内容处理
- **智能摘要**: 对长文档进行分块并行摘要
- **信息融合**: 合并多源信息构建完整上下文

### 4. 答案生成
- **充分性判断**: 评估信息是否足够回答问题
- **迭代改进**: 信息不足时自动扩展搜索
- **质量保证**: 生成引用和推理轨迹

## �️ 核心模块详解

### 主智能体 (MainAgent)
```python
# 执行推理任务
agent = MainAgent()
result = agent.execute_reasoning(query, context)
```

- 控制整个推理流程
- 协调各个工具模块
- 管理推理状态和上下文

### 推理规划器 (DeepSeekPlanner)
```python
# 查询分解
sub_queries = planner.decompose_query(query)

# 进度反思
reflection = planner.reflect_on_progress(query, context)

# 答案生成
answer = planner.generate_final_answer(query, context)
```

- 基于DeepSeek-R1模型
- 支持查询分解、进度评估、答案生成
- 内置重试机制确保响应质量

### 搜索工具 (SearchTools)
```python
# 知识库搜索
kb_result = search_tool.search(query)

# 网络搜索
web_result = web_search_tool.search(query)
```

- **知识库搜索**: FAISS向量检索 + Jina重排序
- **网络搜索**: Jina API实时搜索Wikipedia
- **智能路由**: 基于相关性自动选择搜索源

### 摘要工具 (SummarizerTool)
```python
# 普通摘要
summary = summarizer.summarize(text, max_length=1000)

# 分批摘要（支持并行处理）
summary = summarizer.batch_summarize(query, text, chunk_size=5000)
```

- 支持长文档分块处理
- 可配置摘要长度和风格
- 使用DeepSeek-Chat模型进行高质量摘要

### 记忆管理 (MemoryManager)
```python
# 添加记忆
memory_manager.add_memory_entry(query, context, answer)

# 检索相关历史
context = memory_manager.get_recent_context(num=3)
```

- 持久化存储对话历史
- 基于相似度检索相关记忆
- 自动管理会话状态

## 📊 使用示例

### 示例1: 简单问答
```python
system = MultiAgentResearchSystem()
result = system.research_query("什么是量子计算？")

print(result['answer'])      # 答案内容
print(result['citations'])   # 引用来源
```

### 示例2: 链接分析
```python
query = """
请分析以下文章的主要观点：
https://en.wikipedia.org/wiki/Artificial_intelligence
https://en.wikipedia.org/wiki/Machine_learning
"""

result = system.research_query(query)
# 系统会自动提取链接内容并进行综合分析
```

### 示例3: 交互式研究
```python
# 启动交互模式
python main.py --mode interactive

# 系统会提供持续的对话界面
> 请介绍深度学习的发展历程
> 它与传统机器学习的区别是什么？
> 请推荐一些学习资源
```

## 📈 性能特点

### 检索效率
- **向量检索**: 基于FAISS的高效相似度搜索
- **智能缓存**: 避免重复检索提升响应速度
- **并行处理**: 支持多查询并行处理

### 摘要质量
- **分块策略**: 5000字符块大小平衡质量和效率
- **并行摘要**: 多块同时处理大幅提升速度
- **二级压缩**: 先分块摘要再整体摘要保证质量

### 推理能力
- **多轮迭代**: 最多3轮推理确保答案完整性
- **自我评估**: 内置充分性判断避免信息不足
- **错误恢复**: 完善的异常处理和重试机制

## 🔍 评测与验证

### FRAMES基准测试
```bash
# 运行完整评测
python evaluate.py

# 查看结果统计
cat data/evaluation_results/frames_evaluation_multi_agent.json
```

### 评测指标
- **准确率**: 答案与标准答案的匹配度
- **引用质量**: 参考资料的相关性和可靠性
- **推理质量**: 推理过程的逻辑性和完整性
- **响应时间**: 平均查询处理时间

## � 高级功能

### 1. 批量处理
```python
# 批量处理多个查询
queries = ["问题1", "问题2", "问题3"]
results = []
for query in queries:
    result = system.research_query(query)
    results.append(result)
```

### 2. 自定义配置
```python
# 临时修改配置
Config.MAX_ITERATIONS = 5
Config.TOP_K = 30
Config.TEMPERATURE = 0.3
```

### 3. 错误处理
```python
try:
    result = system.research_query(query)
    if result.get('error'):
        print(f"处理出错: {result['answer']}")
except Exception as e:
    print(f"系统异常: {e}")
```

## � 故障排除

### 常见问题

**1. API密钥配置错误**
```bash
# 检查环境变量
echo $DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY="your_actual_key"
```

**2. 知识库为空**
```bash
# 检查知识库目录
ls -la data/knowledge_base/
# 添加文档后重建索引
python retriever/build_index.py
```

**3. 内存不足**
```python
# 在config.py中调整参数
TOP_K = 10                    # 减少检索数量
MAX_CONTEXT_LENGTH = 4096     # 减少上下文长度
```

**4. 网络连接问题**
```python
# 禁用网络搜索
ENABLE_WEB_SEARCH = False
```

## 🤝 开发指南

### 添加新工具
1. 在 `tools/` 目录创建新工具类
2. 实现标准接口方法
3. 在 `main_agent.py` 中注册工具

### 扩展评测指标
1. 在 `evaluate.py` 中添加新的评估函数
2. 更新统计报告生成逻辑
3. 添加对应的测试用例

### 自定义推理策略
1. 修改 `planner/prompt_templates.py` 中的提示模板
2. 调整 `main_agent.py` 中的推理循环逻辑
3. 在 `config.py` 中添加相关参数

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢以下开源项目和服务：

- [DeepSeek](https://www.deepseek.com/) - 强大的推理语言模型
- [Jina AI](https://jina.ai/) - 高质量的嵌入和重排序服务  
- [FAISS](https://github.com/facebookresearch/faiss) - 高效的向量相似度搜索
- [FRAMES](https://github.com/microsoft/FRAMES) - 多步推理评测基准

---

� **让AI成为你的智能研究伙伴！**
