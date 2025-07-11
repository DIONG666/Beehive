# Beehive: An Efficient Multi-Agent Research System

蜂巢（Beehive）是一个基于多智能体协作与多轮高效检索推理的深度研究系统。使用DeepSeek推理模型，集成知识库检索、网络搜索、智能摘要等功能，专为复杂研究任务设计。

## 🚀 核心特性

- **🧠 智能推理**: 基于DeepSeek-R1模型的多轮推理循环
- **🔍 多源检索**: 知识库检索 + 网络搜索的混合检索策略  
- **📝 智能摘要**: 支持分块并行处理的长文档摘要
- **🔄 自适应路由**: 根据相关性自动选择最佳信息源
- **💾 记忆管理**: 持久化存储推理历史和上下文

## ⚡ 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd Multi-Agent-DeepResearch

# 设置API密钥
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export JINA_API_KEY="your_jina_api_key" 

# 一键安装
bash setup.sh
```

### 2. 基础使用

```bash
# 单次查询
python main.py --query "人工智能的最新发展趋势是什么？"

# 交互模式
python main.py --mode interactive

# 包含指定链接的查询
python main.py --query "请分析 https://en.wikipedia.org/wiki/Machine_learning 这篇文章的主要内容"
```

### 3. 系统评测

```bash
# 运行FRAMES基准评测
python evaluate.py
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
