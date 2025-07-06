# Multi-Agent DeepResearch 智能体系统

一个基于多智能体的深度研究智能体系统，使用ReAct推理模式，集成了多种工具和检索能力。

## 🎯 系统特性

- **ReAct推理**: 使用DeepSeek-R1模型进行思考-行动-观察的推理循环
- **多工具集成**: 知识库搜索、网络搜索、计算器、文本摘要等
- **向量检索**: 基于FAISS和Jina embedding的语义搜索
- **智能重排序**: 使用Jina reranker优化搜索结果
- **记忆管理**: 存储和检索历史对话与推理轨迹
- **引用管理**: 自动生成和管理文献引用
- **FRAMES评测**: 兼容FRAMES基准测试

## 📁 项目结构

```
Multi-Agent-DeepResearch/
│
├── main.py                         # 系统入口
├── config.py                       # 配置文件
│
├── planner/
│   ├── planner.py                  # DeepSeek-R1调用器 + ReAct Prompt封装
│   └── prompt_templates.py         # ReAct Prompt模版
│
├── tools/
│   ├── __init__.py
│   ├── search_tool.py              # 内部知识库搜索
│   ├── web_search_tool.py          # 互联网搜索
│   ├── calculator_tool.py          # 简单计算器
│   └── summarizer_tool.py          # 文本摘要器
│
├── retriever/
│   ├── embedder.py                 # Jina embedding 文本嵌入
│   ├── build_index.py              # 构建 FAISS 索引
│   └── retriever.py                # 实时向量检索器
│
├── reranker/
│   └── reranker.py                 # Jina reranker 重排序
│
├── memory/
│   └── memory_manager.py           # 存储每轮对话与中间检索结果
│
├── citation/
│   └── citation_manager.py         # 引用文献跟踪与整理
│
├── agent/
│   └── main_agent.py               # 主Agent执行ReAct推理循环
│
├── evaluator/
│   └── evaluate.py                 # 评测脚本（兼容FRAMES评测）
│
├── data/
│   ├── frames_dataset/             # FRAMES数据集
│   ├── index/                      # FAISS索引缓存
│   └── memory_cache/               # Agent记忆缓存
│
└── README.md                       # 项目说明
```

## 🚀 快速开始

### 1. 环境配置
```bash
# 克隆并进入项目目录
cd Multi-Agent-DeepResearch

# 设置API密钥
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export JINA_API_KEY="your_jina_api_key"  # 可选
export BING_API_KEY="your_bing_api_key"  # 可选

# 一键安装和配置
bash setup.sh
```

### 2. 测试新系统
```bash
# 测试新的推理流程
python test_new_system.py

# 单次查询测试
python main.py --query "What is the relationship between AI and machine learning?"

# 包含Wikipedia链接的查询
python main.py --query "Tell me about https://en.wikipedia.org/wiki/Deep_learning"
```

### 3. FRAMES评估
```bash
# 评估系统性能（使用样例数据）
python evaluate_frames.py --dataset data/frames_dataset/sample_dataset.json

# 限制样本数量的快速测试
python evaluate_frames.py --dataset data/frames_dataset/sample_dataset.json --max-samples 10
```

### 4. 交互模式
```bash
# 启动交互式研究助手
python main.py --mode interactive
```

## 📊 使用示例

### 示例1: 包含Wikipedia链接的查询
```python
query = """
请分析人工智能的发展历程，参考以下资源：
https://en.wikipedia.org/wiki/Artificial_intelligence
https://en.wikipedia.org/wiki/History_of_artificial_intelligence
"""

# 系统会自动：
# 1. 提取两个Wikipedia链接
# 2. 获取页面内容并总结
# 3. 基于内容生成综合答案
```

### 示例2: 复杂研究查询
```python
query = "What are the latest breakthroughs in quantum computing and their potential applications?"

# 系统会自动：
# 1. 分解为子问题（量子计算原理、最新突破、应用领域等）
# 2. 在知识库中搜索，相关性不足时使用Wikipedia
# 3. 总结各部分信息
# 4. 生成完整答案和引用
```

## 🔧 配置说明

在 `config.py` 中配置以下参数：

- **API密钥**: DeepSeek、Jina、Bing等API的访问密钥
- **模型配置**: 嵌入模型、重排序模型等
- **检索参数**: Top-K数量、重排序数量等
- **Agent参数**: 最大迭代次数、温度等

## 🛠️ 功能模块

### 规划器 (Planner)
- 使用DeepSeek-R1模型进行ReAct推理
- 任务分解和步骤规划
- 反思和自我纠正机制

### 工具管理器 (Tools)
- **知识库搜索**: 基于FAISS的向量检索
- **网络搜索**: Bing API集成的实时搜索
- **计算器**: 安全的数学表达式计算
- **摘要器**: LLM和规则结合的文本摘要

### 检索系统 (Retriever)
- **嵌入器**: Jina embedding模型支持
- **索引构建**: 自动构建和更新FAISS索引
- **混合搜索**: 语义搜索+关键词搜索

### 重排序器 (Reranker)
- **Jina Reranker**: 基于查询-文档相关性重排序
- **简单重排序**: 规则基础的备用方案
- **混合重排序**: 结合多种排序策略

### 记忆管理 (Memory)
- **对话存储**: 持久化存储历史对话
- **检索记忆**: 基于相似性检索相关历史
- **会话管理**: 跟踪当前会话状态

### 引用管理 (Citation)
- **自动引用**: 从搜索结果自动生成引用
- **多种格式**: 支持APA、MLA等引用格式
- **验证检查**: 引用有效性验证

### 主智能体 (Agent)
- **ReAct循环**: 思考-行动-观察的推理模式
- **工具协调**: 智能选择和调用工具
- **上下文管理**: 动态管理推理上下文

### 评测器 (Evaluator)
- **FRAMES兼容**: 支持FRAMES基准测试
- **多维评测**: 准确率、F1、引用质量等
- **详细报告**: 生成全面的评测报告

## 📊 评测指标

系统支持以下评测指标：
- **准确率 (Accuracy)**: 基于关键词重叠的答案质量
- **精确匹配 (Exact Match)**: 与标准答案的完全匹配度
- **F1分数**: 精确率和召回率的调和平均
- **引用准确率**: 引用来源的准确性
- **响应时间**: 系统响应速度

## 🔄 工作流程

1. **接收查询**: 用户输入研究问题
2. **上下文构建**: 整合历史对话和背景信息
3. **ReAct推理**: 循环执行思考-行动-观察
4. **工具调用**: 根据需要调用搜索、计算等工具
5. **结果整合**: 汇总信息生成最终答案
6. **引用生成**: 自动添加文献引用
7. **记忆存储**: 保存推理过程到记忆系统

## 🔄 系统推理流程

新的多智能体推理系统采用以下智能流程：

### 1. 查询分析阶段
- **Wiki链接检测**: 如果查询包含Wikipedia链接，直接提取并处理
- **查询分解**: 如果没有链接，将复杂查询分解为3-5个子查询

### 2. 信息检索阶段
对每个子查询或Wiki链接：
- **知识库优先**: 首先在本地知识库中搜索
- **相关性判断**: 如果FAISS最高相关性 ≥ 0.7，使用知识库结果并通过reranker精排
- **Wikipedia后备**: 如果相关性 < 0.7，在英文Wikipedia上搜索
- **动态扩展**: 将新获取的Wikipedia文档加入知识库

### 3. 内容处理阶段
- **智能总结**: 使用Summarizer对长文档（>500字符）进行总结
- **上下文构建**: 将所有总结内容合并到推理上下文

### 4. 答案生成阶段
- **充分性判断**: 使用Planner判断信息是否足够回答问题
- **迭代推理**: 如果信息不足，生成新的查询方向继续搜索
- **强制终止**: 达到最大迭代次数时强制生成答案

### 5. 质量保证
- **引用生成**: 自动生成所有来源的引用
- **推理轨迹**: 记录完整的推理过程
- **记忆存储**: 保存查询历史和推理结果

## 🎯 核心优势

1. **智能路由**: 根据相关性自动选择知识库或Wikipedia
2. **动态扩展**: 实时扩充知识库内容
3. **迭代优化**: 多轮推理直到获得满意答案
4. **质量评估**: 内置答案充分性判断机制
5. **完整追踪**: 详细记录所有推理步骤

## 🔍 使用示例

### 交互式研究
```python
# 启动交互模式
python main.py --mode interactive

# 输入问题
> 请介绍量子计算的基本原理和最新进展

# 系统会执行以下步骤：
# 1. 分析问题，分解为子任务
# 2. 在知识库中搜索量子计算相关文档
# 3. 在网络上搜索最新进展
# 4. 整合信息生成综合答案
# 5. 提供引用来源
```

### 批量评测
```python
# 准备FRAMES数据集
# 运行评测
python main.py --mode evaluate --dataset data/frames_dataset/

# 查看结果
cat data/evaluation_results/frames_summary_*.json
```

## 🐛 故障排除

### 常见问题

1. **API密钥错误**
   ```bash
   # 检查环境变量
   echo $DEEPSEEK_API_KEY
   echo $JINA_API_KEY
   ```

2. **索引构建失败**
   ```bash
   # 检查数据目录
   ls -la data/frames_dataset/
   # 重新构建索引
   python retriever/build_index.py --data-dir data/frames_dataset/
   ```

3. **内存不足**
   - 减少Top-K参数
   - 使用CPU版本的FAISS
   - 分批处理大数据集

4. **网络连接问题**
   - 检查API服务状态
   - 配置代理（如需要）
   - 使用备用搜索工具

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进系统！

### 开发环境设置
```bash
git clone <repository>
cd Multi-Agent-DeepResearch
pip install -r requirements.txt
```

### 添加新工具
1. 在 `tools/` 目录创建新工具模块
2. 实现标准工具接口
3. 在 `main_agent.py` 中注册工具
4. 添加对应的测试

### 扩展评测指标
1. 在 `evaluator/evaluate.py` 中添加新指标函数
2. 更新 `_calculate_metrics()` 方法
3. 在报告中显示新指标

## 📄 许可证

MIT License

## 🙏 致谢

感谢以下开源项目：
- [DeepSeek](https://www.deepseek.com/) - 强大的语言模型
- [Jina AI](https://jina.ai/) - 嵌入和重排序模型
- [FAISS](https://github.com/facebookresearch/faiss) - 高效向量搜索
- [FRAMES](https://github.com/microsoft/FRAMES) - 研究基准测试

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- Issue追踪: GitHub Issues
- 邮箱: [您的邮箱]
- 微信群: [群二维码]

---

🚀 让我们一起构建更智能的研究助手！
