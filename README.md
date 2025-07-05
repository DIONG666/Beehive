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

### 1. 安装依赖

```bash
pip install openai
pip install numpy
pip install faiss-cpu  # 或 faiss-gpu
pip install aiohttp
pip install requests
```

### 2. 配置环境变量

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export JINA_API_KEY="your_jina_api_key"
export BING_API_KEY="your_bing_api_key"  # 可选
```

### 3. 构建知识库索引

```bash
# 将您的文档放入 data/frames_dataset/ 目录
python retriever/build_index.py --data-dir data/frames_dataset/
```

### 4. 运行系统

```bash
# 交互模式
python main.py --mode interactive

# 单次查询
python main.py --query "人工智能的最新发展趋势是什么？"

# FRAMES评测
python main.py --mode evaluate --dataset data/frames_dataset/
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

## 🎛️ 高级功能

### 智能工具选择
系统能够根据查询类型智能选择最合适的工具：
- 事实性问题 → 知识库搜索
- 时事问题 → 网络搜索  
- 数值计算 → 计算器
- 长文档 → 摘要器

### 自适应上下文管理
动态管理推理上下文长度，避免超出模型限制：
- 保留关键信息
- 压缩中间结果
- 智能截断策略

### 多轮对话支持
支持多轮对话，能够：
- 引用历史对话
- 追踪话题变化
- 维护上下文连贯性

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
