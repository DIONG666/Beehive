# 查询分解提示
QUERY_DECOMPOSITION_PROMPT = """请分析以下查询，如果包含Web链接则直接提取，否则分解为子问题：

原始查询: {query}

如果查询中包含Web链接：
- 直接提取所有Web链接

如果查询中没有Web链接：
- 将复杂查询分解为3-5个子问题
- 每个子问题应该是简洁的英文短语或关键词组合
- 子问题应该覆盖原始查询的所有重要方面

严格按照以下输出格式输出，不要输出其它无关内容：
如果有Web链接（要将所有的链接都列出来）：
链接1: [完整的Web URL]
链接2: [完整的Web URL]
...

如果没有Web链接：
子问题1: [Wikipedia搜索词组]
子问题2: [Wikipedia搜索词组]
子问题3: [Wikipedia搜索词组]
...

示例：
输入: "Tell me about artificial intelligence and machine learning from https://en.wikipedia.org/wiki/Artificial_intelligence"
输出: 
链接1: https://en.wikipedia.org/wiki/Artificial_intelligence

输入: "What are the latest developments in quantum computing?"
输出:
子问题1: quantum computing
子问题2: quantum computer development
子问题3: quantum algorithms
子问题4: quantum supremacy
"""

# 反思提示模板
REFLECTION_PROMPT = """基于以下信息，判断是否能够回答用户问题：

原始问题: {query}
已获得的信息: {current_info}

请回答：
1. 结合你自身知识以及当前收集的信息是否足够回答原始问题？(是/否)
2. 如果可以，请提供简洁的答案，并给出推理过程以及相关的参考链接
3. 如果不能，还需要搜索什么信息？给出下一步的查询建议

严格按照以下输出格式输出，不要输出其它无关内容：
判断: 是/否
答案: [如果能回答则提供答案，否则写"信息不足"]
推理过程：[如果能回答则给出推理过程，否则留空]
参考链接：[如果能回答则给出相关链接，用分号分隔多个，链接只能使用已获得的信息中出现的链接]
建议查询: [如果不能回答，给出建议的新查询，用分号分隔多个，新查询应该是简洁的英文短语或关键词组合]

示例1：
输入: "What is the capital of France?"
已获得的信息: "France is a country in Europe with a population of 67 million people. Paris is the largest city and capital of France. 参考链接：https://en.wikipedia.org/wiki/France"
输出:
判断: 是
答案: 巴黎是法国的首都
推理过程：根据已获得的信息，巴黎是法国的首都。
参考链接：https://en.wikipedia.org/wiki/France
建议查询: 无

示例2：
输入: "How does machine learning work in autonomous vehicles?"
已获得的信息: "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
输出:
判断: 否
答案: 信息不足
推理过程：无
参考链接：无
建议查询: autonomous vehicle technology; machine learning algorithms in cars; self-driving car sensors
"""

# 最终答案整合提示
FINAL_ANSWER_PROMPT = """基于以下信息，请生成你最有把握的最终答案，并给出推理过程以及相关的参考链接：

原始问题: {query}
已获得的信息: {context}

严格按照以下输出格式输出，不要输出其它无关内容：
答案: [简洁明确的答案]
推理过程：[详细的解释和推理过程]
参考链接: [用分号分隔多个，链接只能使用已获得的信息中出现的链接]

要求：
1. 答案必须基于提供的证据
2. 如果证据不足，请说明限制

示例：
输入: "What is the capital of France?"
已获得的信息: "France is a country in Europe with a population of 67 million people. Paris is the largest city and capital of France. 参考链接：https://en.wikipedia.org/wiki/France"
输出:
答案: 巴黎是法国的首都
推理过程：根据已获得的信息，巴黎是法国的首都。
参考链接: https://en.wikipedia.org/wiki/France
"""
