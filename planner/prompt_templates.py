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
<link>完整的Web URL</link>
<link>完整的Web URL</link>
...

如果没有Web链接：
<subquery>Wikipedia搜索词组</subquery>
<subquery>Wikipedia搜索词组</subquery>
<subquery>Wikipedia搜索词组</subquery>
...

示例：
输入: "Tell me about artificial intelligence and machine learning from https://en.wikipedia.org/wiki/Artificial_intelligence"
输出: 
<link>https://en.wikipedia.org/wiki/Artificial_intelligence</link>

输入: "What are the latest developments in quantum computing?"
输出:
<subquery>quantum computing</subquery>
<subquery>quantum computer development</subquery>
<subquery>quantum algorithms</subquery>
<subquery>quantum supremacy</subquery>
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
<judgment>是/否</judgment>
<answer>如果能回答则提供答案，否则写"信息不足"</answer>
<reasoning>如果能回答则给出推理过程，否则留空</reasoning>
<citations>如果能回答则给出相关链接，多个链接用分号分隔，链接只能使用已获得的信息中出现的链接</citations>
<suggestions>如果不能回答，给出建议的新查询，多个查询用分号分隔，新查询应该是简洁的英文短语或关键词组合</suggestions>

示例1：
输入: "What is the capital of France?"
已获得的信息: "France is a country in Europe with a population of 67 million people. Paris is the largest city and capital of France. 参考链接：https://en.wikipedia.org/wiki/France"
输出:
<judgment>是</judgment>
<answer>巴黎是法国的首都</answer>
<reasoning>根据已获得的信息，巴黎是法国的首都。</reasoning>
<citations>https://en.wikipedia.org/wiki/France</citations>
<suggestions></suggestions>

示例2：
输入: "How does machine learning work in autonomous vehicles?"
已获得的信息: "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
输出:
<judgment>否</judgment>
<answer>信息不足</answer>
<reasoning></reasoning>
<citations></citations>
<suggestions>autonomous vehicle technology; machine learning algorithms in cars; self-driving car sensors</suggestions>
"""

# 最终答案整合提示
FINAL_ANSWER_PROMPT = """基于以下信息，请生成你最有把握的最终答案，并给出推理过程以及相关的参考链接：

原始问题: {query}
已获得的信息: {context}

严格按照以下输出格式输出，不要输出其它无关内容：
<answer>简洁明确的答案</answer>
<reasoning>详细的解释和推理过程</reasoning>
<citations>多个链接用分号分隔，链接只能使用已获得的信息中出现的链接</citations>

要求：
1. 答案必须基于提供的证据
2. 如果证据不足，请说明限制

示例：
输入: "What is the capital of France?"
已获得的信息: "France is a country in Europe with a population of 67 million people. Paris is the largest city and capital of France. 参考链接：https://en.wikipedia.org/wiki/France"
输出:
<answer>巴黎是法国的首都</answer>
<reasoning>根据已获得的信息，巴黎是法国的首都。</reasoning>
<citations>https://en.wikipedia.org/wiki/France</citations>
"""
