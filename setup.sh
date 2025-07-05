  #!/bin/bash

# Multi-Agent DeepResearch 快速启动脚本

echo "🚀 Multi-Agent DeepResearch 系统启动"
echo "======================================"

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "Python版本: $python_version"

# 检查依赖
echo "🔍 检查依赖..."
missing_deps=""

# 检查核心依赖
if ! python3 -c "import openai" 2>/dev/null; then
    missing_deps="$missing_deps openai"
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    missing_deps="$missing_deps numpy"
fi

if ! python3 -c "import requests" 2>/dev/null; then
    missing_deps="$missing_deps requests"
fi

if ! python3 -c "import aiohttp" 2>/dev/null; then
    missing_deps="$missing_deps aiohttp"
fi

# 安装缺失的依赖
if [ -n "$missing_deps" ]; then
    echo "⚠️ 缺少以下依赖:$missing_deps"
    echo "正在安装..."
    pip install $missing_deps
fi

# 检查环境变量
echo "🔑 检查API密钥..."
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️ 警告: DEEPSEEK_API_KEY 未设置"
    echo "请设置环境变量: export DEEPSEEK_API_KEY='your_key'"
fi

if [ -z "$JINA_API_KEY" ]; then
    echo "⚠️ 警告: JINA_API_KEY 未设置"
    echo "请设置环境变量: export JINA_API_KEY='your_key'"
fi

# 检查数据目录
echo "📁 检查数据目录..."
if [ ! -d "data/frames_dataset" ]; then
    echo "创建数据目录..."
    mkdir -p data/frames_dataset
fi

# 构建索引（如果有数据文件）
if ls data/frames_dataset/*.{txt,md,json} 1> /dev/null 2>&1; then
    echo "🔧 检测到数据文件，构建索引..."
    python3 retriever/build_index.py --data-dir data/frames_dataset/
else
    echo "⚠️ 未发现数据文件，请将文档放入 data/frames_dataset/ 目录"
fi

echo ""
echo "✅ 系统准备完成！"
echo ""
echo "🎯 使用方式："
echo "1. 交互模式: python3 main.py --mode interactive"
echo "2. 单次查询: python3 main.py --query '你的问题'"
echo "3. 评测模式: python3 main.py --mode evaluate --dataset data/frames_dataset/"
echo ""
echo "📚 更多信息请查看 README.md"
