#!/bin/bash
# 安装 nuScenes EVolSplat 预处理所需的依赖

set -e

echo "=========================================="
echo "安装 nuScenes EVolSplat 预处理依赖"
echo "=========================================="

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

echo "✓ Python 已安装: $(python3 --version)"

# 安装基本依赖
echo ""
echo "安装 Python 包..."
pip install -q open3d nuscenes-devkit pyquaternion

echo ""
echo "检查安装..."
python3 -c "import open3d; print('✓ open3d:', open3d.__version__)" || echo "✗ open3d 安装失败"
python3 -c "import nuscenes; print('✓ nuscenes-devkit 已安装')" || echo "✗ nuscenes-devkit 安装失败"

echo ""
echo "=========================================="
echo "依赖安装完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 运行检查脚本: python3 third_party/EVolSplat/preprocess/check_nuscenes_setup.py"
echo "2. 如果预处理数据不存在，运行 VSCode 配置: 'Step 1: NuScenes Raw Data Preprocess (Mini Demo)'"
echo "3. 然后运行 EVolSplat 预处理: 'Step 2: NuScenes EVolSplat Preprocess (Demo - Small Sample)'"

