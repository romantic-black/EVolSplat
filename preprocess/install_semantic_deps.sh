#!/bin/bash
# 安装语义分割所需的依赖 (OneFormer via Hugging Face Transformers)

set -e

echo "=========================================="
echo "安装 OneFormer 语义分割依赖"
echo "=========================================="

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

echo ""
echo "1. 安装 PyTorch (如果未安装)..."
python3 -c "import torch; print(f'PyTorch 已安装: {torch.__version__}')" 2>/dev/null || {
    echo "  安装 PyTorch (CPU版本，如需GPU请手动安装)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

echo ""
echo "2. 安装 Hugging Face Transformers..."
pip install transformers pillow opencv-python-headless tqdm

echo ""
echo "3. 验证安装..."
python3 -c "
import torch
import transformers
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ OneFormer 模型可以加载')
"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "OneFormer 模型将在首次运行时自动从 Hugging Face 下载"
echo "模型名称: shi-labs/oneformer_cityscapes_swin_large"
echo ""
echo "环境变量设置:"
echo "  export ONEFORMER_MODEL_NAME=\"shi-labs/oneformer_cityscapes_swin_large\""
echo "  export SEMANTIC_GPU_ID=\"0\""

