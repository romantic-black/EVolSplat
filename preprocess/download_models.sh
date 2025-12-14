#!/bin/bash
# 下载 Metric3D 模型权重
# 注意: OneFormer 模型会自动从 Hugging Face 下载，无需手动下载

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "下载 Metric3D 模型权重"
echo "=========================================="

# 创建模型保存目录
METRIC3D_MODEL_DIR="${SCRIPT_DIR}/metric3d/models"
mkdir -p "$METRIC3D_MODEL_DIR"

# 检查是否安装了 gdown
if ! command -v gdown &> /dev/null; then
    echo "安装 gdown..."
    pip install -q gdown
fi

# 下载 Metric3D 模型
echo ""
echo "下载 Metric3D 模型..."
METRIC3D_MODEL="${METRIC3D_MODEL_DIR}/metric_depth_vit_giant2_800k.pth"
if [ -f "$METRIC3D_MODEL" ]; then
    echo "✓ Metric3D 模型已存在: $METRIC3D_MODEL"
else
    echo "正在下载 Metric3D 模型 (File ID: 1KVINiBkVpJylx_6z1lAC7CQ4kmn-RJRN)..."
    gdown "1KVINiBkVpJylx_6z1lAC7CQ4kmn-RJRN" -O "$METRIC3D_MODEL" || {
        echo "警告: gdown 下载失败，请手动下载"
        echo "  链接: https://drive.google.com/file/d/1KVINiBkVpJylx_6z1lAC7CQ4kmn-RJRN/view?usp=drive_link"
        echo "  保存到: $METRIC3D_MODEL"
        exit 1
    }
    if [ -f "$METRIC3D_MODEL" ]; then
        echo "✓ Metric3D 模型下载完成 ($(du -h "$METRIC3D_MODEL" | cut -f1))"
    fi
fi

echo ""
echo "=========================================="
echo "模型下载完成"
echo "=========================================="
echo ""
echo "Metric3D 模型: $METRIC3D_MODEL"
echo ""
echo "环境变量设置:"
echo "  export METRIC3D_PATH=\"$SCRIPT_DIR/metric3d\""
echo "  export METRIC3D_MODEL_PATH=\"$METRIC3D_MODEL\""
echo ""
echo "注意: OneFormer 模型会自动从 Hugging Face 下载（首次运行时）"
echo "  模型名称: shi-labs/oneformer_cityscapes_swin_large"
echo "  下载位置: ~/.cache/huggingface/hub/"



