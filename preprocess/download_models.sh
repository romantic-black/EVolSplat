#!/bin/bash
# 下载 Metric3D 和 NVI_SEM 模型权重

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "下载 Metric3D 和 NVI_SEM 模型权重"
echo "=========================================="

# 创建模型保存目录
METRIC3D_MODEL_DIR="${SCRIPT_DIR}/metric3d/models"
NVI_SEM_MODEL_DIR="${SCRIPT_DIR}/nvi_sem/checkpoints"
mkdir -p "$METRIC3D_MODEL_DIR"
mkdir -p "$NVI_SEM_MODEL_DIR"

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

# 下载 NVI_SEM 模型
echo ""
echo "下载 NVI_SEM 模型..."
NVI_SEM_MODEL="${NVI_SEM_MODEL_DIR}/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth"
if [ -f "$NVI_SEM_MODEL" ]; then
    echo "✓ NVI_SEM 模型已存在: $NVI_SEM_MODEL"
else
    echo "正在下载 NVI_SEM 权重包 (File ID: 1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U)..."
    TEMP_ZIP="${NVI_SEM_MODEL_DIR}/nvi_sem_weights.zip"
    gdown "1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U" -O "$TEMP_ZIP" || {
        echo "警告: gdown 下载失败，请手动下载"
        echo "  链接: https://drive.google.com/file/d/1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U/view?usp=sharing"
        echo "  解压后找到权重文件并保存到: $NVI_SEM_MODEL"
        exit 1
    }
    
    if [ -f "$TEMP_ZIP" ]; then
        echo "解压 NVI_SEM 权重..."
        cd "$NVI_SEM_MODEL_DIR"
        unzip -q -o "$TEMP_ZIP" 2>/dev/null || {
            echo "尝试使用 Python 解压..."
            python3 -c "import zipfile; zipfile.ZipFile('$TEMP_ZIP').extractall('.')" || true
        }
        rm -f "$TEMP_ZIP"
        cd "$SCRIPT_DIR"
        
        # 查找权重文件
        FOUND_MODEL=$(find "$NVI_SEM_MODEL_DIR" -name "*cityscapes*ocrnet*HRNet*Mscale*.pth" -o -name "*outstanding*turtle*.pth" 2>/dev/null | head -1)
        if [ -n "$FOUND_MODEL" ] && [ -f "$FOUND_MODEL" ]; then
            mv "$FOUND_MODEL" "$NVI_SEM_MODEL"
            echo "✓ NVI_SEM 模型提取完成 ($(du -h "$NVI_SEM_MODEL" | cut -f1))"
        else
            echo "警告: 未找到目标权重文件，请手动检查解压后的文件"
            ls -la "$NVI_SEM_MODEL_DIR" | head -10
        fi
    fi
fi

echo ""
echo "=========================================="
echo "模型下载完成"
echo "=========================================="
echo ""
echo "Metric3D 模型: $METRIC3D_MODEL"
echo "NVI_SEM 模型: $NVI_SEM_MODEL"
echo ""
echo "环境变量设置:"
echo "  export METRIC3D_PATH=\"$SCRIPT_DIR/metric3d\""
echo "  export METRIC3D_MODEL_PATH=\"$METRIC3D_MODEL\""
echo "  export NVI_SEM_PATH=\"$SCRIPT_DIR/nvi_sem\""
echo "  export NVI_SEM_CHECKPOINT=\"$NVI_SEM_MODEL\""



