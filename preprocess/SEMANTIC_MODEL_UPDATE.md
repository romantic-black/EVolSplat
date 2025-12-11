# 语义分割模型更新说明

## 变更内容

已将从 **nvi_sem** 和 **SegFormer** 替换为 **OneFormer**（通过 Hugging Face Transformers）。

## OneFormer 简介

- **OneFormer** 是 2023 年 CVPR 的模型，由 Facebook AI Research 发布
- 统一的图像分割框架，支持语义、实例、全景分割
- 在 Cityscapes 数据集上表现优异（mIoU 83.0+）
- 使用 Hugging Face Transformers，安装和使用更简单

## 模型信息

- **模型名称**: `shi-labs/oneformer_cityscapes_swin_large`
- **任务**: 语义分割（semantic segmentation）
- **类别**: Cityscapes 19类（包括天空、车辆、行人等）
- **输出格式**: PNG 图像，像素值为类别ID (0-18)

## 安装依赖

运行以下脚本安装所需依赖：

```bash
cd /root/drivestudio-coding/third_party/EVolSplat/preprocess
bash install_semantic_deps.sh
```

或者手动安装：

```bash
pip install transformers torch pillow opencv-python-headless tqdm
```

## 使用方法

### 1. 在预处理流程中使用

语义分割会自动在预处理流程中调用（当 `--use_semantics` 标志启用时）。

### 2. 独立使用

```bash
python gen_semantic_oneformer.py \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --model_name shi-labs/oneformer_cityscapes_swin_large \
    --task semantic \
    --device cuda \
    --gpu_id 0
```

### 3. 环境变量配置

在 VSCode `launch.json` 中已配置：

```json
"ONEFORMER_MODEL_NAME": "shi-labs/oneformer_cityscapes_swin_large",
"SEMANTIC_GPU_ID": "0"
```

或在命令行中设置：

```bash
export ONEFORMER_MODEL_NAME="shi-labs/oneformer_cityscapes_swin_large"
export SEMANTIC_GPU_ID="0"
```

## 输出格式

输出保存到 `{output_dir}/semantic/instance/` 目录：
- 每个输入图像对应一个 PNG 文件
- 像素值为类别ID（0-18，对应 Cityscapes 19类）
- 文件名与输入图像相同，扩展名为 `.png`

### Cityscapes 类别ID映射

- 0: road
- 1: sidewalk
- 2: building
- 3: wall
- 4: fence
- 5: pole
- 6: traffic light
- 7: traffic sign
- 8: vegetation
- 9: terrain
- **10: sky** ← 用于天空过滤
- 11: person
- 12: rider
- **13: car**
- **14: truck**
- **15: bus**
- 16: train
- 17: motorcycle
- 18: bicycle

**动态物体类别**:
- Vehicles: 13, 14, 15 (car, truck, bus)
- Humans: 11, 12, 17, 18 (person, rider, motorcycle, bicycle)

## 模型下载

模型会在首次运行时自动从 Hugging Face Hub 下载（约 1.5 GB）。

如需手动下载或使用其他模型：

```python
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

# 下载并缓存模型
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
```

## 兼容性

- ✅ 与现有的 `gen_sky_mask()` 方法完全兼容
- ✅ 输出格式与 nvi_sem 相同（PNG，像素值为类别ID）
- ✅ 支持 Cityscapes 19类标准类别定义
- ✅ 支持 GPU 和 CPU 推理

## 性能对比

| 模型 | 发布时间 | Cityscapes mIoU | 依赖 |
|------|---------|----------------|------|
| nvi_sem | 2020 | ~83% | 旧版本 PyTorch |
| SegFormer | 2021 | 83.2% | mmcv-full 1.2.7, PyTorch 1.8 |
| **OneFormer** | **2023** | **83.0%+** | **Hugging Face Transformers** |

## 故障排除

### 1. 模型下载失败

如果自动下载失败，可以手动下载：

```bash
# 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download shi-labs/oneformer_cityscapes_swin_large --local-dir ./oneformer_model
```

### 2. GPU 内存不足

使用较小的模型：

```bash
export ONEFORMER_MODEL_NAME="shi-labs/oneformer_cityscapes_swin_base"
```

或使用 CPU（较慢）：

```bash
--device cpu
```

### 3. 输出格式不匹配

确保使用 `--task semantic` 参数，而不是 `panoptic` 或 `instance`。

## 更新日志

- 2024-12-10: 从 nvi_sem 迁移到 OneFormer (Hugging Face)
- 保持向后兼容的输出格式
- 简化依赖管理（无需 Detectron2 或特殊 PyTorch 版本）

