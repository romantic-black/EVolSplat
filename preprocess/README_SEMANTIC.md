# 语义分割模型使用说明

## 概述

已使用 **OneFormer** (Hugging Face Transformers) 替代了旧的 nvi_sem 和 SegFormer 模型。

## 快速开始

### 1. 安装依赖

```bash
cd /root/drivestudio-coding/third_party/EVolSplat/preprocess
bash install_semantic_deps.sh
```

### 2. 模型会自动下载

首次运行时，OneFormer 模型会自动从 Hugging Face Hub 下载（约 1.5 GB）。

### 3. 使用

在预处理流程中，当启用 `--use_semantics` 时，会自动使用 OneFormer 进行语义分割。

## 优势

- ✅ **更现代**: 2023年 CVPR 模型
- ✅ **更简单**: 通过 Hugging Face Transformers，无需复杂的依赖（如 Detectron2）
- ✅ **高性能**: Cityscapes mIoU 83.0%+
- ✅ **兼容性好**: 输出格式与原有系统完全兼容

## 更多信息

详见 `SEMANTIC_MODEL_UPDATE.md`

