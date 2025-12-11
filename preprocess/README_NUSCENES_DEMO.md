# nuScenes EVolSplat 预处理 Demo 使用指南

本指南说明如何使用 VSCode launch.json 配置来运行 nuScenes 数据集的小样本预处理 demo。

## 前置条件

### 1. 数据集位置
- 原始数据集路径: `/mnt/f/DataSet/nuScenes`
- 确保包含以下目录:
  - `v1.0-mini/` (或 `v1.0-trainval/`)
  - `samples/`
  - `sweeps/`

### 2. 安装依赖

运行检查脚本查看当前状态:
```bash
python3 third_party/EVolSplat/preprocess/check_nuscenes_setup.py
```

如果缺少依赖，运行安装脚本:
```bash
bash third_party/EVolSplat/preprocess/install_nuscenes_deps.sh
```

或手动安装:
```bash
pip install open3d nuscenes-devkit pyquaternion
```

### 3. 外部工具（可选）

如果需要生成深度图和语义分割图，需要设置以下环境变量:
- `METRIC3D_PATH`: Metric3D 工具路径
- `METRIC3D_MODEL_PATH`: Metric3D 模型路径
- `NVI_SEM_PATH`: NVI_SEM 工具路径
- `NVI_SEM_CHECKPOINT`: NVI_SEM 模型路径

**注意**: 如果不设置这些，预处理仍可运行，但会跳过深度和语义生成步骤。

## VSCode 配置说明

在 VSCode 的 "Run and Debug" 面板中，有以下配置可用:

### 1. Check NuScenes Setup
**用途**: 检查环境配置和数据集状态

**运行方式**: 选择此配置并点击运行

**输出**: 显示缺失的依赖、数据集结构状态等信息

### 2. Step 1: NuScenes Raw Data Preprocess (Mini Demo)
**用途**: 预处理原始 nuScenes 数据（如果尚未预处理）

**参数**:
- `--data_root`: `/mnt/f/DataSet/nuScenes` (原始数据路径)
- `--target_dir`: `/mnt/f/DataSet/nuScenes/processed` (输出路径)
- `--num_scenes`: `1` (处理 1 个场景用于 demo)
- `--interpolate_N`: `4` (插值到 10Hz)
- `--process_keys`: `images calib` (只处理图像和标定)

**输出**: 
- 预处理后的数据保存在 `/mnt/f/DataSet/nuScenes/processed_10Hz/mini/000/`
- 包含 `images/`, `extrinsics/`, `intrinsics/` 等目录

**运行时间**: 约 1-5 分钟（取决于场景大小）

### 3. Step 2: NuScenes EVolSplat Preprocess (Demo - Small Sample)
**用途**: EVolSplat 格式的预处理（小样本 demo）

**参数**:
- `--dataset`: `nuscenes`
- `--seq_id`: `0` (场景 ID)
- `--start_index`: `0` (起始帧)
- `--num_images`: `10` (处理 10 帧)
- `--root_dir`: `/mnt/f/DataSet/nuScenes/processed_10Hz/mini` (预处理数据路径)
- `--save_dir`: `${workspaceFolder}/data/evolsplat_nuscenes_preprocessed` (输出路径)
- `--pcd_sparsity`: `Drop50` (点云稀疏度)
- `--nuscenes_cameras`: `0` (只使用前置摄像头)
- `--depth_consistency`: 启用深度一致性检查

**输出**:
- 在 `${workspaceFolder}/data/evolsplat_nuscenes_preprocessed/seq_000_nuscenes_0000_10/` 目录下
- 包含 `transforms.json`, 图像文件, 点云文件等

**运行时间**: 约 2-10 分钟（取决于是否生成深度/语义）

### 4. NuScenes EVolSplat Preprocess (Full Scene with Depth & Semantic)
**用途**: 完整场景预处理（包含深度和语义）

**参数**: 
- `--num_images`: `40` (更多帧)
- `--use_semantic`: 生成语义分割
- `--use_metric_depth`: 生成深度图
- `--filter_sky`: 过滤天空区域

**注意**: 需要设置外部工具环境变量才能生成深度和语义

### 5. NuScenes EVolSplat Preprocess (Multi-Camera)
**用途**: 多相机预处理

**参数**:
- `--nuscenes_cameras`: `0,1,2` (前置、左前、右前摄像头)
- `--num_images`: `20`

## 使用流程

### 首次使用

1. **检查环境**
   - 运行 "Check NuScenes Setup" 配置
   - 查看输出，确认缺失项

2. **安装依赖**（如需要）
   ```bash
   bash third_party/EVolSplat/preprocess/install_nuscenes_deps.sh
   ```

3. **预处理原始数据**（如需要）
   - 运行 "Step 1: NuScenes Raw Data Preprocess (Mini Demo)"
   - 等待完成（约 1-5 分钟）

4. **运行 EVolSplat 预处理**
   - 运行 "Step 2: NuScenes EVolSplat Preprocess (Demo - Small Sample)"
   - 等待完成（约 2-10 分钟）

### 日常使用

如果数据已预处理，直接运行 "Step 2" 配置即可。

## 输出文件结构

```
${workspaceFolder}/data/evolsplat_nuscenes_preprocessed/
└── seq_000_nuscenes_0000_10/
    ├── transforms.json          # 相机参数和位姿
    ├── 000_0.png               # 图像文件
    ├── 001_0.png
    ├── ...
    ├── depth/                  # 深度图（如果生成）
    │   ├── 000_0.npy
    │   └── ...
    ├── semantic/               # 语义分割（如果生成）
    │   └── instance/
    │       ├── 000_0.png
    │       └── ...
    └── Drop50/                 # 点云文件
        └── 0.ply
```

## 常见问题

### Q: 提示 "Scene 000 not found"
**A**: 检查预处理数据是否存在:
- 确认路径: `/mnt/f/DataSet/nuScenes/processed_10Hz/mini/000/`
- 如果不存在，先运行 "Step 1" 配置

### Q: 提示 "Depth directory not found"
**A**: 这是正常的，如果不使用 `--use_metric_depth` 参数，预处理仍可运行。
如果需要深度图，需要设置 Metric3D 环境变量。

### Q: 内存不足
**A**: 可以减少处理的帧数:
- 修改 `--num_images` 参数，从 10 改为 5

### Q: 处理速度慢
**A**: 可以:
- 减少 `--num_images`
- 使用单个摄像头 (`--nuscenes_cameras 0`)
- 不使用深度和语义生成（去掉相关 flags）

## 下一步

预处理完成后，可以使用生成的数据进行 EVolSplat 训练。参考项目主文档了解训练流程。

