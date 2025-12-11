# EVolSplat 项目文件和组件依赖结构文档

## 目录
1. [项目整体结构](#项目整体结构)
2. [核心模块详解](#核心模块详解)
3. [组件依赖关系图](#组件依赖关系图)
4. [数据流和调用链](#数据流和调用链)
5. [外部依赖库](#外部依赖库)
6. [配置文件结构](#配置文件结构)

---

## 项目整体结构

```
EVolSplat/
├── nerfstudio/              # 核心代码库（基于nerfstudio框架）
│   ├── models/              # 模型定义
│   ├── pipelines/           # 训练/推理流水线
│   ├── fields/              # 场（Field）定义
│   ├── field_components/    # 场组件（MLP、编码器等）
│   ├── model_components/    # 模型组件（渲染器、投影器等）
│   ├── data/                # 数据处理
│   ├── cameras/             # 相机相关
│   ├── engine/              # 训练引擎
│   ├── scripts/             # 脚本工具
│   ├── configs/             # 配置系统
│   ├── utils/               # 工具函数
│   ├── viewer/              # 可视化
│   ├── exporter/            # 导出工具
│   ├── Encoder/             # 编码器（ResNet34等）
│   ├── transformer/         # Transformer相关
│   └── process_data/        # 数据预处理
├── preprocess/              # 数据预处理脚本
├── config/                  # 配置文件
├── checkpoints/             # 模型检查点
├── data/                    # 数据目录
├── outputs/                 # 输出目录
├── Zeroshot/                # 零样本推理输出
├── docs/                    # 文档
├── pyproject.toml           # Python项目配置
├── environment.yml          # Conda环境配置
└── README.md                # 项目说明
```

---

## 核心模块详解

### 1. 模型模块 (`nerfstudio/models/`)

#### 1.1 EvolSplatModel (`evolsplat.py`)
**核心模型实现，继承自 `Model` 基类**

**主要依赖：**
```python
# 外部库
- torch, torch.nn
- gsplat (Gaussian Splatting库)
- pytorch_msssim (SSIM损失)
- torchmetrics (PSNR, LPIPS)
- omegaconf (配置管理)
- einops (张量重排)
- numpy

# 内部模块
- nerfstudio.models.base_model (Model, ModelConfig)
- nerfstudio.cameras.cameras (Cameras)
- nerfstudio.data.scene_box (OrientedBox)
- nerfstudio.engine.callbacks (TrainingCallback)
- nerfstudio.engine.optimizers (Optimizers)
- nerfstudio.model_components.projection (Projector)
- nerfstudio.model_components.renderers (RGBRenderer)
- nerfstudio.model_components.sparse_conv (SparseCostRegNet, construct_sparse_tensor, sparse_to_dense_volume)
- nerfstudio.field_components.mlp (MLP)
- nerfstudio.field_components.embedding (Embedding)
- nerfstudio.fields.initial_BgSphere (GaussianBGInitializer)
- nerfstudio.utils (colormaps, rich_utils)
```

**关键组件：**
- `EvolSplatModelConfig`: 模型配置类
- `EvolSplatModel`: 主模型类
  - `sparse_conv`: 稀疏卷积网络（3D特征提取）
  - `mlp_offset`: MLP预测位置偏移
  - `mlp_conv`: MLP预测尺度和旋转
  - `mlp_opacity`: MLP预测不透明度
  - `projector`: 2D特征投影器
  - `renderer`: RGB渲染器

**主要方法：**
- `populate_modules()`: 初始化模型组件
- `get_outputs()`: 前向传播，生成渲染结果
- `get_loss_dict()`: 计算损失
- `get_image_metrics_and_images()`: 评估指标
- `init_volume()`: 初始化3D体积特征
- `output_evosplat()`: 输出Gaussian Splats

#### 1.2 其他模型
- `base_model.py`: 模型基类，定义 `Model` 和 `ModelConfig`
- `splatfacto.py`: 标准Gaussian Splatting模型（参考实现）
- `nerfacto.py`, `mipnerf.py`, `vanilla_nerf.py`: 其他NeRF变体

---

### 2. 流水线模块 (`nerfstudio/pipelines/`)

#### 2.1 BasePipeline (`base_pipeline.py`)
**训练和推理的抽象基类**

**主要依赖：**
```python
- torch, torch.distributed
- nerfstudio.data.datamanagers (DataManager)
- nerfstudio.models.base_model (Model)
- nerfstudio.engine.callbacks (TrainingCallback)
```

**关键组件：**
- `Pipeline`: 流水线基类
  - `model`: 模型实例
  - `datamanager`: 数据管理器
  - `get_train_loss_dict()`: 训练损失
  - `get_eval_loss_dict()`: 评估损失

#### 2.2 DynamicBatchPipeline (`dynamic_batch.py`)
**动态批次处理流水线**

---

### 3. 模型组件模块 (`nerfstudio/model_components/`)

#### 3.1 稀疏卷积 (`sparse_conv.py`)
**3D稀疏卷积网络实现**

**主要依赖：**
```python
- torch, torch.nn
- torchsparse (稀疏卷积库)
- open3d (点云处理)
- numpy
```

**关键组件：**
- `SparseCostRegNet`: 稀疏卷积网络
  - 输入：稀疏张量（SparseTensor）
  - 输出：3D特征体积
- `construct_sparse_tensor()`: 构建稀疏张量
- `sparse_to_dense_volume()`: 稀疏到密集转换

#### 3.2 投影器 (`projection.py`)
**2D特征投影器**

**主要依赖：**
```python
- torch
- nerfstudio.cameras.cameras (Cameras)
- nerfstudio.Encoder (ResNet34编码器)
```

**功能：**
- 从源图像中采样2D特征
- 投影到3D空间
- 多视角特征聚合

#### 3.3 渲染器 (`renderers.py`)
**渲染组件**

**主要依赖：**
```python
- torch
- gsplat.rendering (rasterization)
```

**关键组件：**
- `RGBRenderer`: RGB渲染器
  - 使用Gaussian Splatting进行可微渲染

#### 3.4 其他组件
- `ray_samplers.py`: 光线采样器
- `ray_generators.py`: 光线生成器
- `losses.py`: 损失函数
- `scene_colliders.py`: 场景碰撞器
- `shaders.py`: 着色器

---

### 4. 场组件模块 (`nerfstudio/field_components/`)

#### 4.1 MLP (`mlp.py`)
**多层感知机**

**主要依赖：**
```python
- torch, torch.nn
- nerfstudio.field_components.activations (激活函数)
```

#### 4.2 嵌入 (`embedding.py`)
**位置/特征嵌入**

#### 4.3 编码 (`encodings.py`)
**位置编码（如位置编码、哈希编码等）**

#### 4.4 其他组件
- `field_heads.py`: 场头（输出层）
- `spatial_distortions.py`: 空间扭曲
- `temporal_distortions.py`: 时间扭曲
- `base_field_component.py`: 基类

---

### 5. 场模块 (`nerfstudio/fields/`)

#### 5.1 InitialBgSphere (`initial_BgSphere.py`)
**背景球初始化**

**主要依赖：**
```python
- torch
- nerfstudio.field_components
```

#### 5.2 其他场
- `base_field.py`: 场基类
- `density_fields.py`: 密度场
- `nerfacto_field.py`: Nerfacto场
- `sdf_field.py`: SDF场

---

### 6. 数据模块 (`nerfstudio/data/`)

#### 6.1 数据解析器 (`dataparsers/`)

**ZeroshotDataParser (`zeroshot_dataparser.py`)**
**零样本数据解析器**

**主要依赖：**
```python
- torch, numpy
- PIL (Image)
- nerfstudio.cameras (Cameras, camera_utils)
- nerfstudio.data.dataparsers.base_dataparser (DataParser)
- nerfstudio.data.scene_box (SceneBox)
- nerfstudio.utils.io (load_from_json)
- omegaconf
```

**功能：**
- 解析transforms.json
- 加载图像、深度图、点云
- 构建相机参数
- 创建数据集

#### 6.2 数据管理器 (`datamanagers/`)
- `base_datamanager.py`: 数据管理器基类
- `full_images_datamanager.py`: 全图像数据管理器
- `parallel_datamanager.py`: 并行数据管理器

#### 6.3 数据集 (`datasets/`)
- 数据集类定义

#### 6.4 其他
- `scene_box.py`: 场景边界框
- `pixel_samplers.py`: 像素采样器
- `utils/`: 数据工具函数

---

### 7. 相机模块 (`nerfstudio/cameras/`)

**主要依赖：**
```python
- torch
- numpy
- pyquaternion (四元数)
```

**关键组件：**
- `cameras.py`: 相机类（Cameras）
- `camera_utils.py`: 相机工具函数
- `camera_paths.py`: 相机路径生成
- `camera_optimizers.py`: 相机优化器
- `rays.py`: 光线定义（RayBundle）
- `lie_groups.py`: 李群相关

---

### 8. 引擎模块 (`nerfstudio/engine/`)

#### 8.1 Trainer (`trainer.py`)
**训练器**

**主要依赖：**
```python
- torch, torch.distributed
- nerfstudio.pipelines.base_pipeline (Pipeline)
- nerfstudio.data.datamanagers (DataManager)
- nerfstudio.engine.optimizers (Optimizers)
- nerfstudio.engine.schedulers (Schedulers)
- nerfstudio.engine.callbacks (TrainingCallback)
- nerfstudio.utils (profiler, comms)
```

**关键功能：**
- 训练循环
- 验证循环
- 检查点保存/加载
- 日志记录

#### 8.2 其他组件
- `optimizers.py`: 优化器配置
- `schedulers.py`: 学习率调度器
- `callbacks.py`: 训练回调

---

### 9. 脚本模块 (`nerfstudio/scripts/`)

#### 9.1 推理脚本 (`infer_zeroshot.py`)
**零样本推理脚本**

**主要依赖：**
```python
# 标准库
- argparse, pathlib, os, sys
- random, socket, traceback
- datetime, typing, functools, dataclasses, collections

# 第三方库
- numpy, torch, torch.distributed, torch.multiprocessing
- tyro (命令行解析)
- yaml (配置加载)
- tqdm (进度条)
- torchmetrics (评估指标)
- mediapy (媒体处理)
- wandb (实验跟踪)
- moviepy (视频处理)

# 内部模块
- nerfstudio.configs (config_utils, method_configs)
- nerfstudio.engine.trainer (TrainerConfig)
- nerfstudio.utils (comms, profiler, rich_utils, colormaps)
- nerfstudio.cameras.camera_paths (get_interpolated_camera_path)
- nerfstudio.models.evolsplat (EvolSplatModel, RGB2SH)
- nerfstudio.models.splatfacto (SplatfactoModel)
- nerfstudio.scripts.viewer.run_viewer (_start_viewer)
- gsplat (Gaussian Splatting库)
```

**关键函数：**
- `infer_loop()`: 推理主循环
- `launch()`: 多GPU启动
- `main()`: 主函数
- `entrypoint()`: 入口点

#### 9.2 训练脚本 (`train.py`)
**训练脚本**

**主要依赖：**
```python
- tyro
- nerfstudio.configs
- nerfstudio.engine.trainer
- nerfstudio.utils
```

#### 9.3 导出脚本 (`exporter.py`)
**模型导出脚本**

**主要依赖：**
```python
- nerfstudio.models
- nerfstudio.exporter
- nerfstudio.utils
```

#### 9.4 其他脚本
- `eval.py`: 评估脚本
- `render.py`: 渲染脚本
- `process_data.py`: 数据处理脚本
- `viewer/`: 可视化相关脚本

---

### 10. 编码器模块 (`nerfstudio/Encoder/`)

#### 10.1 ResNet34 (`ResNet34/`)
**ResNet34编码器**

**文件：**
- `ResNet34.py`: ResNet34网络定义
- `custom_ecnoder.py`: 自定义编码器
- `util.py`: 工具函数

**主要依赖：**
```python
- torch, torch.nn
- torchvision
```

---

### 11. Transformer模块 (`nerfstudio/transformer/`)

**主要组件：**
- `multiview_tranformer.py`: 多视角Transformer
- `backbone_cnn.py`: CNN骨干网络
- `backbone_multiview.py`: 多视角骨干网络
- `position.py`: 位置编码
- `trident_conv.py`: 三叉戟卷积
- `utils.py`: 工具函数

---

### 12. 工具模块 (`nerfstudio/utils/`)

**主要工具：**
- `rich_utils.py`: Rich控制台输出
- `colormaps.py`: 颜色映射
- `io.py`: IO操作
- `math.py`: 数学工具
- `poses.py`: 位姿处理
- `profiler.py`: 性能分析
- `comms.py`: 通信工具（分布式）
- `eval_utils.py`: 评估工具
- `plotly_utils.py`: Plotly可视化
- `misc.py`: 杂项工具

---

### 13. 配置模块 (`nerfstudio/configs/`)

**主要组件：**
- `base_config.py`: 配置基类
- `method_configs.py`: 方法配置
- `config_utils.py`: 配置工具

---

### 14. 数据预处理 (`preprocess/`)

**主要组件：**
- `run.py`: 预处理主脚本
- `read_dataset/`: 数据集读取
- `metric3d/`: Metric3D深度估计
- `nvi_sem/`: 语义分割
- `requirements.txt`: 依赖列表

---

## 组件依赖关系图

### 核心依赖层次

```
┌─────────────────────────────────────────┐
│         Scripts (入口点)                 │
│  - infer_zeroshot.py                    │
│  - train.py                             │
│  - exporter.py                           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Engine (训练引擎)                │
│  - trainer.py                            │
│    ├── Pipeline                          │
│    ├── DataManager                       │
│    └── Optimizers                        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Pipeline (流水线)                │
│  - base_pipeline.py                      │
│    ├── Model (evolsplat.py)              │
│    └── DataManager                       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Model (EvolSplatModel)           │
│  - evolsplat.py                          │
│    ├── SparseCostRegNet (sparse_conv)    │
│    ├── Projector (projection)            │
│    ├── MLP (field_components/mlp)        │
│    ├── RGBRenderer (renderers)           │
│    └── GaussianBGInitializer             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Model Components (模型组件)          │
│  - sparse_conv.py (3D特征提取)            │
│  - projection.py (2D特征投影)             │
│  - renderers.py (渲染)                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      External Libraries (外部库)          │
│  - gsplat (Gaussian Splatting)           │
│  - torchsparse (稀疏卷积)                 │
│  - torch (深度学习框架)                    │
└─────────────────────────────────────────┘
```

### 详细依赖关系

#### EvolSplatModel 依赖树

```
EvolSplatModel (evolsplat.py)
│
├── Base Classes
│   ├── Model (base_model.py)
│   └── ModelConfig (base_model.py)
│
├── 3D Feature Extraction
│   ├── SparseCostRegNet (model_components/sparse_conv.py)
│   │   └── torchsparse (外部库)
│   ├── construct_sparse_tensor (sparse_conv.py)
│   └── sparse_to_dense_volume (sparse_conv.py)
│
├── 2D Feature Projection
│   ├── Projector (model_components/projection.py)
│   │   └── ResNet34 (Encoder/ResNet34/)
│   └── Cameras (cameras/cameras.py)
│
├── Feature Decoding
│   ├── MLP (field_components/mlp.py)
│   ├── Embedding (field_components/embedding.py)
│   └── Field Heads (field_components/field_heads.py)
│
├── Rendering
│   ├── RGBRenderer (model_components/renderers.py)
│   │   └── gsplat.rendering (外部库)
│   └── GaussianBGInitializer (fields/initial_BgSphere.py)
│
├── Data Handling
│   ├── Cameras (cameras/cameras.py)
│   ├── SceneBox (data/scene_box.py)
│   └── RayBundle (cameras/rays.py)
│
└── Training Support
    ├── Optimizers (engine/optimizers.py)
    ├── Callbacks (engine/callbacks.py)
    └── Loss Functions (model_components/losses.py)
```

#### 数据流依赖

```
DataParser (zeroshot_dataparser.py)
│
├── Load Data
│   ├── Images (PIL)
│   ├── Depth Maps
│   ├── Point Clouds (ply files)
│   └── Camera Parameters (transforms.json)
│
└── Create Dataset
    ├── SceneBox (data/scene_box.py)
    ├── Cameras (cameras/cameras.py)
    └── DataparserOutputs
         │
         ▼
    DataManager
         │
         ▼
    Pipeline.get_train_loss_dict()
         │
         ▼
    Model.get_outputs()
```

---

## 数据流和调用链

### 训练流程调用链

```
1. ns-train (scripts/train.py)
   │
   ├── 2. TrainerConfig.setup() (engine/trainer.py)
   │      │
   │      ├── 3. Pipeline.setup() (pipelines/base_pipeline.py)
   │      │      │
   │      │      ├── 4. Model.__init__() (models/evolsplat.py)
   │      │      │      │
   │      │      │      └── 5. populate_modules()
   │      │      │             ├── SparseCostRegNet
   │      │      │             ├── Projector
   │      │      │             ├── MLPs (offset, conv, opacity)
   │      │      │             └── RGBRenderer
   │      │      │
   │      │      └── 6. DataManager.setup() (data/datamanagers/)
   │      │             │
   │      │             └── 7. DataParser.setup() (data/dataparsers/)
   │      │
   │      └── 8. Optimizers.setup() (engine/optimizers.py)
   │
   └── 9. Trainer.train() (训练循环)
          │
          ├── 10. DataManager.next_train() (获取批次)
          │
          ├── 11. Pipeline.get_train_loss_dict()
          │        │
          │        └── 12. Model.get_outputs()
          │               │
          │               ├── 13. 3D特征提取 (sparse_conv)
          │               ├── 14. 2D特征投影 (projector)
          │               ├── 15. 特征融合 (MLP)
          │               ├── 16. Gaussian参数预测 (MLP)
          │               └── 17. 渲染 (RGBRenderer)
          │
          ├── 18. Model.get_loss_dict() (计算损失)
          │
          └── 19. Optimizer.step() (反向传播)
```

### 推理流程调用链

```
1. infer_zeroshot.py (scripts/infer_zeroshot.py)
   │
   ├── 2. main() → launch() → infer_loop()
   │      │
   │      ├── 3. TrainerConfig.setup() (加载模型)
   │      │      │
   │      │      └── 4. Pipeline.setup()
   │      │             │
   │      │             └── 5. Model.__init__() (从checkpoint加载)
   │      │
   │      ├── 6. trainer.setup_feedforward()
   │      │
   │      ├── 7. model.init_volume() (初始化3D体积)
   │      │      │
   │      │      └── 8. construct_sparse_tensor()
   │      │      └── 9. sparse_conv() (3D特征提取)
   │      │      └── 10. sparse_to_dense_volume()
   │      │
   │      └── 11. 推理循环 (for each eval image)
   │             │
   │             ├── 12. DataManager.next_eval_image()
   │             │
   │             ├── 13. Pipeline.get_eval_loss_dict()
   │             │        │
   │             │        └── 14. Model.get_outputs()
   │             │               │
   │             │               ├── 15. 特征提取和融合
   │             │               └── 16. 渲染
   │             │
   │             └── 17. 计算指标 (PSNR, SSIM, LPIPS)
   │
   └── 18. 保存结果
```

---

## 外部依赖库

### 核心依赖（必需）

#### 深度学习框架
- **torch** (>=1.13.1): PyTorch深度学习框架
- **torchvision** (>=0.14.1): 计算机视觉工具
- **torchmetrics** (>=1.0.1): 评估指标

#### Gaussian Splatting
- **gsplat** (>=1.0.0): Gaussian Splatting渲染库
  - 提供可微分的Gaussian Splatting渲染
  - CUDA加速实现

#### 稀疏卷积
- **torchsparse** (>=2.1.0): 稀疏卷积库
  - 用于3D稀疏特征提取
  - 依赖: sparsehash, libsparsehash-dev

#### 图像处理
- **opencv-python** (==4.8.0.76): OpenCV
- **Pillow** (>=10.3.0): PIL图像处理
- **imageio** (>=2.21.1): 图像IO
- **scikit-image** (>=0.19.3): 图像处理

#### 3D处理
- **open3d** (>=0.16.0): 3D数据处理
- **trimesh** (>=3.20.2): 网格处理
- **pymeshlab** (>=2022.2.post2): 网格处理

#### 数值计算
- **numpy**: 数值计算
- **einops**: 张量操作

#### 配置管理
- **omegaconf** (==2.3.0): 配置管理
- **easydict**: 字典工具
- **tyro** (>=0.6.6): 命令行解析

#### 评估指标
- **pytorch-msssim**: SSIM损失
- **torchmetrics[image]**: 图像评估指标

### 可选依赖

#### 可视化
- **plotly** (>=5.7.0): 交互式可视化
- **matplotlib** (>=3.6.0): 绘图
- **mediapy** (>=1.1.0): 媒体处理
- **moviepy** (==1.0.3): 视频处理

#### 实验跟踪
- **wandb** (>=0.13.3): Weights & Biases
- **tensorboard** (>=2.13.0): TensorBoard
- **comet_ml** (>=3.33.8): Comet ML

#### 数据格式
- **h5py** (>=2.9.0): HDF5
- **msgpack** (>=1.0.4): MessagePack
- **msgpack_numpy** (>=0.4.8): NumPy序列化

#### 其他工具
- **tqdm**: 进度条
- **rich** (>=12.5.1): 终端美化
- **viser** (==0.1.27): 3D可视化
- **nerfacc** (==0.5.2): NeRF加速
- **timm** (==0.6.7): 预训练模型

### 开发依赖

- **pytest** (==7.1.2): 测试框架
- **ruff** (==0.1.13): 代码检查
- **pyright** (==1.1.331): 类型检查
- **pre-commit** (==3.3.2): Git钩子

---

## 配置文件结构

### 项目配置

#### `pyproject.toml`
- Python项目元数据
- 依赖列表
- 入口点定义（ns-train, ns-viewer等）
- 工具配置（pytest, pyright, ruff）

#### `environment.yml`
- Conda环境配置
- Python版本
- 系统依赖

### 模型配置

#### `config/Neuralsplat.yaml`
**手动配置文件，包含：**
- 模型超参数
  - `sparseConv_outdim`: 稀疏卷积输出维度
  - `local_radius`: 局部半径
  - `offset_max`: 最大偏移
  - `num_neighbour_select`: 邻居选择数量
- 边界框参数
  - `Boundingbox_min`: 最小边界
  - `Boundingbox_max`: 最大边界

### 数据配置

#### `transforms.json` (每个场景)
**包含：**
- 相机内参（intrinsics）
- 相机外参（extrinsics）
- 图像路径
- 深度图路径
- 点云路径

---

## 关键文件索引

### 核心模型文件
- `nerfstudio/models/evolsplat.py`: EvolSplat主模型（832行）
- `nerfstudio/models/base_model.py`: 模型基类
- `nerfstudio/model_components/sparse_conv.py`: 稀疏卷积实现
- `nerfstudio/model_components/projection.py`: 投影器实现
- `nerfstudio/model_components/renderers.py`: 渲染器实现

### 数据处理文件
- `nerfstudio/data/dataparsers/zeroshot_dataparser.py`: 零样本数据解析器（354行）
- `nerfstudio/data/datamanagers/`: 数据管理器
- `nerfstudio/data/datasets/`: 数据集定义

### 训练和推理脚本
- `nerfstudio/scripts/infer_zeroshot.py`: 零样本推理脚本（594行）
- `nerfstudio/scripts/train.py`: 训练脚本
- `nerfstudio/scripts/exporter.py`: 导出脚本（751行）
- `nerfstudio/engine/trainer.py`: 训练器（606行）

### 工具和配置
- `nerfstudio/utils/`: 工具函数集合
- `nerfstudio/configs/`: 配置系统
- `nerfstudio/cameras/`: 相机相关
- `preprocess/run.py`: 数据预处理脚本

---

## 模块间通信模式

### 1. 配置传递
```
Config (YAML/CLI)
    ↓
TrainerConfig
    ↓
PipelineConfig
    ↓
ModelConfig (EvolSplatModelConfig)
    ↓
Model.__init__()
```

### 2. 数据传递
```
DataParser
    ↓ (DataparserOutputs)
DataManager
    ↓ (Dict[str, Tensor])
Pipeline.get_train_loss_dict()
    ↓ (Dict[str, Tensor])
Model.get_outputs()
    ↓ (Dict[str, Tensor])
Renderer / Loss Functions
```

### 3. 梯度流
```
Loss
    ↓ (backward)
Model Parameters
    ├── SparseConv weights
    ├── Projector weights
    ├── MLP weights
    └── Gaussian parameters (means, scales, etc.)
```

---

## 总结

EVolSplat项目基于nerfstudio框架构建，采用模块化设计：

1. **模型层**: EvolSplatModel整合3D特征提取、2D特征投影、特征融合和渲染
2. **组件层**: 可复用的组件（稀疏卷积、投影器、渲染器等）
3. **数据层**: 数据解析、管理和加载
4. **引擎层**: 训练和推理流程管理
5. **工具层**: 脚本、可视化和导出工具

主要依赖关系：
- **gsplat**: Gaussian Splatting渲染核心
- **torchsparse**: 3D稀疏特征提取
- **torch**: 深度学习基础框架
- **nerfstudio**: 框架基础设施

项目支持多场景训练和零样本推理，通过稀疏卷积提取3D特征，结合2D图像特征，生成高质量的3D Gaussian Splats用于实时渲染。

