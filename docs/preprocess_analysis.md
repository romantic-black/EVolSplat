# Preprocess 模块算法流程与数据流分析

## 目录
1. [算法流程概览](#算法流程概览)
2. [数据流与关键维度](#数据流与关键维度)
3. [关键组件详解](#关键组件详解)
4. [反直觉检查](#反直觉检查)
5. [错误检验与修复建议](#错误检验与修复建议)

---

## 算法流程概览

### 整体流程图

```
原始数据集 (KITTI-360/Waymo)
    ↓
[1] 数据读取模块 (ReadKITTI360Data/ReadWaymoData)
    ├─ 读取图像: (H, W, 3) RGB
    ├─ 读取相机内参: 3×3 或 4×4 矩阵
    ├─ 读取相机位姿: 4×4 变换矩阵
    ├─ 位姿归一化: 以中间帧为原点
    └─ 生成 transforms.json
    ↓
[2] 深度估计模块 (Metric3D) [可选]
    ├─ 输入: RGB图像 (H, W, 3)
    └─ 输出: 深度图 (H, W) - .npy格式
    ↓
[3] 语义分割模块 (nvi_sem) [可选]
    ├─ 输入: RGB图像 (H, W, 3)
    └─ 输出: 语义实例图 (H, W) - instance/*.png
    ↓
[4] 天空掩码生成 (gen_sky_mask)
    ├─ 输入: 语义实例图 (instance==10 表示天空)
    └─ 输出: 二值掩码 (H, W) - mask/*.png
    ↓
[5] 点云生成模块 (PCDGenerator/WaymoPCDGenerator)
    ├─ 帧选择: 根据sparsity参数选择帧
    ├─ 深度一致性检查: [可选] 跨帧深度验证
    ├─ 反投影: 深度图 → 3D点云
    ├─ 坐标变换: 相机坐标系 → 世界坐标系
    ├─ 掩码应用: 天空过滤 + 深度一致性掩码
    └─ 点云累积: 多帧点云合并
    ↓
[6] 点云后处理
    ├─ 边界框裁剪: 保留指定范围内的点
    ├─ 统计离群点移除: Open3D滤波
    ├─ 均匀下采样: 降低点云密度
    └─ 坐标归一化: 转换到归一化相机坐标系
    ↓
输出: .ply点云文件 + transforms.json
```

### 详细步骤说明

#### 步骤1: 数据读取与位姿归一化

**KITTI-360流程:**
- 读取双相机图像 (image_00, image_01)
- 读取相机内参矩阵 K_00, K_01
- 读取相机到世界坐标变换矩阵
- **位姿归一化**: 以中间帧为原点，所有位姿相对化
- 坐标系统转换: OpenCV → OpenGL (旋转矩阵乘以 [1, -1, -1])

**Waymo流程:**
- 读取单相机图像 (camera 1)
- 图像resize到 (640, 960)
- 读取车辆到世界坐标变换 (v2w)
- 读取相机到车辆变换 (c2v)
- 组合变换: c2w = v2w @ c2v @ opengl2waymo
- **位姿归一化**: 以中间帧为原点

#### 步骤2: 深度估计 (Metric3D)

- 使用Metric3D Giant模型进行单目深度估计
- 输出为metric depth (真实尺度深度)
- 保存为.npy格式，维度 (H, W)

#### 步骤3: 语义分割 (nvi_sem)

- 使用OCRNet + HRNet进行语义分割
- 输出实例分割图，其中instance==10表示天空类别
- 保存到 semantic/instance/ 目录

#### 步骤4: 天空掩码生成

- 从语义实例图中提取天空区域 (instance==10)
- 生成二值掩码: 天空=0, 非天空=255
- 可选: 对掩码进行腐蚀操作以减少边界噪声

#### 步骤5: 点云生成

**帧选择策略 (Sparsity):**
- `full`: 使用所有帧
- `Drop25`: 每4帧丢弃第3帧 (保留75%)
- `Drop50`: 每4帧保留前2帧 (保留50%)
- `Drop80`: 每10帧保留前2帧 (保留20%)
- `Drop90`: 每10帧保留第1帧 (保留10%)

**深度一致性检查 (可选):**
- 将当前帧的3D点投影到上一帧
- 比较投影位置的深度值与上一帧深度值
- 如果深度差异小于平均值，则认为一致

**反投影公式:**
```
x_cam = (u - cx) * depth / fx
y_cam = (v - cy) * depth / fy
z_cam = depth
P_cam = [x_cam, y_cam, z_cam, 1]^T
P_world = c2w @ P_cam
```

**掩码组合:**
- 深度一致性掩码 (如果启用)
- 天空掩码 (如果启用filter_sky)
- 下采样掩码 (如果down_scale > 1)
- 最终掩码 = 所有掩码的逻辑与

#### 步骤6: 点云后处理

**边界框裁剪:**
- KITTI-360: X∈[-16,16], Y∈[-9,3.8], Z∈[-30,30]
- Waymo: X∈[-20,20], Y∈[-20,4.8], Z∈[-20,70]

**统计离群点移除:**
- KITTI-360: nb_neighbors=20, std_ratio=2.0
- Waymo (inside): nb_neighbors=35, std_ratio=1.5
- Waymo (outside): nb_neighbors=20, std_ratio=2.0

**均匀下采样:**
- KITTI-360: every_k_points=5
- Waymo: every_k_points=2

**坐标归一化:**
- 使用transforms.json中的inv_pose将点云转换到归一化相机坐标系
- 这是为了与NeRF训练时的坐标系一致

---

## 数据流与关键维度

### 输入数据维度

| 数据类型 | 维度 | 格式 | 说明 |
|---------|------|------|------|
| RGB图像 (KITTI-360) | (376, 1408, 3) | uint8 | 原始图像 |
| RGB图像 (Waymo) | (640, 960, 3) | uint8 | Resize后的图像 |
| 相机内参 K | (3, 3) 或 (4, 4) | float32 | 焦距和主点坐标 |
| 相机位姿 c2w | (4, 4) | float32 | 齐次变换矩阵 |
| 深度图 | (H, W) | float32 | Metric depth, .npy格式 |
| 语义实例图 | (H, W) | uint16 | 实例ID，instance==10为天空 |

### 中间数据维度

| 数据类型 | 维度 | 说明 |
|---------|------|------|
| 归一化位姿 | (N, 4, 4) | N为帧数，以中间帧为原点 |
| 深度一致性掩码 | (H, W) | bool数组，True表示一致 |
| 天空掩码 | (H, W) | uint8，0=天空，255=非天空 |
| 最终掩码 | (H, W) | bool数组，True表示保留该像素 |
| 相机坐标系点云 | (M, 3) | M为有效像素数 |
| 世界坐标系点云 | (M, 3) | 累积后的点云 |
| 带颜色的点云 | (M, 6) | [x, y, z, r, g, b] |

### 输出数据维度

| 数据类型 | 维度 | 格式 | 说明 |
|---------|------|------|------|
| 点云文件 | N×6 | .ply | N为最终点数，每点包含xyz+RGB |
| transforms.json | - | JSON | 包含相机参数、位姿、点云路径 |

### 数据流转换链

```
原始图像 (H, W, 3) uint8
    ↓ [读取]
RGB数组 (H, W, 3) float32 [0,1]
    ↓ [深度估计]
深度图 (H, W) float32
    ↓ [语义分割]
语义图 (H, W) uint16
    ↓ [掩码生成]
天空掩码 (H, W) bool
    ↓ [掩码组合]
最终掩码 (H, W) bool
    ↓ [像素选择]
有效像素索引 (M, 2) int32
    ↓ [反投影]
相机坐标点 (M, 3) float32
    ↓ [坐标变换]
世界坐标点 (M, 3) float32
    ↓ [多帧累积]
累积点云 (N, 3) float32, N >> M
    ↓ [边界框裁剪]
裁剪点云 (K, 3) float32, K ≤ N
    ↓ [离群点移除]
滤波点云 (L, 3) float32, L ≤ K
    ↓ [下采样]
最终点云 (P, 3) float32, P ≤ L
    ↓ [坐标归一化]
归一化点云 (P, 3) float32
    ↓ [保存]
.ply文件
```

### 关键数据维度变化示例

假设处理40帧KITTI-360图像 (H=376, W=1408):

1. **原始输入**: 40帧 × 2相机 = 80张图像
2. **深度图**: 80张 (H, W) 深度图
3. **帧选择 (Drop50)**: 保留40张 (每4帧保留2帧)
4. **有效像素**: 假设每张图保留50%像素 → 约 376×1408×0.5 = 265,024 点/帧
5. **累积点云**: 40帧 × 265,024 ≈ 10.6M 点
6. **边界框裁剪**: 假设保留30% → 约 3.2M 点
7. **离群点移除**: 假设保留80% → 约 2.6M 点
8. **下采样 (every_k=5)**: 最终约 520K 点

---

## 关键组件详解

### 1. System类 (`run.py`)

**职责**: 主流程控制器，协调各个模块

**关键方法:**
- `__init__`: 初始化数据读取器和点云生成器
- `forward` (KITTI-360): 执行完整预处理流程
- `Waymo_forward` (Waymo): Waymo专用流程
- `gen_metric_depth`: 调用Metric3D进行深度估计
- `gen_semantic`: 调用nvi_sem进行语义分割
- `gen_sky_mask`: 从语义图中提取天空掩码

**关键参数:**
- `pcd_sparsity`: 点云稀疏度 ('Drop90', 'Drop50', 'Drop80', 'Drop25', 'full')
- `depth_cosistency`: 是否启用深度一致性检查
- `use_semantics`: 是否生成语义图
- `use_metric`: 是否生成metric depth
- `filter_pcd_sky`: 是否过滤天空区域

### 2. ReadKITTI360Data类 (`read_kitti360.py`)

**职责**: 读取KITTI-360数据集

**关键方法:**
- `generate_json`: 生成transforms.json和保存图像
- `read_data`: 读取图像、内参、位姿
- `pose_normalization`: 位姿归一化（以中间帧为原点）
- `loadCameraToPose`: 加载相机到车辆位姿变换

**数据流:**
```
KITTI-360原始数据
    ↓
读取图像 (image_00, image_01)
    ↓
读取内参 (K_00, K_01)
    ↓
读取位姿 (cam2world_00, cam2world_01)
    ↓
位姿归一化
    ↓
生成transforms.json
```

### 3. ReadWaymoData类 (`read_waymo.py`)

**职责**: 读取Waymo数据集

**关键方法:**
- `generate_json`: 生成transforms.json和保存图像
- `read_data`: 从TFRecord读取数据
- `pose_normalization`: 位姿归一化

**特殊处理:**
- 图像resize: 原始 → (640, 960)
- 内参缩放: 内参值除以2（因为图像resize了）
- 坐标变换: opengl2waymo矩阵转换

### 4. PCDGenerator类 (`generate_pcd.py`)

**职责**: KITTI-360点云生成

**关键方法:**
- `forward`: 主流程
- `extract_c2w`: 从文件名提取对应位姿
- `accumulat_pcd`: 累积多帧点云
- `depth_cosistency_check`: 深度一致性检查
- `depth_projection_check`: 深度投影验证
- `crop_pointcloud`: 边界框裁剪

**点云生成流程:**
```
深度图 + RGB + 位姿
    ↓
掩码应用 (深度一致性 + 天空 + 下采样)
    ↓
像素选择
    ↓
反投影到相机坐标系
    ↓
变换到世界坐标系
    ↓
多帧累积
    ↓
边界框裁剪
    ↓
离群点移除
    ↓
下采样
    ↓
坐标归一化
    ↓
保存.ply
```

### 5. WaymoPCDGenerator类 (`generate_waymo_pcd.py`)

**职责**: Waymo点云生成

**与PCDGenerator的区别:**
- 使用单相机（而非双相机）
- 边界框范围不同
- 离群点移除参数不同（inside/outside分别处理）
- 默认不启用深度一致性检查

### 6. 深度一致性检查算法

**原理:**
1. 将当前帧的3D点投影到上一帧的相机坐标系
2. 将投影点重投影到上一帧的图像平面
3. 比较重投影位置的深度值与上一帧的深度值
4. 如果深度差异小于平均值，则认为一致

**数学公式:**
```
# 当前帧点云 → 上一帧坐标系
T = inv(last_c2w) @ c2w
P_last = T @ P_current

# 重投影到上一帧图像
u_last = (fx * P_last.x + cx * P_last.z) / P_last.z
v_last = (fy * P_last.y + cy * P_last.z) / P_last.z

# 深度一致性判断
depth_diff = |depth_current[u,v] - depth_last[u_last, v_last]|
consistent = depth_diff < mean(depth_diff)
```

---

## 反直觉检查

### 1. 位姿归一化使用中间帧而非第一帧

**直觉**: 通常以第一帧为原点

**实际情况**: 代码使用中间帧 (`mid_frames = poses.shape[0] // 2`)

**原因分析**:
- 中间帧作为原点可以保证前后帧的位姿分布更均匀
- 减少数值精度问题（避免某些帧距离原点过远）
- 符合NeRF训练时的常见做法

**验证**: ✅ 合理

### 2. 深度一致性检查的阈值使用平均值

**直觉**: 应该使用固定阈值或标准差倍数

**实际情况**: 使用 `depth_diff < depth_diff.mean()`

**潜在问题**:
- 如果大部分点都不一致，平均值会很大，导致很多不一致的点被误判为一致
- 应该使用更稳定的统计量（如中位数）或固定阈值

**建议**: ⚠️ 需要改进

### 3. Waymo图像resize后内参除以2

**直觉**: 如果图像resize，内参应该按比例缩放

**实际情况**: 代码中内参值除以2（`/ 2`）

**验证**: 
- Waymo原始图像可能是 (1280, 1920)
- Resize到 (640, 960) 确实是除以2
- ✅ 正确

### 4. 点云先累积再裁剪，而非先裁剪再累积

**直觉**: 先裁剪可以减少计算量

**实际情况**: 先在世界坐标系累积所有点，再裁剪

**原因分析**:
- 需要先变换到世界坐标系才能正确累积
- 边界框是在归一化相机坐标系定义的，需要先归一化再裁剪
- 但代码中是在世界坐标系裁剪，然后才归一化

**潜在问题**: ⚠️ 边界框定义和裁剪时机可能不一致

### 5. KITTI-360使用双相机但Waymo使用单相机

**直觉**: 应该统一处理方式

**实际情况**: 
- KITTI-360: image_00 和 image_01 都处理
- Waymo: 只处理camera 1

**原因**: 数据集特性不同，KITTI-360提供双相机数据

**验证**: ✅ 合理

### 6. 深度一致性检查中像素坐标交换

**代码片段**:
```python
pixels[:, [0, 1]] = pixels[:, [1, 0]]
last_pixels[:, [0, 1]] = last_pixels[:, [1, 0]]
```

**直觉**: 为什么要交换坐标？

**分析**: 
- 这可能是为了匹配numpy数组的索引习惯（行在前，列在后）
- 但这样容易出错，应该使用更清晰的变量命名

**建议**: ⚠️ 代码可读性需要改进

### 7. 天空掩码的腐蚀操作

**KITTI-360**: 使用 (1, 1) 的kernel，几乎无效果
**Waymo**: 使用 (30, 30) 的kernel，效果明显

**直觉**: 应该统一或根据实际需求调整

**实际情况**: 两个数据集使用不同的kernel大小

**建议**: ⚠️ 应该统一或添加参数配置

---

## 错误检验与修复建议

### 错误1: Drop80逻辑错误 (generate_pcd.py:71)

**位置**: `preprocess/read_dataset/generate_pcd.py:71`

**错误代码**:
```python
elif self.sparsity == 'Drop80':
    if i % 10 != 0 or i % 10 != 1:
        continue
```

**问题**: 
- `i % 10 != 0 or i % 10 != 1` 永远为True
- 因为一个数不可能同时不等于0和1（例如，如果i%10==0，则i%10!=1为True）

**正确逻辑**: 应该保留 i%10==0 或 i%10==1 的帧

**修复建议**:
```python
elif self.sparsity == 'Drop80':
    if i % 10 not in [0, 1]:  # 或者: if i % 10 != 0 and i % 10 != 1:
        continue
```

**严重程度**: 🔴 严重 - 导致Drop80模式无法正常工作

### 错误2: Waymo Drop80逻辑不一致 (generate_waymo_pcd.py:68)

**位置**: `preprocess/read_dataset/generate_waymo_pcd.py:68`

**错误代码**:
```python
elif self.sparsity == 'Drop80':
    if i % 5 != 0:
        continue
```

**问题**: 
- 使用 `i % 5 != 0` 意味着每5帧保留1帧（保留20%）
- 但Drop80的定义应该是每10帧保留2帧（保留20%）
- 虽然保留比例相同，但帧选择模式不同

**修复建议**: 应该与KITTI-360保持一致
```python
elif self.sparsity == 'Drop80':
    if i % 10 not in [0, 1]:
        continue
```

**严重程度**: 🟡 中等 - 功能可用但行为不一致

### 错误3: Waymo深度反投影使用错误的内参 (generate_waymo_pcd.py:223)

**位置**: `preprocess/read_dataset/generate_waymo_pcd.py:223`

**错误代码**:
```python
y = (pixels[..., 1] - K[1,2]) * depth.reshape(-1) / K[0,0]  # 错误: 应该用K[1,1]
```

**问题**: 
- y坐标计算应该使用 `K[1,1]` (fy)，但代码使用了 `K[0,0]` (fx)
- 这会导致y坐标计算错误

**修复建议**:
```python
y = (pixels[..., 1] - K[1,2]) * depth.reshape(-1) / K[1,1]
```

**严重程度**: 🔴 严重 - 导致点云y坐标错误

### 错误4: KITTI-360内参fl_y错误 (read_kitti360.py:161)

**位置**: `preprocess/read_dataset/read_kitti360.py:161`

**错误代码**:
```python
'fl_y': K_00[0][0].item(),  # 错误: 应该用K_00[1][1]
```

**问题**: 
- fl_y应该使用K_00[1][1]（y方向焦距），但代码使用了K_00[0][0]（x方向焦距）
- 这会导致transforms.json中的fl_y值错误

**修复建议**:
```python
'fl_y': K_00[1][1].item(),
```

**严重程度**: 🟡 中等 - 影响transforms.json的正确性，但可能不影响训练（如果模型不使用fl_y）

### 错误5: 硬编码图像尺寸 (generate_pcd.py:186)

**位置**: `preprocess/read_dataset/generate_pcd.py:186`

**错误代码**:
```python
pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(376, 1408, 2)
```

**问题**: 
- 硬编码了KITTI-360的图像尺寸 (376, 1408)
- 应该使用 `self.H, self.W` 以支持不同尺寸

**修复建议**:
```python
pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(self.H, self.W, 2)
```

**严重程度**: 🟡 中等 - 如果处理不同尺寸的图像会出错

### 错误6: 变量名拼写错误 (多处)

**问题**: 
- `depth_cosistency` 应该是 `depth_consistency` (consistency拼写错误)
- `cosistency_mask` 应该是 `consistency_mask`
- `filer_sky` 应该是 `filter_sky` (filter拼写错误)

**严重程度**: 🟢 轻微 - 不影响功能，但影响代码可读性

### 错误7: 深度一致性检查中的坐标索引可能越界

**位置**: `generate_pcd.py:262`, `generate_waymo_pcd.py:257`

**潜在问题**:
```python
depth_diff = np.abs(depth[pixels[valid_mask, 0], pixels[valid_mask, 1]] - 
                    last_depth[last_pixels[valid_mask, 0], last_pixels[valid_mask, 1]])
```

**问题**: 
- `last_pixels` 是int32类型，但可能包含负值或超出图像范围的值
- 虽然前面有valid_mask检查，但索引时仍可能有问题

**建议**: 添加边界检查或使用np.clip限制索引范围

**严重程度**: 🟡 中等 - 可能导致索引错误

### 错误总结表

| 错误编号 | 位置 | 严重程度 | 类型 | 状态 |
|---------|------|---------|------|------|
| 1 | generate_pcd.py:71 | 🔴 严重 | 逻辑错误 | 需修复 |
| 2 | generate_waymo_pcd.py:68 | 🟡 中等 | 逻辑不一致 | 建议修复 |
| 3 | generate_waymo_pcd.py:223 | 🔴 严重 | 计算错误 | 需修复 |
| 4 | read_kitti360.py:161 | 🟡 中等 | 数据错误 | 建议修复 |
| 5 | generate_pcd.py:186 | 🟡 中等 | 硬编码 | 建议修复 |
| 6 | 多处 | 🟢 轻微 | 拼写错误 | 建议修复 |
| 7 | generate_pcd.py:262 | 🟡 中等 | 潜在越界 | 建议检查 |

---

## 总结

### 算法流程完整性
✅ 预处理流程完整，涵盖了数据读取、深度估计、语义分割、点云生成等关键步骤

### 数据流正确性
⚠️ 大部分数据流正确，但存在几处关键错误需要修复

### 代码质量
⚠️ 存在硬编码、拼写错误、逻辑错误等问题，需要改进

### 建议优先级
1. **高优先级**: 修复错误1和错误3（严重功能错误）
2. **中优先级**: 修复错误2、4、5、7（影响正确性或可维护性）
3. **低优先级**: 修复错误6（代码规范问题）

### 改进建议
1. 统一两个数据集的sparsity逻辑
2. 使用配置参数而非硬编码
3. 添加输入验证和边界检查
4. 改进变量命名和代码注释
5. 添加单元测试验证关键功能

