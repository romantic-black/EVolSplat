# 深度图工具函数使用说明

## 概述

`depth_utils.py` 提供了在使用深度图时进行插值和其他预处理的工具函数。

由于 `test_scale_cano_hub.py` 现在保存的深度图是去除padding后的模型输出尺寸（不插值回原始尺寸），以节省存储空间，因此需要使用这些工具函数在需要时将深度图插值回原始尺寸。

## 主要功能

### 1. 加载深度图和元数据

```python
from depth_utils import load_depth_with_metadata

depth, metadata = load_depth_with_metadata('/path/to/depth/000_0.npy')
print(f"Depth shape: {depth.shape}")  # 例如: (616, 1064)
print(f"Original image shape: {metadata['ori_shape']}")  # 例如: [900, 1600]
```

### 2. 插值到原始尺寸（推荐）

```python
from depth_utils import process_depth_for_use

# 自动加载并插值到原始图像尺寸
depth, metadata = process_depth_for_use('/path/to/depth/000_0.npy')
print(f"Depth shape: {depth.shape}")  # 例如: (900, 1600)，与原始图像尺寸一致
```

### 3. 插值到指定尺寸

```python
from depth_utils import process_depth_for_use

# 插值到指定尺寸
depth, metadata = process_depth_for_use(
    '/path/to/depth/000_0.npy',
    target_shape=(900, 1600),
    interpolation_mode='bilinear',  # 或 'nearest'
)
```

### 4. 可视化深度图

```python
from depth_utils import process_depth_for_use, visualize_depth
import cv2

depth, metadata = process_depth_for_use('/path/to/depth/000_0.npy')
depth_vis = visualize_depth(depth, max_depth=200.0)  # max_depth 单位为米
cv2.imwrite('depth_visualization.png', depth_vis)
```

### 5. 转换为点云

```python
from depth_utils import process_depth_for_use, depth_to_pointcloud
import cv2

# 加载深度图和RGB图像
depth, metadata = process_depth_for_use('/path/to/depth/000_0.npy')
rgb = cv2.imread('/path/to/image/000_0.jpg')

# 转换为点云（带颜色）
if metadata['intrinsic'] is not None:
    points = depth_to_pointcloud(depth, metadata['intrinsic'], rgb)
    print(f"Point cloud shape: {points.shape}")  # [N, 6] (x, y, z, r, g, b)
```

### 6. 批量处理

```python
from depth_utils import batch_process_depths

# 批量处理目录中的所有深度图
results = batch_process_depths(
    depth_dir='/path/to/depth/dir',
    target_shape=(900, 1600),  # 可选，None 则使用每个深度图的原始尺寸
    interpolation_mode='bilinear',
    save_interpolated=True,  # 是否保存插值后的深度图
    output_dir='/path/to/output/dir',  # 输出目录
)

# results 是一个字典，key 为文件名，value 为 (depth, metadata) 元组
for filename, (depth, metadata) in results.items():
    print(f"{filename}: {depth.shape}")
```

## 元数据说明

每个深度图文件（`.npy`）都会有一个对应的元数据文件（`_meta.npz`），包含以下信息：

- `ori_shape`: 原始图像尺寸 `[H, W]`
- `depth_shape`: 当前深度图尺寸 `[H, W]`（去除padding后的模型输出尺寸）
- `scale_info`: 尺度因子（用于深度值恢复）
- `normalize_scale`: 归一化尺度（通常为 1.0）
- `intrinsic`: 相机内参 `[fx, fy, cx, cy]`

## 注意事项

1. **存储空间**: 不插值保存可以显著减少存储空间（例如从 900×1600 减少到 616×1064）
2. **插值模式**: 
   - `bilinear`: 双线性插值，适合连续深度值
   - `nearest`: 最近邻插值，适合稀疏深度图
3. **深度值单位**: 深度值单位为**米**（metric depth）
4. **性能**: 批量处理时，如果所有深度图需要插值到相同尺寸，建议使用 `batch_process_depths`

## 完整示例

```python
import sys
import os
sys.path.append('/path/to/metric3d/mono/tools')

from depth_utils import process_depth_for_use, visualize_depth, depth_to_pointcloud
import cv2
import numpy as np

# 1. 加载并处理深度图
depth_path = '/path/to/depth/000_0.npy'
depth, metadata = process_depth_for_use(depth_path)

print(f"Depth shape: {depth.shape}")
print(f"Original shape: {metadata['ori_shape']}")
print(f"Depth range: [{depth[depth > 0].min():.2f}, {depth[depth > 0].max():.2f}] meters")

# 2. 可视化
depth_vis = visualize_depth(depth, max_depth=200.0)
cv2.imwrite('depth_vis.png', depth_vis)

# 3. 转换为点云（如果有RGB图像）
rgb_path = '/path/to/image/000_0.jpg'
if os.path.exists(rgb_path) and metadata['intrinsic'] is not None:
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    points = depth_to_pointcloud(depth, metadata['intrinsic'], rgb)
    
    # 保存点云（PLY格式）
    with open('pointcloud.ply', 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for p in points:
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(p[3])} {int(p[4])} {int(p[5])}\n')
```

