"""
深度图工具函数：用于在使用深度图时进行插值和其他预处理

主要功能：
1. 将保存的深度图（去除padding后的模型输出尺寸）插值回原始图像尺寸
2. 处理深度图的尺度恢复
3. 提供其他常用的深度图预处理功能
"""

import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Tuple, Optional, Dict, Union


def load_depth_with_metadata(depth_path: str) -> Tuple[np.ndarray, Dict]:
    """
    加载深度图和对应的元数据
    
    Args:
        depth_path: 深度图文件路径（.npy）
    
    Returns:
        depth: 深度图数组 [H, W]
        metadata: 元数据字典，包含：
            - ori_shape: 原始图像尺寸 [H, W]
            - depth_shape: 当前深度图尺寸 [H, W]
            - scale_info: 尺度因子
            - normalize_scale: 归一化尺度
            - intrinsic: 相机内参 [fx, fy, cx, cy]
    """
    if not osp.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    
    depth = np.load(depth_path)
    
    # 尝试加载元数据
    metadata_path = depth_path.replace('.npy', '_meta.npz')
    if osp.exists(metadata_path):
        metadata = dict(np.load(metadata_path))
    else:
        # 如果没有元数据，使用默认值（假设深度图已经是原始尺寸）
        metadata = {
            'ori_shape': list(depth.shape),
            'depth_shape': list(depth.shape),
            'scale_info': 1.0,
            'normalize_scale': 1.0,
            'intrinsic': None,
        }
    
    return depth, metadata


def interpolate_depth_to_original_size(
    depth: Union[np.ndarray, torch.Tensor],
    ori_shape: Tuple[int, int],
    mode: str = 'bilinear',
) -> Union[np.ndarray, torch.Tensor]:
    """
    将深度图插值回原始图像尺寸
    
    Args:
        depth: 深度图，可以是 numpy array 或 torch.Tensor，形状 [H, W]
        ori_shape: 目标尺寸 (H, W)
        mode: 插值模式，'bilinear' 或 'nearest'
    
    Returns:
        插值后的深度图，与输入类型相同
    """
    is_torch = isinstance(depth, torch.Tensor)
    
    if not is_torch:
        depth = torch.from_numpy(depth).float()
    
    # 确保深度图是 2D
    if depth.dim() == 2:
        depth = depth[None, None, :, :]  # [1, 1, H, W]
    elif depth.dim() == 3:
        depth = depth.unsqueeze(0)  # [1, C, H, W]
    
    # 插值
    if mode == 'bilinear':
        depth_resized = F.interpolate(
            depth,
            size=ori_shape,
            mode='bilinear',
            align_corners=False,
        )
    elif mode == 'nearest':
        depth_resized = F.interpolate(
            depth,
            size=ori_shape,
            mode='nearest',
        )
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
    
    # 恢复原始维度
    depth_resized = depth_resized.squeeze()
    
    if not is_torch:
        depth_resized = depth_resized.numpy()
    
    return depth_resized


def process_depth_for_use(
    depth_path: str,
    target_shape: Optional[Tuple[int, int]] = None,
    interpolation_mode: str = 'bilinear',
) -> Tuple[np.ndarray, Dict]:
    """
    完整处理深度图以便使用：加载、插值到目标尺寸
    
    Args:
        depth_path: 深度图文件路径（.npy）
        target_shape: 目标尺寸 (H, W)，如果为 None 则使用原始图像尺寸
        interpolation_mode: 插值模式，'bilinear' 或 'nearest'
    
    Returns:
        depth: 处理后的深度图 [H, W]
        metadata: 元数据字典
    """
    depth, metadata = load_depth_with_metadata(depth_path)
    
    # 确定目标尺寸
    if target_shape is None:
        target_shape = tuple(metadata['ori_shape'])
    
    # 如果当前尺寸与目标尺寸不同，进行插值
    current_shape = tuple(metadata['depth_shape'])
    if current_shape != target_shape:
        depth = interpolate_depth_to_original_size(
            depth,
            target_shape,
            mode=interpolation_mode,
        )
    
    return depth, metadata


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将深度图转换为点云
    
    Args:
        depth: 深度图 [H, W]，单位为米
        intrinsic: 相机内参 [fx, fy, cx, cy]
        rgb: 可选的RGB图像 [H, W, 3]，用于着色点云
    
    Returns:
        points: 点云数组 [N, 3] 或 [N, 6]（如果提供了RGB）
    """
    H, W = depth.shape
    fx, fy, cx, cy = intrinsic
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 过滤有效深度点
    valid_mask = (depth > 0) & np.isfinite(depth)
    
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth[valid_mask]
    
    # 转换为3D点
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid
    
    points_3d = np.stack([x, y, z], axis=1)
    
    # 如果提供了RGB，添加颜色
    if rgb is not None:
        rgb_valid = rgb[valid_mask]
        points_3d = np.concatenate([points_3d, rgb_valid], axis=1)
    
    return points_3d


def visualize_depth(
    depth: np.ndarray,
    max_depth: Optional[float] = None,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    将深度图可视化为彩色图像
    
    Args:
        depth: 深度图 [H, W]
        max_depth: 最大深度值，用于归一化。如果为 None，使用 depth 的最大值
        colormap: OpenCV colormap，默认 JET
    
    Returns:
        可视化图像 [H, W, 3]，BGR格式
    """
    # 过滤无效值
    valid_mask = (depth > 0) & np.isfinite(depth)
    
    if valid_mask.sum() == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    
    # 归一化到 0-255
    depth_normalized = depth.copy()
    depth_normalized[~valid_mask] = 0
    
    if max_depth is None:
        max_depth = depth_normalized[valid_mask].max()
    
    if max_depth > 0:
        depth_normalized = np.clip(depth_normalized / max_depth * 255, 0, 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth_normalized, dtype=np.uint8)
    
    # 应用colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    
    return depth_colored


def batch_process_depths(
    depth_dir: str,
    target_shape: Optional[Tuple[int, int]] = None,
    interpolation_mode: str = 'bilinear',
    save_interpolated: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """
    批量处理深度图目录中的所有深度图
    
    Args:
        depth_dir: 深度图目录路径
        target_shape: 目标尺寸 (H, W)，如果为 None 则使用每个深度图的原始尺寸
        interpolation_mode: 插值模式
        save_interpolated: 是否保存插值后的深度图
        output_dir: 输出目录，如果 save_interpolated=True 且 output_dir 不为 None
    
    Returns:
        results: 字典，key 为文件名，value 为 (depth, metadata) 元组
    """
    results = {}
    
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy') and not f.endswith('_meta.npz')]
    
    for depth_file in depth_files:
        depth_path = osp.join(depth_dir, depth_file)
        try:
            depth, metadata = process_depth_for_use(
                depth_path,
                target_shape=target_shape,
                interpolation_mode=interpolation_mode,
            )
            results[depth_file] = (depth, metadata)
            
            # 保存插值后的深度图
            if save_interpolated and output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                output_path = osp.join(output_dir, depth_file)
                np.save(output_path, depth.astype(np.float16))
                
        except Exception as e:
            print(f"Error processing {depth_file}: {e}")
            continue
    
    return results


# 示例使用
if __name__ == '__main__':
    # 示例：加载并处理单个深度图
    depth_path = '/path/to/depth/000_0.npy'
    
    # 方法1：加载并插值到原始尺寸
    depth, metadata = process_depth_for_use(depth_path)
    print(f"Depth shape: {depth.shape}")
    print(f"Original shape: {metadata['ori_shape']}")
    
    # 方法2：加载并插值到指定尺寸
    depth_resized, metadata = process_depth_for_use(
        depth_path,
        target_shape=(900, 1600),
    )
    print(f"Resized depth shape: {depth_resized.shape}")
    
    # 方法3：可视化
    depth_vis = visualize_depth(depth, max_depth=200.0)
    cv2.imwrite('depth_visualization.png', depth_vis)
    
    # 方法4：转换为点云
    if metadata['intrinsic'] is not None:
        points = depth_to_pointcloud(depth, metadata['intrinsic'])
        print(f"Point cloud shape: {points.shape}")

