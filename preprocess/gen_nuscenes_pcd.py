#!/usr/bin/env python3
"""
可配置的点云生成脚本：为 NuScenes 场景生成点云

功能：
1. 从深度图生成点云
2. 支持多种稀疏度级别（Drop90, Drop80, Drop50, Drop25, full）
3. 支持天空过滤和深度一致性检查
4. 支持边界框裁剪

用法：
    python gen_nuscenes_pcd.py \
        --scene_dir /path/to/scene/000 \
        --sparsity Drop50 \
        --filter_sky \
        --depth_consistency \
        --downscale 2
"""

import os
import sys
import numpy as np
import imageio.v2 as imageio
from argparse import ArgumentParser
from rich.console import Console

# 添加路径以导入模块
sys.path.insert(0, os.path.dirname(__file__))
from read_dataset.generate_nuscenes_pcd import NuScenesPCDGenerator

CONSOLE = Console(width=120)


def get_image_dimensions(scene_dir: str):
    """
    从场景目录获取图像尺寸。
    
    Args:
        scene_dir: 场景目录路径
        
    Returns:
        (H, W): 图像高度和宽度
    """
    images_dir = os.path.join(scene_dir, 'images')
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    image_files = [f for f in os.listdir(images_dir) 
                   if f.endswith('.jpg') or f.endswith('.png')]
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {images_dir}")
    
    # 读取第一张图像获取尺寸
    first_image_path = os.path.join(images_dir, sorted(image_files)[0])
    img = imageio.imread(first_image_path)
    H, W = img.shape[0], img.shape[1]
    
    return H, W


def read_poses_intrinsics_direct(scene_dir: str, frame_start: int = 0, num_frames: int = None):
    """
    直接从场景目录读取poses和intrinsics，按照NuScenes场景文件夹的预定结构。
    
    Args:
        scene_dir: 场景目录路径
        frame_start: 起始帧索引（默认0）
        num_frames: 要读取的帧数（None表示读取所有帧）
        
    Returns:
        poses: List of poses (num_frames * num_cameras, 4, 4)
        intrinsics: List of intrinsics (num_frames * num_cameras, 4, 4)
        info: Tuple (H, W) - 图像尺寸
    """
    extrinsics_dir = os.path.join(scene_dir, 'extrinsics')
    intrinsics_dir = os.path.join(scene_dir, 'intrinsics')
    images_dir = os.path.join(scene_dir, 'images')
    
    if not os.path.exists(extrinsics_dir):
        raise ValueError(f"Extrinsics directory not found: {extrinsics_dir}")
    if not os.path.exists(intrinsics_dir):
        raise ValueError(f"Intrinsics directory not found: {intrinsics_dir}")
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # 获取所有帧索引和相机ID
    extrinsic_files = sorted([f for f in os.listdir(extrinsics_dir) if f.endswith('.txt')])
    frame_cam_pairs = []
    for f in extrinsic_files:
        try:
            parts = f.replace('.txt', '').split('_')
            if len(parts) >= 2:
                frame_idx = int(parts[0])
                cam_id = int(parts[1])
                frame_cam_pairs.append((frame_idx, cam_id))
        except (ValueError, IndexError):
            continue
    
    # 过滤帧范围
    if num_frames is not None:
        frame_end = frame_start + num_frames
        frame_cam_pairs = [(f, c) for f, c in frame_cam_pairs if frame_start <= f < frame_end]
    else:
        frame_cam_pairs = [(f, c) for f, c in frame_cam_pairs if f >= frame_start]
    
    # 按frame_idx和cam_id排序
    frame_cam_pairs.sort(key=lambda x: (x[0], x[1]))
    
    # 加载第一帧第一相机的pose用于对齐
    camera_front_start = None
    first_frame_cam = None
    for frame_idx, cam_id in frame_cam_pairs:
        if cam_id == 0:  # 使用相机0作为参考
            first_extrinsic_file = os.path.join(extrinsics_dir, f'{frame_idx:03d}_{cam_id}.txt')
            if os.path.exists(first_extrinsic_file):
                camera_front_start = np.loadtxt(first_extrinsic_file)
                first_frame_cam = (frame_idx, cam_id)
                break
    
    if camera_front_start is None:
        # Fallback: 使用第一个可用的extrinsic
        if len(frame_cam_pairs) > 0:
            first_frame_cam = frame_cam_pairs[0]
            first_extrinsic_file = os.path.join(extrinsics_dir, f'{first_frame_cam[0]:03d}_{first_frame_cam[1]}.txt')
            if os.path.exists(first_extrinsic_file):
                camera_front_start = np.loadtxt(first_extrinsic_file)
                CONSOLE.log(f"Warning: Using frame {first_frame_cam[0]} camera {first_frame_cam[1]} for alignment")
    
    # 读取所有poses和intrinsics
    poses = []
    intrinsics = []
    
    # 获取相机ID列表
    cam_ids = sorted(set(cam_id for _, cam_id in frame_cam_pairs))
    
    # 读取每个相机的内参（固定值）
    cam_intrinsics_dict = {}
    for cam_id in cam_ids:
        intrinsic_file = os.path.join(intrinsics_dir, f'{cam_id}.txt')
        if os.path.exists(intrinsic_file):
            intrinsic_data = np.loadtxt(intrinsic_file)
            fx, fy, cx, cy = intrinsic_data[0], intrinsic_data[1], intrinsic_data[2], intrinsic_data[3]
            # 转换为4x4矩阵
            cam_intrinsic = np.eye(4)
            cam_intrinsic[0, 0] = fx
            cam_intrinsic[1, 1] = fy
            cam_intrinsic[0, 2] = cx
            cam_intrinsic[1, 2] = cy
            cam_intrinsics_dict[cam_id] = cam_intrinsic
        else:
            CONSOLE.log(f"Warning: Intrinsic file not found: {intrinsic_file}")
    
    # 读取每个frame_cam对的pose
    for frame_idx, cam_id in frame_cam_pairs:
        extrinsic_file = os.path.join(extrinsics_dir, f'{frame_idx:03d}_{cam_id}.txt')
        if not os.path.exists(extrinsic_file):
            CONSOLE.log(f"Warning: Extrinsic file not found: {extrinsic_file}, skipping")
            continue
        
        # 加载外参（cam_to_world）
        cam2world = np.loadtxt(extrinsic_file)
        
        # 对齐到第一帧第一相机（如果可用）
        if camera_front_start is not None:
            cam2world = np.linalg.inv(camera_front_start) @ cam2world
        
        # OPENCV2DATASET是单位矩阵，不需要转换
        OPENCV2DATASET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        cam2world = cam2world @ OPENCV2DATASET
        
        poses.append(cam2world)
        
        # 添加对应的内参
        if cam_id in cam_intrinsics_dict:
            intrinsics.append(cam_intrinsics_dict[cam_id])
        else:
            # 如果内参不存在，使用单位矩阵
            intrinsics.append(np.eye(4))
    
    # 获取图像尺寸
    H, W = get_image_dimensions(scene_dir)
    
    return np.array(poses), np.array(intrinsics), (H, W)


def main():
    parser = ArgumentParser(description="Generate point cloud for NuScenes scenes")
    parser.add_argument('--scene_dir', type=str, required=True,
                       help='Path to scene directory (e.g., /path/to/processed/mini/000)')
    parser.add_argument('--root_dir', type=str, default=None,
                       help='Root directory containing processed scenes (optional, for finding scene)')
    parser.add_argument('--sequence', type=int, default=None,
                       help='Sequence ID (optional, if not provided, will extract from scene_dir)')
    parser.add_argument('--cameras', type=str, default='0',
                       help='Comma-separated list of camera IDs (0-5, default: 0 for CAM_FRONT)')
    parser.add_argument('--frame_start', type=int, default=0,
                       help='Starting frame index (default: 0)')
    parser.add_argument('--num_frames', type=int, default=None,
                       help='Number of frames to process (None means all frames, default: None)')
    parser.add_argument('--sparsity', type=str, 
                       choices=['Drop90', 'Drop50', 'Drop80', 'Drop25', 'full'],
                       default='full',
                       help='Point cloud sparsity level (default: full)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Subdirectory name for saving point cloud (default: same as sparsity)')
    parser.add_argument('--filter_sky', action='store_true',
                       help='Filter sky regions using sky_masks')
    parser.add_argument('--depth_consistency', action='store_true', default=True,
                       help='Enable depth consistency check (default: True)')
    parser.add_argument('--no_depth_consistency', dest='depth_consistency', action='store_false',
                       help='Disable depth consistency check')
    parser.add_argument('--downscale', type=int, default=2,
                       help='Downsampling scale for point cloud generation (default: 2)')
    parser.add_argument('--use_bbx', action='store_true', default=True,
                       help='Use bounding box filtering (default: True)')
    parser.add_argument('--no_bbx', dest='use_bbx', action='store_false',
                       help='Disable bounding box filtering')
    
    args = parser.parse_args()
    
    # 验证场景目录
    if not os.path.exists(args.scene_dir):
        CONSOLE.log(f"[red]Error: Scene directory not found: {args.scene_dir}[/red]")
        sys.exit(1)
    
    # 检查必要的目录和文件
    required_dirs = ['images', 'extrinsics', 'intrinsics', 'depth']
    for dir_name in required_dirs:
        dir_path = os.path.join(args.scene_dir, dir_name)
        if not os.path.exists(dir_path):
            CONSOLE.log(f"[red]Error: Required directory not found: {dir_path}[/red]")
            sys.exit(1)
    
    if args.filter_sky:
        sky_masks_dir = os.path.join(args.scene_dir, 'sky_masks')
        if not os.path.exists(sky_masks_dir):
            CONSOLE.log(f"[yellow]Warning: Sky masks directory not found: {sky_masks_dir}[/yellow]")
            CONSOLE.log("[yellow]Sky filtering will be disabled. Run gen_nuscenes_depth_mask.py --gen_sky_mask first.[/yellow]")
            args.filter_sky = False
    
    # 直接从场景目录读取位姿和内参（按照NuScenes场景文件夹的预定结构）
    CONSOLE.log(f"Reading poses and intrinsics directly from scene: {args.scene_dir}")
    try:
        poses, intrinsics, info = read_poses_intrinsics_direct(
            scene_dir=args.scene_dir,
            frame_start=args.frame_start,
            num_frames=args.num_frames
        )
        H, W = info[0], info[1]
        CONSOLE.log(f"Loaded {len(poses)} poses and intrinsics (H={H}, W={W})")
    except Exception as e:
        CONSOLE.log(f"[red]Error reading poses and intrinsics: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 验证深度文件存在
    depth_dir = os.path.join(args.scene_dir, 'depth')
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]
    if len(depth_files) == 0:
        CONSOLE.log(f"[red]Error: No depth files found in {depth_dir}[/red]")
        CONSOLE.log("[yellow]Please run gen_nuscenes_depth_mask.py --gen_depth first.[/yellow]")
        sys.exit(1)
    
    # 创建点云生成器
    save_dir = args.save_dir if args.save_dir is not None else args.sparsity
    pcd_generator = NuScenesPCDGenerator(
        spars=args.sparsity,
        save_dir=save_dir,
        frame_start=args.frame_start,
        filer_sky=args.filter_sky,
        depth_cosistency=args.depth_consistency
    )
    pcd_generator.use_bbx = args.use_bbx
    
    # 生成点云
    CONSOLE.log(f"Generating point cloud with sparsity: {args.sparsity}")
    try:
        pcd_generator.forward(
            dir_name=args.scene_dir,
            poses=poses,
            intrinsics=intrinsics,
            H=H,
            W=W,
            down_scale=args.downscale
        )
        CONSOLE.log(f"[green]Point cloud generation completed![/green]")
        CONSOLE.log(f"Point cloud saved to: {os.path.join(args.scene_dir, save_dir)}")
    except Exception as e:
        CONSOLE.log(f"[red]Error generating point cloud: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

