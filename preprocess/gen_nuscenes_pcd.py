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
from argparse import ArgumentParser
from rich.console import Console

# 添加路径以导入模块
sys.path.insert(0, os.path.dirname(__file__))
from read_dataset.read_nuscenes import ReadNuScenesData
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
    import imageio.v2 as imageio
    first_image_path = os.path.join(images_dir, sorted(image_files)[0])
    img = imageio.imread(first_image_path)
    H, W = img.shape[0], img.shape[1]
    
    return H, W


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
    
    # 解析相机列表
    cameras = [int(c.strip()) for c in args.cameras.split(',')]
    
    # 确定序列ID（从scene_dir提取或使用参数）
    if args.sequence is None:
        # 尝试从scene_dir提取序列ID
        scene_name = os.path.basename(os.path.normpath(args.scene_dir))
        try:
            sequence = int(scene_name)
        except ValueError:
            CONSOLE.log(f"[yellow]Warning: Cannot extract sequence ID from {scene_name}, using 0[/yellow]")
            sequence = 0
    else:
        sequence = args.sequence
    
    # 确定root_dir（如果未提供，使用scene_dir的父目录）
    # read_nuscenes.py 会从 root_dir 中查找场景，所以 root_dir 应该是包含场景目录的父目录
    if args.root_dir is None:
        # scene_dir 本身已经是场景目录，所以 root_dir 应该是其父目录
        args.root_dir = os.path.dirname(os.path.normpath(args.scene_dir))
    
    # 读取位姿和内参
    CONSOLE.log(f"Reading poses and intrinsics from scene: {args.scene_dir}")
    try:
        reader = ReadNuScenesData(
            save_dir=args.root_dir,  # 不使用，但需要提供
            sequence=sequence,
            root_dir=args.root_dir,
            cameras=cameras
        )
        # generate_json 现在直接返回场景路径，不再创建新目录
        poses, intrinsics, dir_name, info = reader.generate_json(
            frame_start=args.frame_start,
            num_frames=args.num_frames
        )
        H, W = info[0], info[1]
        CONSOLE.log(f"Loaded {len(poses)} poses and intrinsics (H={H}, W={W})")
        
        # 验证 dir_name 是否与 scene_dir 一致
        if os.path.normpath(dir_name) != os.path.normpath(args.scene_dir):
            CONSOLE.log(f"[yellow]Warning: dir_name ({dir_name}) != scene_dir ({args.scene_dir})[/yellow]")
            CONSOLE.log(f"[yellow]Using provided scene_dir: {args.scene_dir}[/yellow]")
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

