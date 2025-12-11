#!/usr/bin/env python3
"""
独立脚本：为 NuScenes 场景生成深度图和语义mask

功能：
1. 生成 metric depth（使用 Metric3D）
2. 生成语义分割（使用 OneFormer）
3. 生成天空mask（从语义分割中提取）

用法：
    python gen_nuscenes_depth_mask.py \
        --scene_dir /path/to/scene/000 \
        --gen_depth \
        --gen_semantic \
        --gen_sky_mask \
        --depth_gpu_id 6 \
        --semantic_gpu_id 0
"""

import os
import sys
import cv2
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from rich.console import Console

CONSOLE = Console(width=120)


def gen_metric_depth(scene_dir: str, dataset: str = 'nuscenes', gpu_id: str = '6'):
    """
    生成 metric depth 使用 Metric3D。
    
    Args:
        scene_dir: 场景目录路径（包含 images/ 子目录）
        dataset: 数据集类型（用于 Metric3D 确定内参）
        gpu_id: GPU ID
    """
    if not os.path.exists(scene_dir):
        raise ValueError(f"Scene directory not found: {scene_dir}")
    
    images_dir = os.path.join(scene_dir, 'images')
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    depth_dir = os.path.join(scene_dir, 'depth')
    if os.path.exists(depth_dir):
        CONSOLE.log(f"Depth directory already exists: {depth_dir}, skipping depth generation")
        return
    
    metric3d_path = os.getenv('METRIC3D_PATH', '/home/smiao/Gen_Dataset/dataset_methods/metric3d')
    model_path = os.getenv('METRIC3D_MODEL_PATH', '/nas/users/smiao/model_zoo/metric3d/metric_depth_vit_giant2_800k.pth')
    
    if not os.path.exists(metric3d_path):
        raise ValueError(f"METRIC3D_PATH not found: {metric3d_path}. Please set METRIC3D_PATH environment variable.")
    
    CONSOLE.log(f"Generating metric depth for scene: {scene_dir}")
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {metric3d_path}/mono/tools/test_scale_cano.py \
        {metric3d_path}/mono/configs/HourglassDecoder/vit.raft5.giant2.py \
        --load-from {model_path} \
        --test_data_path {scene_dir} --show-dir {depth_dir} \
        --dataset {dataset} \
        --launcher None"
    
    os.system(cmd)
    CONSOLE.log(f"Depth generation completed. Results saved to: {depth_dir}")


def gen_semantic(scene_dir: str, gpu_id: str = '0', model_name: str = 'shi-labs/oneformer_cityscapes_swin_large'):
    """
    生成语义分割使用 OneFormer。
    
    Args:
        scene_dir: 场景目录路径（包含 images/ 子目录）
        gpu_id: GPU ID
        model_name: OneFormer 模型名称
    """
    if not os.path.exists(scene_dir):
        raise ValueError(f"Scene directory not found: {scene_dir}")
    
    images_dir = os.path.join(scene_dir, 'images')
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    semantic_dir = os.path.join(scene_dir, 'semantic')
    if os.path.exists(semantic_dir):
        CONSOLE.log(f"Semantic directory already exists: {semantic_dir}, skipping semantic generation")
        return
    
    script_path = os.path.join(os.path.dirname(__file__), 'gen_semantic_oneformer.py')
    if not os.path.exists(script_path):
        raise ValueError(f"Semantic generation script not found: {script_path}")
    
    CONSOLE.log(f"Generating semantic segmentation for scene: {scene_dir}")
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {script_path} \
           --input_dir {scene_dir} \
           --output_dir {scene_dir} \
           --model_name {model_name} \
           --task semantic \
           --device cuda \
           --gpu_id {gpu_id}"
    
    os.system(cmd)
    CONSOLE.log(f"Semantic generation completed. Results saved to: {semantic_dir}")


def gen_sky_mask(scene_dir: str):
    """
    从语义分割结果生成天空mask。
    
    Args:
        scene_dir: 场景目录路径（包含 semantic/instance/ 子目录）
    """
    if not os.path.exists(scene_dir):
        raise ValueError(f"Scene directory not found: {scene_dir}")
    
    instance_path = os.path.join(scene_dir, 'semantic', 'instance')
    if not os.path.exists(instance_path):
        raise ValueError(f"Semantic instance directory not found: {instance_path}. Please run semantic generation first.")
    
    save_path = os.path.join(scene_dir, 'sky_masks')
    os.makedirs(save_path, exist_ok=True)
    
    # 获取图像尺寸（从第一张图像）
    instance_files = sorted([f for f in os.listdir(instance_path) if f.endswith('.png')])
    if len(instance_files) == 0:
        raise ValueError(f"No instance files found in {instance_path}")
    
    first_instance = cv2.imread(os.path.join(instance_path, instance_files[0]), -1)
    if first_instance is None:
        raise ValueError(f"Failed to read first instance file: {instance_files[0]}")
    
    image_height, image_width = first_instance.shape[:2]
    
    CONSOLE.log(f"Generating sky masks for {len(instance_files)} images...")
    
    for instance_fn in instance_files:
        mask = np.ones((image_height, image_width), dtype=np.uint8)
        instance_file = os.path.join(instance_path, instance_fn)
        instance = cv2.imread(instance_file, -1)
        
        if instance is None:
            CONSOLE.log(f"Warning: Failed to read {instance_fn}, skipping")
            continue
        
        # Sky class ID = 10, set to 0 (sky), others to 255 (non-sky)
        mask[instance == 10] = 0
        mask[instance != 10] = 255
        mask = Image.fromarray(mask)
        
        # Save with same filename format: {frame_idx:03d}_{cam_id}.png
        mask.save(os.path.join(save_path, instance_fn))
    
    CONSOLE.log(f"Sky mask generation completed. Results saved to: {save_path}")


def main():
    parser = ArgumentParser(description="Generate depth and masks for NuScenes scenes")
    parser.add_argument('--scene_dir', type=str, required=True,
                       help='Path to scene directory (e.g., /path/to/processed/mini/000)')
    parser.add_argument('--gen_depth', action='store_true',
                       help='Generate metric depth using Metric3D')
    parser.add_argument('--gen_semantic', action='store_true',
                       help='Generate semantic segmentation using OneFormer')
    parser.add_argument('--gen_sky_mask', action='store_true',
                       help='Generate sky masks from semantic segmentation')
    parser.add_argument('--dataset', type=str, default='nuscenes',
                       help='Dataset type for Metric3D (default: nuscenes)')
    parser.add_argument('--depth_gpu_id', type=str, default='6',
                       help='GPU ID for depth generation (default: 6)')
    parser.add_argument('--semantic_gpu_id', type=str, default='0',
                       help='GPU ID for semantic generation (default: 0)')
    parser.add_argument('--model_name', type=str, default='shi-labs/oneformer_cityscapes_swin_large',
                       help='OneFormer model name (default: shi-labs/oneformer_cityscapes_swin_large)')
    
    args = parser.parse_args()
    
    # 验证场景目录
    if not os.path.exists(args.scene_dir):
        CONSOLE.log(f"[red]Error: Scene directory not found: {args.scene_dir}[/red]")
        sys.exit(1)
    
    images_dir = os.path.join(args.scene_dir, 'images')
    if not os.path.exists(images_dir):
        CONSOLE.log(f"[red]Error: Images directory not found: {images_dir}[/red]")
        sys.exit(1)
    
    # 执行生成任务
    if args.gen_depth:
        try:
            gen_metric_depth(args.scene_dir, args.dataset, args.depth_gpu_id)
        except Exception as e:
            CONSOLE.log(f"[red]Error generating depth: {e}[/red]")
            sys.exit(1)
    
    if args.gen_semantic:
        try:
            gen_semantic(args.scene_dir, args.semantic_gpu_id, args.model_name)
        except Exception as e:
            CONSOLE.log(f"[red]Error generating semantic: {e}[/red]")
            sys.exit(1)
    
    if args.gen_sky_mask:
        try:
            gen_sky_mask(args.scene_dir)
        except Exception as e:
            CONSOLE.log(f"[red]Error generating sky mask: {e}[/red]")
            sys.exit(1)
    
    if not (args.gen_depth or args.gen_semantic or args.gen_sky_mask):
        CONSOLE.log("[yellow]Warning: No generation task specified. Use --gen_depth, --gen_semantic, or --gen_sky_mask[/yellow]")
    
    CONSOLE.log("[green]All tasks completed![/green]")


if __name__ == "__main__":
    main()

