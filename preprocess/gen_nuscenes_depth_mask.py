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
import subprocess
import shutil
from argparse import ArgumentParser
from PIL import Image
from rich.console import Console

CONSOLE = Console(width=120)


def gen_metric_depth(scene_dir: str, dataset: str = 'nuscenes', gpu_id: str = '6', cam_id: int = 0):
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
    
    # 如果目录已存在，删除它以重新生成
    if os.path.exists(depth_dir):
        depth_files = [f for f in os.listdir(depth_dir) if f.endswith(('.png', '.npy', '.npz'))]
        if len(depth_files) > 0:
            CONSOLE.log(f"Depth directory exists with {len(depth_files)} files. Removing to regenerate...")
        else:
            CONSOLE.log(f"Depth directory exists but is empty. Removing to regenerate...")
        shutil.rmtree(depth_dir)
    
    # 创建depth目录
    os.makedirs(depth_dir, exist_ok=True)
    
    # 默认路径：相对于当前脚本的路径
    default_metric3d_path = os.path.join(os.path.dirname(__file__), 'metric3d')
    metric3d_path = os.getenv('METRIC3D_PATH', default_metric3d_path)
    
    if not os.path.exists(metric3d_path):
        raise ValueError(f"METRIC3D_PATH not found: {metric3d_path}. Please set METRIC3D_PATH environment variable.")
    
    CONSOLE.log(f"Generating metric depth for scene: {scene_dir}")
    CONSOLE.log(f"Using torch.hub Metric3D model (no local model file required)")
    
    # 构建命令
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    # 使用基于 torch.hub 的 Metric3D 推理脚本
    # 注意：hub 版本不需要 --load-from 参数（模型从 GitHub 加载）
    cmd = [
        sys.executable,
        os.path.join(metric3d_path, 'mono', 'tools', 'test_scale_cano_hub.py'),
        os.path.join(metric3d_path, 'mono', 'configs', 'HourglassDecoder', 'vit.raft5.giant2.py'),
        '--test_data_path', scene_dir,
        '--show-dir', depth_dir,
        '--dataset', dataset,
        '--launcher', 'None',
        '--cam_id', str(cam_id)
    ]
    
    try:
        CONSOLE.log(f"Running command: python {' '.join(cmd[1:])}")  # 不显示完整路径以保持简洁
        
        result = subprocess.run(
            cmd,
            env=env,
            cwd=metric3d_path,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            CONSOLE.log(f"[red]Depth generation command failed with exit code {result.returncode}[/red]")
            if error_msg:
                # 显示完整的错误信息（最多2000字符）
                error_display = error_msg[-2000:] if len(error_msg) > 2000 else error_msg
                CONSOLE.log(f"[red]Error output:[/red]")
                CONSOLE.log(f"[red]{error_display}[/red]")
            raise RuntimeError(f"Depth generation failed with exit code {result.returncode}")
        
        # 显示输出（如果有）
        if result.stdout:
            CONSOLE.log(f"[dim]{result.stdout[-500:]}[/dim]")  # 显示最后500字符
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Depth generation timed out after 2 hours")
    except Exception as e:
        raise RuntimeError(f"Depth generation error: {e}")
    
    # 验证输出目录是否存在且有文件
    if not os.path.exists(depth_dir):
        raise RuntimeError(f"Depth directory not created: {depth_dir}")
    
    # 检查是否有输出文件
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith(('.png', '.npy', '.npz'))]
    if len(depth_files) == 0:
        raise RuntimeError(f"No depth files found in {depth_dir}. Generation may have failed.")
    
    CONSOLE.log(f"[green]Depth generation completed. {len(depth_files)} files saved to: {depth_dir}[/green]")


def gen_semantic(scene_dir: str, gpu_id: str = '0', model_name: str = 'shi-labs/oneformer_cityscapes_swin_large', cam_id: int = 0):
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
    instance_dir = os.path.join(semantic_dir, 'instance')
    
    # 如果目录已存在，删除它以重新生成
    if os.path.exists(semantic_dir):
        instance_files = []
        if os.path.exists(instance_dir):
            instance_files = [f for f in os.listdir(instance_dir) if f.endswith('.png')]
        if len(instance_files) > 0:
            CONSOLE.log(f"Semantic directory exists with {len(instance_files)} files. Removing to regenerate...")
        else:
            CONSOLE.log(f"Semantic directory exists. Removing to regenerate...")
        shutil.rmtree(semantic_dir)
    
    # 创建semantic目录（如果不存在，子脚本也会创建，但先创建更安全）
    os.makedirs(instance_dir, exist_ok=True)
    
    script_path = os.path.join(os.path.dirname(__file__), 'gen_semantic_oneformer.py')
    if not os.path.exists(script_path):
        raise ValueError(f"Semantic generation script not found: {script_path}")
    
    CONSOLE.log(f"Generating semantic segmentation for scene: {scene_dir}")
    
    # 图像目录应该是 scene_dir/images
    images_dir = os.path.join(scene_dir, 'images')
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # 构建命令
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    cmd = [
        sys.executable, script_path,
        '--input_dir', images_dir,  # 使用 images 目录作为输入
        '--output_dir', scene_dir,  # 输出到 scene_dir
        '--model_name', model_name,
        '--task', 'semantic',
        '--device', 'cuda',
        '--gpu_id', gpu_id,
        '--cam_id', str(cam_id)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.path.dirname(script_path),
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            CONSOLE.log(f"[red]Semantic generation command failed with exit code {result.returncode}[/red]")
            if error_msg:
                # 显示完整的错误信息（最多2000字符）
                error_display = error_msg[-2000:] if len(error_msg) > 2000 else error_msg
                CONSOLE.log(f"[red]Error output:[/red]")
                CONSOLE.log(f"[red]{error_display}[/red]")
            else:
                CONSOLE.log(f"[red]No error output captured. Check if transformers and other dependencies are installed.[/red]")
                CONSOLE.log(f"[yellow]Try running: pip install transformers torch pillow opencv-python-headless tqdm[/yellow]")
            raise RuntimeError(f"Semantic generation failed with exit code {result.returncode}")
        
        # 显示输出（如果有）
        if result.stdout:
            CONSOLE.log(f"[dim]{result.stdout[-500:]}[/dim]")  # 显示最后500字符
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Semantic generation timed out after 2 hours")
    except Exception as e:
        raise RuntimeError(f"Semantic generation error: {e}")
    
    # 验证输出目录是否存在
    instance_dir = os.path.join(semantic_dir, 'instance')
    if not os.path.exists(instance_dir):
        raise RuntimeError(f"Semantic instance directory not created: {instance_dir}")
    
    # 检查是否有输出文件
    instance_files = [f for f in os.listdir(instance_dir) if f.endswith('.png')]
    if len(instance_files) == 0:
        raise RuntimeError(f"No semantic segmentation files found in {instance_dir}")
    
    CONSOLE.log(f"[green]Semantic generation completed. {len(instance_files)} files saved to: {instance_dir}[/green]")


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
        raise ValueError(
            f"Semantic instance directory not found: {instance_path}.\n"
            f"Please run semantic generation first using --gen_semantic flag."
        )
    
    # 检查是否有实例文件
    instance_files = [f for f in os.listdir(instance_path) if f.endswith('.png')]
    if len(instance_files) == 0:
        raise ValueError(
            f"No semantic segmentation files found in {instance_path}.\n"
            f"Please run semantic generation first using --gen_semantic flag."
        )
    
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
    parser.add_argument('--cam_id', type=int, default=0, help='the camera id')
    
    args = parser.parse_args()
    
    # 验证场景目录
    if not os.path.exists(args.scene_dir):
        CONSOLE.log(f"[red]Error: Scene directory not found: {args.scene_dir}[/red]")
        sys.exit(1)
    
    images_dir = os.path.join(args.scene_dir, 'images')
    if not os.path.exists(images_dir):
        CONSOLE.log(f"[red]Error: Images directory not found: {images_dir}[/red]")
        sys.exit(1)
    
    # 执行生成任务（注意顺序：先语义分割，再天空mask）
    if args.gen_depth:
        try:
            gen_metric_depth(args.scene_dir, args.dataset, args.depth_gpu_id, args.cam_id)
        except Exception as e:
            CONSOLE.log(f"[red]Error generating depth: {e}[/red]")
            sys.exit(1)
    
    if args.gen_semantic:
        try:
            gen_semantic(args.scene_dir, args.semantic_gpu_id, args.model_name, args.cam_id)
        except Exception as e:
            CONSOLE.log(f"[red]Error generating semantic: {e}[/red]")
            sys.exit(1)
    
    if args.gen_sky_mask:
        try:
            # 检查语义分割是否已完成（不仅检查目录，还要检查是否有文件）
            instance_path = os.path.join(args.scene_dir, 'semantic', 'instance')
            instance_files = []
            if os.path.exists(instance_path):
                instance_files = [f for f in os.listdir(instance_path) if f.endswith('.png')]
            
            if not os.path.exists(instance_path) or len(instance_files) == 0:
                CONSOLE.log(f"[yellow]Warning: Semantic instance directory not found or empty: {instance_path}[/yellow]")
                CONSOLE.log(f"[yellow]Attempting to generate semantic segmentation first...[/yellow]")
                # 如果语义分割未完成，尝试生成
                if not args.gen_semantic:
                    try:
                        gen_semantic(args.scene_dir, args.semantic_gpu_id, args.model_name, args.cam_id)
                        # 再次检查
                        if os.path.exists(instance_path):
                            instance_files = [f for f in os.listdir(instance_path) if f.endswith('.png')]
                        if len(instance_files) == 0:
                            raise RuntimeError("Semantic generation completed but no files were created")
                    except Exception as e:
                        CONSOLE.log(f"[red]Error: Cannot generate sky mask without semantic segmentation: {e}[/red]")
                        sys.exit(1)
            gen_sky_mask(args.scene_dir)
        except Exception as e:
            CONSOLE.log(f"[red]Error generating sky mask: {e}[/red]")
            sys.exit(1)
    
    if not (args.gen_depth or args.gen_semantic or args.gen_sky_mask):
        CONSOLE.log("[yellow]Warning: No generation task specified. Use --gen_depth, --gen_semantic, or --gen_sky_mask[/yellow]")
    
    CONSOLE.log("[green]All tasks completed![/green]")


if __name__ == "__main__":
    main()

