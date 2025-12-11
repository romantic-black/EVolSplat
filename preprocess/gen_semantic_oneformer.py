#!/usr/bin/env python3
"""
使用 OneFormer (Hugging Face Transformers) 生成语义分割掩码
用于替换 nvi_sem，提取天空和动态物体
"""
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

# Cityscapes 类别定义
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

# 需要提取的类别 ID
SKY_CLASS_ID = 10  # sky
DYNAMIC_CLASS_IDS = {
    'Vehicle': [13, 14, 15],   # car, truck, bus
    'human': [11, 12, 17, 18], # person, rider, motorcycle, bicycle
}


def process_images(input_dir, output_dir, model_name="shi-labs/oneformer_cityscapes_swin_large", task="semantic", device="cuda"):
    """
    处理图像目录，生成语义分割掩码
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录（将创建 semantic/instance/ 子目录）
        model_name: Hugging Face 模型名称
        task: 任务类型 ("semantic", "instance", "panoptic")
        device: 设备 ("cuda" 或 "cpu")
    """
    # 加载模型和处理器
    print(f"加载 OneFormer 模型: {model_name}")
    processor = OneFormerProcessor.from_pretrained(model_name)
    model = OneFormerForUniversalSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    
    # 创建输出目录
    instance_dir = os.path.join(output_dir, 'semantic', 'instance')
    os.makedirs(instance_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
    image_files = sorted(image_files)
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 处理每张图像
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="处理图像"):
            # 读取图像
            image = Image.open(img_path).convert("RGB")
            
            # 预处理
            inputs = processor(images=image, task_inputs=[task], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 推理
            outputs = model(**inputs)
            
            # 后处理
            if task == "semantic":
                # 语义分割：每个像素是类别ID
                predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                semantic_map = predicted_map.cpu().numpy().astype(np.uint8)
            elif task == "panoptic":
                # 全景分割：需要提取语义ID
                predicted_map = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                # 从全景分割结果中提取语义ID
                # predicted_map 是一个字典，包含 'segmentation' 和 'segments_info'
                # segments_info 包含每个segment的类别信息
                panoptic_seg = predicted_map['segmentation'].cpu().numpy()
                segments_info = predicted_map['segments_info']
                
                # 创建语义映射
                semantic_map = np.zeros_like(panoptic_seg, dtype=np.uint8)
                for segment in segments_info:
                    segment_id = segment['id']
                    label_id = segment['label_id']
                    semantic_map[panoptic_seg == segment_id] = label_id
            else:
                raise ValueError(f"不支持的任务类型: {task}")
            
            # 保存语义分割结果
            output_filename = os.path.splitext(os.path.basename(img_path))[0] + '.png'
            output_path = os.path.join(instance_dir, output_filename)
            cv2.imwrite(output_path, semantic_map)
    
    print(f"语义分割结果已保存到: {instance_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="使用 OneFormer 生成语义分割掩码")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model_name", type=str, default="shi-labs/oneformer_cityscapes_swin_large",
                       help="Hugging Face 模型名称")
    parser.add_argument("--task", type=str, default="semantic", choices=["semantic", "panoptic"],
                       help="任务类型: semantic 或 panoptic")
    parser.add_argument("--device", type=str, default="cuda", help="设备: cuda 或 cpu")
    parser.add_argument("--gpu_id", type=str, default=None, help="GPU ID (如果指定，将设置 CUDA_VISIBLE_DEVICES)")
    
    args = parser.parse_args()
    
    # 设置 GPU
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = "cpu"
    
    process_images(args.input_dir, args.output_dir, args.model_name, args.task, args.device)


if __name__ == "__main__":
    main()

