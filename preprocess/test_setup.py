#!/usr/bin/env python3
"""
测试预处理环境是否配置正确
"""
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

def test_imports():
    """测试必要的 Python 包导入"""
    print("1. 测试 Python 包导入...")
    try:
        import numpy
        import cv2
        import PIL
        import rich
        import imageio
        import tqdm
        import open3d
        print("   ✓ 基础包导入成功")
    except ImportError as e:
        print(f"   ✗ 导入失败: {e}")
        return False
    return True

def test_metric3d():
    """测试 Metric3D 路径和脚本"""
    print("\n2. 测试 Metric3D...")
    metric3d_path = SCRIPT_DIR / "metric3d"
    test_script = metric3d_path / "mono" / "tools" / "test_scale_cano.py"
    config_file = metric3d_path / "mono" / "configs" / "HourglassDecoder" / "vit.raft5.giant2.py"
    model_path = metric3d_path / "models" / "metric_depth_vit_giant2_800k.pth"
    
    if not test_script.exists():
        print(f"   ✗ 测试脚本不存在: {test_script}")
        return False
    if not config_file.exists():
        print(f"   ✗ 配置文件不存在: {config_file}")
        return False
    if not model_path.exists():
        print(f"   ⚠ 模型文件不存在: {model_path}")
        print(f"     请运行: bash download_models.sh")
        return False
    
    print(f"   ✓ Metric3D 路径: {metric3d_path}")
    print(f"   ✓ 测试脚本: {test_script.name}")
    print(f"   ✓ 配置文件: {config_file.name}")
    print(f"   ✓ 模型文件: {model_path.name} ({model_path.stat().st_size / (1024*1024):.1f} MB)")
    
    # 检查环境变量
    metric3d_env = os.getenv('METRIC3D_PATH', '')
    if metric3d_env:
        print(f"   ✓ METRIC3D_PATH 环境变量: {metric3d_env}")
    else:
        print(f"   ⚠ METRIC3D_PATH 环境变量未设置，将使用默认路径")
    
    return True

def test_nvi_sem():
    """测试 NVI_SEM 路径和脚本"""
    print("\n3. 测试 NVI_SEM...")
    nvi_sem_path = SCRIPT_DIR / "nvi_sem"
    train_script = nvi_sem_path / "train.py"
    model_path = nvi_sem_path / "checkpoints" / "cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth"
    
    if not train_script.exists():
        print(f"   ✗ 训练脚本不存在: {train_script}")
        return False
    if not model_path.exists():
        print(f"   ⚠ 模型文件不存在: {model_path}")
        print(f"     请运行: bash download_models.sh")
        return False
    
    print(f"   ✓ NVI_SEM 路径: {nvi_sem_path}")
    print(f"   ✓ 训练脚本: {train_script.name}")
    print(f"   ✓ 模型文件: {model_path.name} ({model_path.stat().st_size / (1024*1024):.1f} MB)")
    
    # 检查环境变量
    nvi_sem_env = os.getenv('NVI_SEM_PATH', '')
    if nvi_sem_env:
        print(f"   ✓ NVI_SEM_PATH 环境变量: {nvi_sem_env}")
    else:
        print(f"   ⚠ NVI_SEM_PATH 环境变量未设置，将使用默认路径")
    
    return True

def test_nuscenes_reader():
    """测试 nuScenes 读取器"""
    print("\n4. 测试 nuScenes 读取器...")
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from read_dataset.read_nuscenes import ReadNuScenesData
        print("   ✓ ReadNuScenesData 导入成功")
        return True
    except Exception as e:
        print(f"   ✗ 导入失败: {e}")
        return False

def test_pcd_generator():
    """测试点云生成器"""
    print("\n5. 测试点云生成器...")
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from read_dataset.generate_nuscenes_pcd import NuScenesPCDGenerator
        print("   ✓ NuScenesPCDGenerator 导入成功")
        return True
    except Exception as e:
        print(f"   ✗ 导入失败: {e}")
        return False

def main():
    print("=" * 60)
    print("测试预处理环境配置")
    print("=" * 60)
    
    results = []
    results.append(("Python 包", test_imports()))
    results.append(("Metric3D", test_metric3d()))
    results.append(("NVI_SEM", test_nvi_sem()))
    results.append(("NuScenes 读取器", test_nuscenes_reader()))
    results.append(("点云生成器", test_pcd_generator()))
    
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    
    all_ok = True
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"{status} {name}")
        if not ok:
            all_ok = False
    
    print("\n环境变量设置 (用于 launch.json):")
    print("-" * 60)
    print(f'METRIC3D_PATH="{SCRIPT_DIR}/metric3d"')
    print(f'METRIC3D_MODEL_PATH="{SCRIPT_DIR}/metric3d/models/metric_depth_vit_giant2_800k.pth"')
    print(f'NVI_SEM_PATH="{SCRIPT_DIR}/nvi_sem"')
    print(f'NVI_SEM_CHECKPOINT="{SCRIPT_DIR}/nvi_sem/checkpoints/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth"')
    
    if all_ok:
        print("\n✓ 所有测试通过！可以运行预处理")
        return 0
    else:
        print("\n⚠ 部分测试未通过，请检查上述问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())



