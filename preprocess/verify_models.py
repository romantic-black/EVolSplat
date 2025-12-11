#!/usr/bin/env python3
"""
验证 Metric3D 和 NVI_SEM 模型文件是否存在且可用
"""
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()

def verify_metric3d():
    """验证 Metric3D 模型"""
    metric3d_path = SCRIPT_DIR / "metric3d"
    model_path = SCRIPT_DIR / "metric3d" / "models" / "metric_depth_vit_giant2_800k.pth"
    test_script = metric3d_path / "mono" / "tools" / "test_scale_cano.py"
    config_file = metric3d_path / "mono" / "configs" / "HourglassDecoder" / "vit.raft5.giant2.py"
    
    print("检查 Metric3D:")
    print(f"  路径: {metric3d_path}")
    print(f"  脚本: {test_script.exists()}")
    print(f"  配置: {config_file.exists()}")
    print(f"  模型: {model_path.exists()}", end="")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f" ({size_mb:.1f} MB)")
        
        # 检查是否有部分下载的文件
        part_files = list(metric3d_path.glob("**/*.part"))
        if part_files:
            print(f"  警告: 发现部分下载文件: {part_files}")
            return False
        return True
    else:
        # 检查是否有部分下载的文件
        part_files = list((SCRIPT_DIR / "metric3d" / "models").glob("*.part"))
        if part_files:
            print(f"  (发现部分下载: {part_files[0].name}, 大小: {part_files[0].stat().st_size / (1024*1024):.1f} MB)")
        else:
            print(" (不存在)")
        return False

def verify_nvi_sem():
    """验证 NVI_SEM 模型"""
    nvi_sem_path = SCRIPT_DIR / "nvi_sem"
    model_path = nvi_sem_path / "checkpoints" / "cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth"
    train_script = nvi_sem_path / "train.py"
    
    print("\n检查 NVI_SEM:")
    print(f"  路径: {nvi_sem_path}")
    print(f"  脚本: {train_script.exists()}")
    print(f"  模型: {model_path.exists()}", end="")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f" ({size_mb:.1f} MB)")
        return True
    else:
        # 检查是否有 zip 文件需要解压
        zip_files = list((SCRIPT_DIR / "nvi_sem" / "checkpoints").glob("*.zip"))
        if zip_files:
            print(f"  (发现 zip 文件: {zip_files[0].name})")
        else:
            print(" (不存在)")
        return False

def main():
    print("=" * 60)
    print("验证模型文件")
    print("=" * 60)
    
    metric3d_ok = verify_metric3d()
    nvi_sem_ok = verify_nvi_sem()
    
    print("\n" + "=" * 60)
    print("环境变量设置建议:")
    print("=" * 60)
    print(f'export METRIC3D_PATH="{SCRIPT_DIR}/metric3d"')
    print(f'export METRIC3D_MODEL_PATH="{SCRIPT_DIR}/metric3d/models/metric_depth_vit_giant2_800k.pth"')
    print(f'export NVI_SEM_PATH="{SCRIPT_DIR}/nvi_sem"')
    print(f'export NVI_SEM_CHECKPOINT="{SCRIPT_DIR}/nvi_sem/checkpoints/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth"')
    
    print("\n" + "=" * 60)
    if metric3d_ok and nvi_sem_ok:
        print("✓ 所有模型文件已就绪")
        return 0
    else:
        print("⚠ 部分模型文件缺失")
        if not metric3d_ok:
            print("  - Metric3D 模型需要下载")
        if not nvi_sem_ok:
            print("  - NVI_SEM 模型需要下载")
        print("\n运行下载脚本: bash download_models.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())



