#!/usr/bin/env python3
"""
测试 OneFormer 语义分割环境配置
"""
import os
import sys
from pathlib import Path

def test_imports():
    """测试必要的包导入"""
    print("1. 测试 Python 包导入...")
    try:
        import torch
        print(f"   ✓ PyTorch: {torch.__version__}")
        print(f"   ✓ CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ GPU 数量: {torch.cuda.device_count()}")
    except ImportError:
        print("   ✗ PyTorch 未安装")
        return False
    
    try:
        import transformers
        print(f"   ✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("   ✗ Transformers 未安装")
        return False
    
    try:
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        print("   ✓ OneFormer 模型类可导入")
    except ImportError as e:
        print(f"   ✗ OneFormer 导入失败: {e}")
        return False
    
    return True

def test_model_loading():
    """测试模型加载（不下载，只检查配置）"""
    print("\n2. 测试模型配置...")
    try:
        from transformers import AutoConfig
        model_name = os.getenv('ONEFORMER_MODEL_NAME', 'shi-labs/oneformer_cityscapes_swin_large')
        print(f"   模型名称: {model_name}")
        
        # 尝试加载配置（不需要下载模型）
        try:
            config = AutoConfig.from_pretrained(model_name)
            print(f"   ✓ 模型配置可访问")
            print(f"   ✓ 架构: {config.model_type}")
        except Exception as e:
            print(f"   ⚠ 无法访问模型配置（可能需要网络连接或首次下载）: {e}")
        
        return True
    except Exception as e:
        print(f"   ✗ 模型配置测试失败: {e}")
        return False

def test_script_exists():
    """测试脚本是否存在"""
    print("\n3. 测试脚本文件...")
    script_path = Path(__file__).parent / 'gen_semantic_oneformer.py'
    if script_path.exists():
        print(f"   ✓ 脚本存在: {script_path}")
        return True
    else:
        print(f"   ✗ 脚本不存在: {script_path}")
        return False

def main():
    print("=" * 60)
    print("测试 OneFormer 语义分割环境")
    print("=" * 60)
    
    results = []
    results.append(("包导入", test_imports()))
    results.append(("模型配置", test_model_loading()))
    results.append(("脚本文件", test_script_exists()))
    
    print("\n" + "=" * 60)
    print("测试结果:")
    print("=" * 60)
    
    all_ok = True
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"{status} {name}")
        if not ok:
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ 所有测试通过！可以运行语义分割")
        print("\n首次运行时，模型会自动从 Hugging Face 下载")
        print("模型大小约 1.5 GB，请确保有足够的磁盘空间和网络连接")
        return 0
    else:
        print("⚠ 部分测试未通过")
        print("\n请运行以下命令安装依赖:")
        print("  bash install_semantic_deps.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())

