#!/usr/bin/env python3
"""
检查 nuScenes 预处理所需的依赖和文件是否齐全
"""
import os
import sys
from pathlib import Path

def check_python_packages():
    """检查 Python 包是否安装"""
    required_packages = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'rich': 'rich',
        'imageio': 'imageio',
        'tqdm': 'tqdm',
        'open3d': 'open3d',
        'nuscenes': 'nuscenes-devkit',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"✗ {package_name} NOT installed")
            missing.append(package_name)
    
    return missing

def check_dataset_structure(data_root):
    """检查数据集目录结构"""
    print(f"\n检查数据集目录: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"✗ 数据集根目录不存在: {data_root}")
        return False
    
    # 检查原始数据
    raw_paths = [
        os.path.join(data_root, 'v1.0-mini'),
        os.path.join(data_root, 'samples'),
        os.path.join(data_root, 'sweeps'),
    ]
    
    print("\n原始数据检查:")
    raw_ok = True
    for path in raw_paths:
        if os.path.exists(path):
            print(f"✓ {path} 存在")
        else:
            print(f"✗ {path} 不存在")
            raw_ok = False
    
    # 检查预处理数据
    processed_paths = [
        os.path.join(data_root, 'processed_10Hz', 'mini', '000'),
        os.path.join(data_root, 'processed', 'mini', '000'),
    ]
    
    print("\n预处理数据检查:")
    processed_ok = False
    processed_base = None
    
    for base_path in [
        os.path.join(data_root, 'processed_10Hz'),
        os.path.join(data_root, 'processed'),
    ]:
        if os.path.exists(base_path):
            print(f"✓ 找到预处理目录: {base_path}")
            processed_base = base_path
            # 检查第一个场景
            scene_0 = os.path.join(base_path, 'mini', '000')
            if os.path.exists(scene_0):
                print(f"✓ 场景 000 存在")
                # 检查必需的文件
                required_dirs = ['images', 'extrinsics', 'intrinsics']
                all_ok = True
                for subdir in required_dirs:
                    subdir_path = os.path.join(scene_0, subdir)
                    if os.path.exists(subdir_path):
                        file_count = len(os.listdir(subdir_path))
                        print(f"  ✓ {subdir}/ ({file_count} files)")
                    else:
                        print(f"  ✗ {subdir}/ 不存在")
                        all_ok = False
                
                if all_ok:
                    processed_ok = True
            else:
                print(f"✗ 场景 000 不存在")
                # 列出可用的场景
                mini_dir = os.path.join(base_path, 'mini')
                if os.path.exists(mini_dir):
                    scenes = sorted([d for d in os.listdir(mini_dir) 
                                   if os.path.isdir(os.path.join(mini_dir, d)) and d.isdigit()])
                    if scenes:
                        print(f"  可用场景: {scenes[:5]}...")
            break
    
    if not processed_ok:
        print("\n⚠ 预处理数据不存在，需要先运行预处理步骤")
        print("  运行配置: 'Step 1: NuScenes Raw Data Preprocess (Mini Demo)'")
    
    return processed_ok, processed_base

def check_external_tools():
    """检查外部工具路径"""
    print("\n检查外部工具:")
    
    tools = {
        'METRIC3D_PATH': os.getenv('METRIC3D_PATH', ''),
        'METRIC3D_MODEL_PATH': os.getenv('METRIC3D_MODEL_PATH', ''),
        'NVI_SEM_PATH': os.getenv('NVI_SEM_PATH', ''),
        'NVI_SEM_CHECKPOINT': os.getenv('NVI_SEM_CHECKPOINT', ''),
    }
    
    all_ok = True
    for tool_name, tool_path in tools.items():
        if tool_path and os.path.exists(tool_path):
            print(f"✓ {tool_name}: {tool_path}")
        elif tool_path:
            print(f"✗ {tool_name}: {tool_path} (路径不存在)")
            all_ok = False
        else:
            print(f"⚠ {tool_name}: 未设置环境变量")
            all_ok = False
    
    return all_ok

def main():
    """主函数"""
    print("=" * 60)
    print("nuScenes EVolSplat 预处理环境检查")
    print("=" * 60)
    
    # 检查 Python 包
    print("\n1. 检查 Python 依赖包:")
    missing_packages = check_python_packages()
    
    # 检查数据集
    data_root = "/mnt/f/DataSet/nuScenes"
    processed_ok, processed_base = check_dataset_structure(data_root)
    
    # 检查外部工具
    tools_ok = check_external_tools()
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结:")
    print("=" * 60)
    
    if missing_packages:
        print(f"\n缺失的 Python 包: {', '.join(missing_packages)}")
        print("安装命令: pip install " + " ".join(missing_packages))
    
    if not processed_ok:
        print("\n⚠ 预处理数据不存在或不全")
        print("  请先运行 'Step 1: NuScenes Raw Data Preprocess (Mini Demo)' 配置")
        if processed_base:
            print(f"  预处理数据应位于: {processed_base}/mini/000/")
    
    if not tools_ok:
        print("\n⚠ 外部工具路径未设置或不正确")
        print("  如果不需要深度估计和语义分割，可以跳过")
        print("  否则请设置以下环境变量:")
        print("    - METRIC3D_PATH")
        print("    - METRIC3D_MODEL_PATH")
        print("    - NVI_SEM_PATH")
        print("    - NVI_SEM_CHECKPOINT")
    
    if not missing_packages and processed_ok:
        print("\n✓ 所有检查通过！可以运行 EVolSplat 预处理")
        print("  使用配置: 'Step 2: NuScenes EVolSplat Preprocess (Demo - Small Sample)'")
    else:
        print("\n⚠ 请先解决上述问题后再运行预处理")

if __name__ == "__main__":
    main()

