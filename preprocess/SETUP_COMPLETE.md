# nuScenes 预处理环境设置完成说明

## ✅ 已完成的工作

### 1. 项目代码
- ✓ Metric3D 代码已存在于 `metric3d/` 目录
- ✓ NVI_SEM 代码已存在于 `nvi_sem/` 目录
- ✓ nuScenes 读取器和点云生成器已实现

### 2. 环境变量配置
- ✓ `launch.json` 已更新，使用正确的路径：
  - `METRIC3D_PATH`: `${workspaceFolder}/third_party/EVolSplat/preprocess/metric3d`
  - `METRIC3D_MODEL_PATH`: `${workspaceFolder}/third_party/EVolSplat/preprocess/metric3d/models/metric_depth_vit_giant2_800k.pth`
  - `NVI_SEM_PATH`: `${workspaceFolder}/third_party/EVolSplat/preprocess/nvi_sem`
  - `NVI_SEM_CHECKPOINT`: `${workspaceFolder}/third_party/EVolSplat/preprocess/nvi_sem/checkpoints/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth`

### 3. 工具脚本
- ✓ `download_models.sh`: 自动下载模型权重
- ✓ `verify_models.py`: 验证模型文件
- ✓ `test_setup.py`: 测试环境配置
- ✓ `check_nuscenes_setup.py`: 检查数据集和环境

## ⏳ 进行中的工作

### 模型下载
模型权重正在后台下载中：

1. **Metric3D 模型** (~813 MB)
   - 位置: `metric3d/models/metric_depth_vit_giant2_800k.pth`
   - 状态: 下载中

2. **NVI_SEM 模型**
   - 位置: `nvi_sem/checkpoints/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth`
   - 状态: 下载中（需要解压 zip 文件）

## 📋 后续步骤

### 1. 检查下载状态

运行以下命令检查模型是否下载完成：

```bash
cd /root/drivestudio-coding/third_party/EVolSplat/preprocess
python3 verify_models.py
```

或者：

```bash
python3 test_setup.py
```

### 2. 如果下载未完成

如果模型下载中断，可以手动运行：

```bash
bash download_models.sh
```

或者手动下载：

**Metric3D 模型:**
```bash
cd /root/drivestudio-coding/third_party/EVolSplat/preprocess/metric3d/models
gdown 1KVINiBkVpJylx_6z1lAC7CQ4kmn-RJRN -O metric_depth_vit_giant2_800k.pth
```

**NVI_SEM 模型:**
```bash
cd /root/drivestudio-coding/third_party/EVolSplat/preprocess/nvi_sem/checkpoints
gdown 1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U -O nvi_sem_weights.zip
unzip nvi_sem_weights.zip
# 找到并重命名权重文件
find . -name "*cityscapes*ocrnet*HRNet*Mscale*.pth" -exec mv {} cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth \;
```

### 3. 验证环境

运行完整测试：

```bash
python3 test_setup.py
```

预期输出应该显示所有测试通过（✓）。

### 4. 测试 Raw Data Preprocess

在 VSCode 中：

1. 打开 "Run and Debug" 面板
2. 选择 "Check NuScenes Setup" 配置
3. 运行以检查数据集状态

如果预处理数据不存在，运行：
- "Step 1: NuScenes Raw Data Preprocess (Mini Demo)"

然后运行：
- "Step 2: NuScenes EVolSplat Preprocess (Demo - Small Sample)"

## 🔍 故障排除

### 问题: 模型下载失败

**解决方案:**
1. 检查网络连接
2. 尝试使用 VPN 或代理
3. 手动从 Google Drive 下载：
   - Metric3D: https://drive.google.com/file/d/1KVINiBkVpJylx_6z1lAC7CQ4kmn-RJRN/view?usp=drive_link
   - NVI_SEM: https://drive.google.com/file/d/1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U/view?usp=sharing

### 问题: 找不到预处理数据

**解决方案:**
1. 确认数据集路径: `/mnt/f/DataSet/nuScenes`
2. 运行 "Step 1" 配置预处理原始数据
3. 预处理后的数据将保存在: `/mnt/f/DataSet/nuScenes/processed_10Hz/mini/`

### 问题: GPU 内存不足

**解决方案:**
1. 修改 `launch.json` 中的 `DEPTH_GPU_ID` 和 `SEMANTIC_GPU_ID`
2. 减少 `--num_images` 参数值
3. 不使用深度和语义生成（去掉相关 flags）

## 📝 环境变量参考

在 VSCode launch.json 中已配置，如果需要手动设置：

```bash
export METRIC3D_PATH="/root/drivestudio-coding/third_party/EVolSplat/preprocess/metric3d"
export METRIC3D_MODEL_PATH="/root/drivestudio-coding/third_party/EVolSplat/preprocess/metric3d/models/metric_depth_vit_giant2_800k.pth"
export NVI_SEM_PATH="/root/drivestudio-coding/third_party/EVolSplat/preprocess/nvi_sem"
export NVI_SEM_CHECKPOINT="/root/drivestudio-coding/third_party/EVolSplat/preprocess/nvi_sem/checkpoints/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth"
export DEPTH_GPU_ID="0"
export SEMANTIC_GPU_ID="0"
```

## ✨ 测试成功标志

当所有配置正确时，运行 `test_setup.py` 应该看到：

```
============================================================
测试结果总结:
============================================================
✓ Python 包
✓ Metric3D
✓ NVI_SEM
✓ NuScenes 读取器
✓ 点云生成器

✓ 所有测试通过！可以运行预处理
```



