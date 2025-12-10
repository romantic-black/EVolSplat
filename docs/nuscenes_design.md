# NuScenes 数据接入与 EVolSplat/DriveStudio 设计草案

> 目标：同时复用 `nerfstudio` 下的 EVolSplat 代码和 `third_party/drivestudio` 的节点式 3DGS 渲染，在 **不再依赖 preprocess 中转目录** 的前提下，为 NuScenes 提供新的 dataloader，并明确训练/评估代码和项目组织方式。

## 1. NuScenes 直接加载方案
- **设计输出形态**：直接产出与 `third_party/drivestudio/datasets/nuscenes/nuscenes_sourceloader.py` 一致的 `ScenePixelSource` / `SceneLidarSource` 对象，而不是落盘中转；同时提供一个轻量 `DataparserOutputs`，便于走 nerfstudio 的 `SplatDatamanager`。
- **关键读取逻辑（复用 sourceloader 行为）**
  - 坐标系：沿用 `OPENCV2DATASET` 变换，并用首帧 `CAM_FRONT` 对齐世界系（参考 `NuScenesCameraData.load_calibrations` / `NuScenesLiDARSource.load_calibrations`）。
  - 相机：从 NuScenes 原始 `calibrated_sensor` 读取 intrinsics / distortions，按 `pixel_source.downscale_when_loading` 与 `downscale` 重新缩放，保持 6-cam 顺序与 `AVAILABLE_CAM_LIST` 对齐。
  - 激光雷达：直接用 `nuscenes-devkit` 读取 `LIDAR_TOP`，在内存中完成位姿对齐（无需写入 `lidar/*.bin` 临时文件），并返回 `origins/viewdirs/ranges/timestamps`。
  - 实例/节点：利用标注 cuboid 构造 `instances_pose/instances_size/per_frame_instance_mask/instances_model_types`，类别映射沿用 `OBJECT_CLASS_NODE_MAPPING`，并保留可选 `SMPL` 加载逻辑。
  - 遮罩：可选择实时生成（语义分割 + box 投影）或缓存到 `data_root/cache/...`，避免引入新的 preprocess 路径。
- **配置对齐**：沿用 `third_party/drivestudio/configs/datasets/nuscenes/6cams.yaml` 的字段，新增/复用字段：
  - `data_root`: 指向 NuScenes 原始数据根目录。
  - `process_mode: raw`：标记走“直接读”而非预处理产物。
  - `cache_dir`（可选）：存放一次性的掩码 / SMPL / box cache，默认同 `data_root/cache`。
- **接口落位**
  - 新增 `nerfstudio/data/datasets/nuscenes_direct_loader.py`（命名示例）：实现 `NuScenesDirectPixelSource` / `NuScenesDirectLiDARSource`，复用 drivestudio 的基类与对齐逻辑。
  - 新增 `nerfstudio/data/dataparsers/nuscenes_driving_dataparser.py`：把 `ScenePixelSource` 中的多相机位姿汇总为 nerfstudio `Cameras`，`scene_box` 直接复用 `pixel_source.get_aabb()`。
  - 保持 `third_party/drivestudio` 代码原样，仅通过 wrapper 调用其类，避免双份实现。

## 2. 模型与渲染复用策略
- **继续使用的 EVolSplat 组件（nerfstudio）**
  - 特征/体积侧：`model_components/sparse_conv.py`，`model_components/projection.py`，`fields/initial_BgSphere.py` 及 `EvolSplatModel` 中的 densify / offset / scale 预测逻辑。
  - 训练基类/指标：`Model` 基类的优化、`get_image_metrics_and_images`、`SSIM/PSNR/LPIPS` 计算。
- **切换为 drivestudio 的渲染骨架**
  - 使用 `models/trainers/scene_graph.py:MultiTrainer` 的节点式渲染入口，启用 `Background` + `RigidNodes`（必要时 `DeformableNodes`/`SMPLNodes`）。
  - 高斯容器与混合：复用 drivestudio `models/nodes/*` 和 `render_gaussians`，保证背景与刚体节点的独立可视化与混合逻辑。
- **衔接方式（建议）**
  - 在 nerfstudio 侧新增一个 `DrivingSplatModel`（命名示例），内部组合 EVolSplat 的特征编码与 drivestudio 的高斯节点：EVolSplat 负责从多源图像提取体素/偏移/尺度/颜色初值；drivestudio 节点负责存储、高斯参数优化与渲染。
  - 渲染前把 EVolSplat 预测的参数写入 drivestudio 节点（背景 + 刚体），再调用 drivestudio 的 `render_gaussians` 获得 `rgb/opacity/depth`。
  - 这样保留 EVolSplat 的学习表征与 densify 逻辑，同时利用 drivestudio 的节点分层与多组件混合。

## 3. 训练与评估的使用建议
- **训练外壳**：继续用 nerfstudio 的 `TrainerConfig`/`Pipeline`，因为现有 CLI、日志、checkpoint 都已接好；在 Pipeline 内部调度 `DrivingSplatModel`，而模型内部再持有 drivestudio 的 `MultiTrainer` 或等价渲染模块。
- **评估**：
  - 直接复用 nerfstudio 的指标与可视化（PSNR/SSIM/LPIPS），保证与现有 EVolSplat 结果可比。
  - 如需节点级可视化/视频，调用 `third_party/drivestudio/models/video_utils.py` 和 `tools/eval.py` 生成分层输出（Background/RigidNodes 等）。
- **可选基线**：保留 drivestudio 原生 `tools/train.py` 跑通一条“纯 drivestudio”链路，对比混合方案的收益；两套脚本共享同一份 dataloader，减少维护成本。

## 4. 项目目录组织建议
- `nerfstudio/data/datasets/nuscenes_direct_loader.py`：NuScenes 直接加载的 ScenePixelSource/LiDARSource 封装。
- `nerfstudio/data/dataparsers/nuscenes_driving_dataparser.py`：把 drivestudio 数据源转成 nerfstudio `DataparserOutputs`。
- `nerfstudio/models/driving_splat_model.py`：EVolSplat 特征 + drivestudio 渲染的组合模型（调用 `EvolSplatModel` 部分模块与 drivestudio 节点）。
- `config/datasets/nuscenes_direct.yaml`：继承 `third_party/drivestudio/configs/datasets/nuscenes/6cams.yaml`，仅补充 `data_root/process_mode/cache_dir` 等字段。
- `config/methods/evolsplat_drivestudio.yaml`：新的 method preset，指向上述 dataparser/model/pipeline。
- 文档归档：保留本文档于 `docs/nuscenes_design.md`，后续实现细节可按模块在 docs 下补充短文档。

## 5. 后续落地优先级
1. 抽取 `NuScenesPixelSource/NuScenesLiDARSource` 的核心逻辑，完成“直接加载”版实现，并跑通最小样例（mini split）。
2. 编写 dataparser + pipeline skeleton，把驱动链路串到 nerfstudio 的训练循环。
3. 将 EVolSplat 的 densify/offset/scale 写入 drivestudio 节点，完成第一次渲染闭环。
4. 增加分层评估/可视化，确认 Background + RigidNodes 的渲染输出。
5. 清理配置与 CLI 接口，补充 README/usage 片段。
