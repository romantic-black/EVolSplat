# EVolSplat 在 nerfstudio 框架中的集成与改动

本文档详细说明项目原作者如何基于 nerfstudio 框架进行改动，实现 EVolSplat 模型。

## 目录

1. [概述](#概述)
2. [核心改动架构](#核心改动架构)
3. [模型层改动](#模型层改动)
4. [数据层改动](#数据层改动)
5. [组件层改动](#组件层改动)
6. [配置系统集成](#配置系统集成)
7. [训练流程集成](#训练流程集成)
8. [关键设计决策](#关键设计决策)

---

## 概述

EVolSplat 基于 nerfstudio 框架构建，采用了模块化的集成方式。主要改动包括：

- **新增模型**：实现 `EvolSplatModel`，继承自 nerfstudio 的 `Model` 基类
- **新增数据解析器**：实现零样本数据解析器，支持 KITTI-360 和 Waymo 数据集
- **新增模型组件**：稀疏卷积网络、2D特征投影器等
- **配置系统扩展**：在 nerfstudio 配置系统中注册新方法
- **Pipeline 集成**：通过 VanillaPipeline 传递点云数据

---

## 核心改动架构

```
nerfstudio 框架
    │
    ├── 模型层 (models/)
    │   └── evolsplat.py          [新增] EvolSplatModel
    │
    ├── 数据层 (data/dataparsers/)
    │   └── zeroshot_dataparser.py  [新增] 零样本数据解析器
    │   └── evolsplat_dataparser.py [新增] 训练数据解析器
    │
    ├── 组件层 (model_components/)
    │   ├── sparse_conv.py        [新增] 稀疏卷积网络
    │   └── projection.py         [新增] 2D特征投影器
    │
    ├── 场组件 (fields/)
    │   └── initial_BgSphere.py   [新增] 背景球初始化
    │
    ├── 配置系统 (configs/)
    │   └── method_configs.py     [修改] 注册 evolsplat 方法
    │   └── dataparser_configs.py [修改] 注册数据解析器
    │
    └── 脚本 (scripts/)
        └── infer_zeroshot.py     [新增] 零样本推理脚本
```

---

## 模型层改动

### 1. EvolSplatModel 实现

**文件位置**：`nerfstudio/models/evolsplat.py`

#### 1.1 继承关系

```python
class EvolSplatModel(Model):
    """继承自 nerfstudio.models.base_model.Model"""
```

EVolSplat 通过继承 nerfstudio 的 `Model` 基类，自动获得：
- 训练/评估流程集成
- 检查点保存/加载机制
- 指标计算框架
- 优化器管理

#### 1.2 模型配置类

```130:181:nerfstudio/models/evolsplat.py
@dataclass
class EvolSplatModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""
    _target: Type = field(default_factory=lambda: EvolSplatModel)
    validate_every: int = 8000
    """period of steps where gaussians are culled and densified"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    enabale_appearance_embedding: bool = False
    """whether enable the appearance embedding"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    entropy_loss: float = 0.1
    """weight of Entropy loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 1
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    """Config of the camera optimizer to use"""
    freeze_volume: bool = False
```

**关键设计**：
- 使用 `_target` 字段指定模型类，这是 nerfstudio 的配置系统要求
- 继承 `ModelConfig`，自动获得所有基础配置选项
- 添加 EVolSplat 特有的配置参数（如 `freeze_volume`、`ssim_lambda` 等）

#### 1.3 模型初始化

```193:201:nerfstudio/models/evolsplat.py
    def __init__(
        self,
        *args,
        seed_points: List,
        **kwargs,
    ):
        self.seed_points = seed_points
        self.num_scenes = len(seed_points) # type: ignore
        super().__init__(*args, **kwargs)
```

**改动说明**：
- 接收 `seed_points` 参数（3D点云数据），这是从 Pipeline 传递过来的
- 支持多场景训练（`num_scenes`）
- 调用 `super().__init__()` 完成基类初始化

#### 1.4 模块初始化

```203:329:nerfstudio/models/evolsplat.py
    def populate_modules(self, opts=None):

        ## Important: input the 3D point of the scene. All scenes data should be stroed as List as the unsame point number
        self.means = [] # type: ignore
        self.anchor_feats = []
        self.scales = [] # type: ignore
        self.offset = [] # type: ignore
        if self.seed_points is not None:
            for i in tqdm(range(self.num_scenes)):
                means = self.seed_points[i]['points3D_xyz']
                anchors_feat =   self.seed_points[i]['points3D_rgb'] / 255
                offsets = torch.zeros_like(means)
                distances, _ = self.k_nearest_sklearn(means.data, 3)
                distances = torch.from_numpy(distances)
                avg_dist = distances.mean(dim=-1, keepdim=True)
                scales = torch.log(avg_dist.repeat(1, 3))
                ## stack the parameters into list
                self.means.append(means)
                self.anchor_feats.append(anchors_feat)
                self.scales.append(scales)
                self.offset.append(offsets)
       
       

        ## load mannul param:
        assert opts is not None 
        self.local_radius = getattr(opts.model, 'local_radius', 1)
        self.sparseConv_outdim = opts.model.sparseConv_outdim
        self.offset_max = opts.model.offset_max 
        self.num_neibours = opts.model.num_neighbour_select 
        self.bbx_min = torch.tensor(opts.Boundingbox_min).float()
        self.bbx_max = torch.tensor(opts.Boundingbox_max).float()
        
        
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        ## config the projecter
        self.projector = Projector()

         ## construct the sparse tensor
        self.sparse_conv = SparseCostRegNet(d_in=3, d_out=self.sparseConv_outdim).cuda()
        self.voxel_size = opts.encoder.voxel_size
        
        self.feature_dim_out = 3*num_sh_bases(self.config.sh_degree)

        self.feature_dim_in = 4*self.num_neibours*(2*self.local_radius+1)**2
       
        if self.config.enabale_appearance_embedding:
            self.embedding_appearance = Embedding(self.num_train_data, self.appearance_embedding_dim)
        else:
            self.embedding_appearance = None
        

        ## gaussian appearance MLP, predict the SH coefficients
        self.gaussion_decoder = MLP(
                in_dim= self.feature_dim_in+4,
                num_layers=3,
                layer_width=128,
                out_dim=self.feature_dim_out,
                activation=nn.ReLU(),
                out_activation=None,
                implementation="torch",
            )
        
        self.mlp_conv = MLP(
                in_dim= self.sparseConv_outdim+4,
                num_layers=2,
                layer_width=64,
                out_dim=3+4,
                activation=nn.Tanh(),
                out_activation=None,
                implementation="torch",
            )
        
        self.mlp_opacity = MLP(
                in_dim=self.sparseConv_outdim+4,
                num_layers=2,
                layer_width=64,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation="torch",
            )
        
        self.mlp_offset = MLP(
                in_dim=self.sparseConv_outdim,
                num_layers=2,
                layer_width=64,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Tanh(),
                implementation="torch",
            )
        
      
        ## Background Model for sky & distant view
        if self.config.enable_collider:
            Res = getattr(opts.bg_model,"res", 700)
            Radius = getattr(opts.bg_model,"radius", 25)
            self.scene_center = np.array(getattr(opts.bg_model,"center", [0,3.8,5.6]))
            gs_sky_initlial = GaussianBGInitializer(resolution=Res, radius=Radius,center=self.scene_center)
            bg_pnt = gs_sky_initlial.build_model()
            bg_distances, _ = self.k_nearest_sklearn(torch.from_numpy(bg_pnt), 3)
            bg_distances = torch.from_numpy(bg_distances)
            avg_dist = bg_distances.mean(dim=-1, keepdim=True)
            self.bg_scales = []
            self.bg_pcd = []
            for i in tqdm(range(self.num_scenes)):
                bg_scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
                bg_pcd = torch.tensor(bg_pnt)
                self.bg_scales.append(bg_scales)
                self.bg_pcd.append(bg_pcd)

            self.bg_field = MLP(
                in_dim=9,
                num_layers=2,
                layer_width=64,
                out_dim=6,
                activation=nn.ReLU(),
                out_activation=nn.Tanh(),
                implementation="torch",
            )
        
        self.renderer_rgb = RGBRenderer(background_color='black')
```

**关键组件**：
1. **稀疏卷积网络** (`sparse_conv`)：用于3D特征提取
2. **投影器** (`projector`)：从2D图像采样特征
3. **MLP解码器**：多个MLP用于预测Gaussian参数
   - `gaussion_decoder`：预测球谐系数（颜色）
   - `mlp_conv`：预测尺度和旋转
   - `mlp_opacity`：预测不透明度
   - `mlp_offset`：预测位置偏移
4. **背景模型** (`bg_field`)：处理天空和远景

#### 1.5 参数组管理

```379:393:nerfstudio/models/evolsplat.py
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = {}
        ## add mlp decoder parameters
        gps['gaussianDecoder'] = list(self.gaussion_decoder.parameters())
        gps['mlp_conv'] = list(self.mlp_conv.parameters())
        gps['mlp_opacity'] = list(self.mlp_opacity.parameters())
        gps['mlp_offset'] = list(self.mlp_offset.parameters())
        gps['sparse_conv'] = list(self.sparse_conv.parameters())
        gps['background_model'] = list(self.bg_field.parameters())
        return gps
```

**设计说明**：
- 实现 `get_param_groups()` 方法，返回不同参数组
- 每个参数组可以在配置中设置不同的学习率和调度器
- 这是 nerfstudio 框架的标准接口

---

## 数据层改动

### 1. 零样本数据解析器

**文件位置**：`nerfstudio/data/dataparsers/zeroshot_dataparser.py`

#### 1.1 数据解析器配置

```40:91:nerfstudio/data/dataparsers/zeroshot_dataparser.py
@dataclass
class ZeroshotDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Zeroshot)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    eval_mode: Literal["manner", "filename", "interval", "all"] = "manner"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    mask_color: Optional[Tuple[float, float, float]] = None
    """Replace the unknown pixels with this color. Relevant if you have a mask but still sample everywhere."""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction."""
    pcd_ration: int = 1
    """ the downscale ration of input pointcloud """
    kitti: bool = False
    """Enable inference on KITTI360 dataset"""
    waymo: bool = False
    """Enable inference on Waymo dataset"""
    include_depth: bool = True
    """whether or not to include loading of Metric Depth"""
    num_scenes:int = 1
    """Number of Pretrain Scenes"""

    def validate(self) -> None:
        """Validate the configuration"""
        if self.kitti == self.waymo:
            raise ValueError(
                "Exactly one dataset type must be selected. Set either kitti=True or waymo=True, but not both or neither."
            )
```

**关键特性**：
- 继承 `DataParserConfig`，符合 nerfstudio 数据解析器接口
- 支持 KITTI-360 和 Waymo 数据集
- 支持加载3D点云数据（`load_3D_points`）
- 支持深度图加载（`include_depth`）

#### 1.2 点云数据加载

```314:341:nerfstudio/data/dataparsers/zeroshot_dataparser.py
    def _load_3D_points(self, ply_file_path: Path, ratio: int = 3):
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
        """
        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        pcd = o3d.io.read_point_cloud(str(ply_file_path))
        # if no points found don't read in an initial point cloud
        if len(pcd.points) == 0:
            return None
        points3D = np.asarray(pcd.points, dtype=np.float32)[::ratio,:]

        points3D = torch.from_numpy(points3D)
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))[::ratio,:]

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }

        return out
```

**设计说明**：
- 使用 Open3D 加载 PLY 格式点云
- 支持点云下采样（`ratio` 参数）
- 返回包含位置和颜色的字典，格式与模型期望一致

#### 1.3 数据传递机制

```298:312:nerfstudio/data/dataparsers/zeroshot_dataparser.py
        # reinitialize metadata for dataparser_outputs
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=1.0,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "mask_color": self.config.mask_color,
                "input_pnt": seed_point,
            },
        )
        return dataparser_outputs
```

**关键设计**：
- 将点云数据存储在 `metadata["input_pnt"]` 中
- 通过 `DataparserOutputs` 传递给 DataManager
- Pipeline 会从 metadata 中提取并传递给模型

---

## 组件层改动

### 1. 稀疏卷积网络

**文件位置**：`nerfstudio/model_components/sparse_conv.py`

#### 1.1 SparseCostRegNet 实现

```148:184:nerfstudio/model_components/sparse_conv.py
class SparseCostRegNet(nn.Module):

    def __init__(self, d_in,d_out=8):
        super(SparseCostRegNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.conv0 = BasicSparseConvolutionBlock(d_in, d_out)

        self.conv1 = BasicSparseConvolutionBlock(d_out, 16, stride=2)
        self.conv2 = BasicSparseConvolutionBlock(16, 16)

        self.conv3 = BasicSparseConvolutionBlock(16, 32, stride=2)
        self.conv4 = BasicSparseConvolutionBlock(32, 32)

        self.conv5 = BasicSparseConvolutionBlock(32, 64, stride=2)
        self.conv6 = BasicSparseConvolutionBlock(64, 64)

        self.conv7 = BasicSparseDeconvolutionBlock(64, 32, ks=3, stride=2)

        self.conv9 = BasicSparseDeconvolutionBlock(32, 16, ks=3, stride=2)

        self.conv11 = BasicSparseDeconvolutionBlock(16, d_out, ks=3, stride=2)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        return x.F
```

**架构特点**：
- 使用 torchsparse 库实现稀疏卷积
- U-Net 结构：下采样 + 上采样 + 残差连接
- 只处理有效体素，节省内存

#### 1.2 稀疏张量构建

```198:221:nerfstudio/model_components/sparse_conv.py
def construct_sparse_tensor(raw_coords, feats, Bbx_min: torch.Tensor, Bbx_max: torch.Tensor, voxel_size=0.1):
    X_MIN, X_MAX = Bbx_min[0], Bbx_max[0]
    Y_MIN, Y_MAX = Bbx_min[1], Bbx_max[1]
    Z_MIN, Z_MAX = Bbx_min[2], Bbx_max[2]

    if isinstance(raw_coords, torch.Tensor) or isinstance(feats, torch.Tensor):
        raw_coords = raw_coords.cpu().numpy()
        feats = feats.cpu().numpy()

    bbx_max = np.array([X_MAX,Y_MAX,Z_MAX])
    bbx_min = np.array([X_MIN,Y_MIN,Z_MIN])
    vol_dim = (bbx_max - bbx_min) / 0.1
    vol_dim = vol_dim.astype(int).tolist()

    raw_coords -= np.array([X_MIN,Y_MIN,Z_MIN]).astype(int)
    coords, indices = sparse_quantize(raw_coords, voxel_size, return_index=True)  ## voxelize the pnt to discrete formation
    coords = torch.tensor(coords, dtype=torch.int).cuda()
    zeros = torch.zeros(coords.shape[0], 1).cuda()
    ## Note: [B,X,Y,Z] in Torch sparsev 2.1
    coords = torch.cat((zeros,coords), dim=1).to(torch.int32)  

    feats = torch.tensor(feats[indices], dtype=torch.float).cuda()
    sparse_feat = SparseTensor(feats,coords=coords)
    return sparse_feat, vol_dim, coords[:,1:]
```

**功能**：
- 将3D点云体素化为稀疏张量
- 计算体积维度
- 返回 torchsparse 的 SparseTensor 格式

### 2. 2D特征投影器

**文件位置**：`nerfstudio/model_components/projection.py`

#### 2.1 Projector 类

```24:103:nerfstudio/model_components/projection.py
class Projector():
    def __init__(self):
        print("Init the Projector in OpenGL system")

    
    def inbound(self, pixel_locations, h, w):
        '''
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        '''
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras,train_intrinsics):
        '''
        project 3D points into cameras
        :param xyz: [..., 3]  Opencv
        :param train_cameras: [n_views, 4, 4]  OpenGL
        :param camera intrinsics: [n_views, 4, 4]
        :return: pixel locations [..., 2], mask [...]
        '''
        original_shape = xyz.shape[:1]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        train_cameras = train_cameras * torch.tensor([1, -1, -1, 1],device="cuda")
        train_poses = train_cameras.reshape(-1, 4, 4)  # [n_views, 4, 4]

        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
        projections = train_intrinsics.bmm(torch.inverse(train_poses)) \
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        mask = projections[..., 2] > 0   # a point is invalid if behind the camera

        depth = projections[..., 2].reshape((num_views, ) + original_shape)
        return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
               mask.reshape((num_views, ) + original_shape),\
               depth
    
    ## compute the projection of the query points to model the Background
    def compute(self,  xyz, train_imgs, train_cameras, train_intrinsics,cam_idx=0):
        '''
        :param xyz: [n_samples, 3]
        :param source_imgs: [ n_views, c, h, w]
        :param source_cameras: [ n_views, 4, 4], in OpnecGL
        :param source_intrinsics: [ n_views, 4, 4]
        :return: rgb_feat_sampled: [n_samples,n_views,c],
                 mask: [n_samples,n_views,1]
        '''
    
        xyz = xyz.detach()
        h, w = train_imgs.shape[2:]

        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front, _ = self.compute_projections(xyz, train_cameras,train_intrinsics.clone())
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_points, 2]
        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(dim=1) # [n_views, 1, n_points, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=False)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1).squeeze(dim=0)  # [n_points, n_views, 3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().permute(1, 0)[..., None]   # [n_rays, n_samples, n_views, 1]
        rgb = rgb_sampled.masked_fill(mask==0, 0)

        projection_mask = mask[..., :].sum(dim=1) > 0
        return rgb[projection_mask.squeeze()], projection_mask.squeeze()
```

**功能**：
- 将3D点投影到2D图像平面
- 从多视角图像中采样RGB特征
- 处理遮挡和边界检查

#### 2.2 窗口采样

```106:165:nerfstudio/model_components/projection.py
    def sample_within_window(self,  xyz, train_imgs, train_cameras, train_intrinsics, source_depth=None, local_radius = 2, depth_delta=0.2):
        '''
        :param xyz: [n_samples, 3]
        :param source_imgs: [ n_views, c, h, w]
        :param source_cameras: [ n_views, 4, 4], in OpnecGL
        :param source_intrinsics: [ n_views, 4, 4]
        :param source_depth: [ n_views , h, w] for occlusion-aware IBR
        :return: rgb_feat_sampled: [n_samples,n_views,c],
                 mask: [n_samples,n_views,1]
        '''
        n_views, _ ,_ = train_cameras.shape
        n_samples = xyz.shape[0]
        
        local_h = 2 * local_radius + 1
        local_w = 2 * local_radius + 1
        window_grid = self.generate_window_grid(-local_radius, local_radius,
                                                -local_radius, local_radius,
                                                local_h, local_w, device=xyz.device)  # [2R+1, 2R+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(n_views, 1, 1)

        xyz = xyz.detach()
        h, w = train_imgs.shape[2:]

        # sample within the window size
        pixel_locations, mask_in_front, project_depth = self.compute_projections(xyz, train_cameras,train_intrinsics.clone())

        ## Occlusion-Aware check for IBR:
        if source_depth is not None:
            source_depth = source_depth.unsqueeze(-1).permute(0, 3, 1, 2).cuda()
            depths_sampled = F.grid_sample(source_depth, self.normalize(pixel_locations, h, w).unsqueeze(dim=1), align_corners=False)
            depths_sampled = depths_sampled.squeeze()
            retrived_depth = depths_sampled.masked_fill(mask_in_front==0, 0)
            projected_depth = project_depth*mask_in_front

            """Use depth priors to distinguish the Occlusion Region"""
            visibility_map = projected_depth - retrived_depth
            visibility_map = visibility_map.unsqueeze(-1).repeat(1,1, local_h*local_w).reshape(n_views,n_samples,-1)
        else:
            visibility_map = torch.ones_like(project_depth)

        pixel_locations = pixel_locations.unsqueeze(dim=2) + window_grid.unsqueeze(dim=1)
        pixel_locations = pixel_locations.reshape(n_views,-1,2)  ## [N_view, N_points,2]

        ## boardcasting the mask
        mask_in_front = mask_in_front.unsqueeze(-1).repeat(1,1, local_h*local_w).reshape(n_views,-1)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_points, 2]
        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(dim=1) # [n_views, 1, n_points, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=False)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1).squeeze(dim=0)  # [n_points, n_views, 3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front ).float().permute(1, 0)[..., None]  
        rgb = rgb_sampled.masked_fill(mask==0, 0)

        return rgb.reshape(n_samples,n_views,local_w*local_h,3), \
                mask.reshape(n_samples,n_views,local_w*local_h),\
                visibility_map.permute(1,0,2).unsqueeze(-1)
```

**关键特性**：
- 在投影点周围采样局部窗口（`local_radius`）
- 支持遮挡感知（使用深度图）
- 返回可见性图用于特征融合

---

## 配置系统集成

### 1. 方法配置注册

**文件位置**：`nerfstudio/configs/method_configs.py`

```654:704:nerfstudio/configs/method_configs.py
method_configs["evolsplat"] = TrainerConfig(
    method_name="evolsplat",
    steps_per_eval_image=4000,
    steps_per_eval_batch=0,
    steps_per_save=3000,
    steps_per_eval_all_images=50000000,
    max_num_iterations=50000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=SplatDatamanagerConfig(
            dataparser=EvolSplatDataParserConfig(load_3D_points=True)    ## pretrain on multi-scene
        ),
        model=EvolSplatModelConfig(),
    ),
    
    optimizers={
        "sparse_conv": {
            "optimizer": AdamOptimizerConfig(lr=1*1e-3, eps=1e-15),
             "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=500, lr_pre_warmup=0
            ),
        },

        "mlp_conv": {
            "optimizer": AdamOptimizerConfig(lr=1*1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001,max_steps=30000),
        },

          "mlp_opacity": {
            "optimizer": AdamOptimizerConfig(lr=1*1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001,max_steps=30000),
        },

         "mlp_offset": {
            "optimizer": AdamOptimizerConfig(lr=1*1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001,max_steps=30000),
        },

        "gaussianDecoder": {
            "optimizer": AdamOptimizerConfig(lr=1*1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001,max_steps=30000),
        },

        "background_model": {
            "optimizer": AdamOptimizerConfig(lr=1*1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001,max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis= "tensorboard",
)
```

**关键设计**：
- 在 `method_configs` 字典中注册 `"evolsplat"` 方法
- 配置 Pipeline、DataManager 和 Model
- 为每个参数组设置独立的优化器和学习率调度器
- 使用 `SplatDatamanagerConfig` 管理数据

### 2. 数据解析器注册

**文件位置**：`nerfstudio/configs/dataparser_configs.py`

```40:61:nerfstudio/configs/dataparser_configs.py
from nerfstudio.data.dataparsers.evolsplat_dataparser import EvolSplatDataParserConfig
from nerfstudio.data.dataparsers.zeroshot_dataparser import ZeroshotDataParserConfig

dataparsers = {
    "nerfstudio-data": NerfstudioDataParserConfig(),
    "minimal-parser": MinimalDataParserConfig(),
    "arkit-data": ARKitScenesDataParserConfig(),
    "blender-data": BlenderDataParserConfig(),
    "instant-ngp-data": InstantNGPDataParserConfig(),
    "nuscenes-data": NuScenesDataParserConfig(),
    "dnerf-data": DNeRFDataParserConfig(),
    "phototourism-data": PhototourismDataParserConfig(),
    "dycheck-data": DycheckDataParserConfig(),
    "scannet-data": ScanNetDataParserConfig(),
    "sdfstudio-data": SDFStudioDataParserConfig(),
    "nerfosr-data": NeRFOSRDataParserConfig(),
    "sitcoms3d-data": Sitcoms3DDataParserConfig(),
    "scannetpp-data": ScanNetppDataParserConfig(),
    "colmap": ColmapDataParserConfig(),
    "evolsplat-data":EvolSplatDataParserConfig(),
    "zeronpt-data":ZeroshotDataParserConfig(),
}
```

**说明**：
- 注册两个数据解析器：`evolsplat-data`（训练）和 `zeronpt-data`（零样本推理）
- 通过字典注册，nerfstudio 会自动发现并集成

---

## 训练流程集成

### 1. Pipeline 数据传递

**文件位置**：`nerfstudio/pipelines/base_pipeline.py`

```259:270:nerfstudio/pipelines/base_pipeline.py
        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        elif "input_pnt" in self.datamanager.train_dataparser_outputs.metadata:
            seed_pts = self.datamanager.train_dataparser_outputs.metadata["input_pnt"]
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
```

**数据流**：
1. DataParser 将点云数据存储在 `metadata["input_pnt"]`
2. Pipeline 从 DataManager 的 `train_dataparser_outputs.metadata` 中提取
3. Pipeline 在创建 Model 时传递 `seed_points` 参数

### 2. 模型接口实现

EVolSplat 实现了 nerfstudio Model 基类的关键方法：

- `get_outputs()`: 前向传播，返回渲染结果
- `get_loss_dict()`: 计算损失字典
- `get_metrics_dict()`: 计算评估指标
- `get_image_metrics_and_images()`: 生成评估图像
- `get_param_groups()`: 返回参数组
- `get_training_callbacks()`: 返回训练回调

这些方法符合 nerfstudio 的接口规范，确保与训练流程无缝集成。

---

## 关键设计决策

### 1. 多场景支持

EVolSplat 支持多场景训练，通过以下方式实现：

- **点云存储**：每个场景的点云存储在独立的列表中
- **场景ID**：batch 中包含 `scene_id`，用于索引对应场景的数据
- **参数管理**：每个场景的 means、scales、offset 等参数独立存储

### 2. 体积冻结机制

通过 `freeze_volume` 配置参数，支持：

- **训练阶段**：动态更新3D体积特征
- **推理阶段**：冻结体积，只优化Gaussian参数

### 3. 背景模型

使用半球背景模型处理天空和远景：

- 初始化背景球点云
- 使用独立的MLP预测背景颜色和尺度
- 与前景Gaussian Splats 混合渲染

### 4. 遮挡感知

在投影器中实现遮挡感知：

- 使用深度图判断点是否被遮挡
- 生成可见性图用于特征融合
- 提高多视角特征聚合的准确性

---

## 总结

EVolSplat 通过以下方式成功集成到 nerfstudio 框架：

1. **模块化设计**：新增组件遵循 nerfstudio 的接口规范
2. **配置驱动**：通过配置文件注册方法和数据解析器
3. **数据流清晰**：通过 metadata 和 Pipeline 传递点云数据
4. **接口兼容**：实现 Model 基类的所有必需方法
5. **扩展性好**：可以轻松添加新功能而不破坏现有框架

这种集成方式使得 EVolSplat 能够充分利用 nerfstudio 的训练、评估、可视化等功能，同时保持代码的模块化和可维护性。

