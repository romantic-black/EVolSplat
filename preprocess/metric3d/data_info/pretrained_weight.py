
import os

mldb_info = {}

# Root directory for pretrained backbones.
# We redirect this to the local Metric3D models folder inside this repo,
# so you don't need the original /data0/jxhuang/models path.
_default_metric3d_root = os.getenv(
    "METRIC3D_PATH",
    "/root/drivestudio-coding/third_party/EVolSplat/preprocess/metric3d",
)
_default_mldb_root = os.path.join(_default_metric3d_root, "models")

mldb_info["checkpoint"] = {
    # Root for all backbone checkpoints
    "mldb_root": _default_mldb_root,

    # pretrained weight for convnext
    "convnext_tiny": "convnext/convnext_tiny_22k_1k_384.pth",
    "convnext_small": "convnext/convnext_small_22k_1k_384.pth",
    "convnext_base": "convnext/convnext_base_22k_1k_384.pth",
    "convnext_large": "convnext/convnext_large_22k_1k_384.pth",

    # pretrained weight for DINOv2 ViT backbones
    "vit_large": "vit/dinov2_vitl14_pretrain.pth",
    "vit_small_reg": "vit/dinov2_vits14_reg4_pretrain.pth",
    "vit_large_reg": "dinov2/dinov2_vitl14_reg4_pretrain.pth",
    "vit_giant2_reg": "vit/dinov2_vitg14_reg4_pretrain.pth",
}