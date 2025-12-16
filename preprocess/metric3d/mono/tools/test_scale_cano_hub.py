import os
import os.path as osp
import sys
import time
import argparse

CODE_SPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)

import cv2
import numpy as np
import torch

try:
    from mmcv.utils import Config
except Exception:
    from mmengine import Config

from mono.utils.custom_data import load_data
from mono.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Metric3D inference via torch.hub (NuScenes, Scale-Cano-like).')
    parser.add_argument('config', help='config file path (used to get data_basic only)')
    parser.add_argument('--show-dir', help='the dir to save logs and visualization results')
    parser.add_argument('--load-from', help='unused, kept for CLI compatibility with test_scale_cano.py')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--cam_id', type=int, default=0, help='the camera id (None means process all cameras)')
    parser.add_argument('--options', nargs='+', help='unused, kept for CLI compatibility')
    parser.add_argument(
        '--launcher',
        choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'],
        default='None',
        help='job launcher (only None is supported here)',
    )
    parser.add_argument(
        '--test_data_path',
        default='None',
        type=str,
        help='the path of test data (NuScenes scene root)',
    )
    parser.add_argument(
        '--dataset',
        default='nuscenes',
        type=str,
        help='dataset type (only nuscenes is tested in this script)',
    )
    args = parser.parse_args()
    return args


def build_logger_and_cfg(args):
    os.chdir(CODE_SPACE)
    cfg = Config.fromfile(args.config)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp

    if args.show_dir is not None:
        show_dir = args.show_dir
    else:
        show_dir = osp.join(
            './show_dirs',
            osp.splitext(osp.basename(args.config))[0],
            timestamp,
        )

    cfg.show_dir = show_dir
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)

    cfg.log_file = osp.join(cfg.show_dir, f'{timestamp}.log')
    logger = setup_logger(cfg.log_file)

    logger.info(f'Config loaded from {args.config}')
    logger.info(f'Show dir: {cfg.show_dir}')

    return cfg, logger


def prepare_test_data(args):
    test_data_path = args.test_data_path
    if not os.path.isabs(test_data_path):
        test_data_path = osp.join(CODE_SPACE, test_data_path)

    if not osp.exists(test_data_path):
        raise FileNotFoundError(f'test_data_path not found: {test_data_path}')

    data = load_data(args.test_data_path, args.dataset, args.cam_id)
    return data


def transform_single(rgb_path, intrinsic, data_basic):
    """
    参考 mono.utils.do_test.transform_test_data_scalecano 的实现，
    但简化为当前脚本内的局部函数，避免对 Distributed/DataParallel 的依赖。
    """
    from mono.utils.do_test import resize_for_input

    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise RuntimeError(f'Failed to read image: {rgb_path}')

    # BGR to RGB
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    canonical_space = data_basic['canonical_space']
    forward_size = data_basic.crop_size

    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    ori_h, ori_w, _ = rgb.shape
    ori_focal = (intrinsic[0] + intrinsic[1]) / 2
    canonical_focal = canonical_space['focal_length']

    cano_label_scale_ratio = canonical_focal / max(ori_focal, 1e-6)

    canonical_intrinsic = [
        intrinsic[0] * cano_label_scale_ratio,
        intrinsic[1] * cano_label_scale_ratio,
        intrinsic[2],
        intrinsic[3],
    ]

    rgb_resized, cam_model, pad, resize_label_scale_ratio = resize_for_input(
        rgb,
        forward_size,
        canonical_intrinsic,
        [ori_h, ori_w],
        1.0,
    )

    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio

    rgb_tensor = torch.from_numpy(rgb_resized.transpose((2, 0, 1))).float()
    rgb_tensor = torch.div((rgb_tensor - mean), std)
    rgb_tensor = rgb_tensor[None, :, :, :].cuda(non_blocking=True)

    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :].cuda(non_blocking=True)

    cam_model_stacks = [
        torch.nn.functional.interpolate(
            cam_model,
            size=(cam_model.shape[2] // i, cam_model.shape[3] // i),
            mode='bilinear',
            align_corners=False,
        )
        for i in [2, 4, 8, 16, 32]
    ]

    pad_info = pad
    ori_shape = [ori_h, ori_w]

    return rgb_tensor, cam_model_stacks, pad_info, label_scale_factor, ori_shape


def run_inference_hub(model, rgb_tensor, cam_models_stacks, pad_info, scale_info, normalize_scale, ori_shape):
    """
    类似 mono.utils.do_test.get_prediction，但针对 hub 加载的单机模型（不使用 model.module）。
    
    注意：不进行插值回原始尺寸，以节省存储空间。深度图保存为去除padding后的模型输出尺寸。
    如需使用原始尺寸，请使用 depth_utils.py 中的工具函数进行插值。
    """
    data = dict(
        input=rgb_tensor,
        cam_model=cam_models_stacks,
    )

    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference(data)

    pred_depth = pred_depth.squeeze()
    # 去除 padding，得到有效区域的深度图
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[0] - pad_info[1],
        pad_info[2] : pred_depth.shape[1] - pad_info[3],
    ]

    # 应用尺度恢复（但不插值到原始尺寸）
    pred_depth = pred_depth * normalize_scale / max(scale_info, 1e-6)

    return pred_depth


def main():
    args = parse_args()
    cfg, logger = build_logger_and_cfg(args)

    if args.launcher != 'None':
        logger.warning('Only launcher == \"None\" is supported in test_scale_cano_hub.py, forcing to None.')

    # load test data (NuScenes)
    test_data = prepare_test_data(args)
    if len(test_data) == 0:
        logger.warning('No test data found, nothing to do.')
        return

    # load model from torch.hub
    try:
        logger.info('Loading Metric3D model from torch.hub: yvanyin/metric3d, metric3d_vit_giant2')
        # trust_repo=True 消除信任警告（官方 Metric3D 仓库是可信的）
        model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_giant2', pretrain=True, trust_repo=True)
    except Exception as e:
        logger.error(f'Failed to load Metric3D model from torch.hub: {e}')
        logger.error(
            'Please check network connection or pre-download the repository and set TORCH_HUB_DIR accordingly.'
        )
        raise

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    normalize_scale = cfg.data_basic.depth_range[1]

    num_saved = 0
    for an in test_data:
        rgb_path = an['rgb']
        intrinsic = an.get('intrinsic', None)

        if intrinsic is None:
            logger.warning(
                f'Intrinsic not found for {rgb_path}, using default fx=fy=1000, cx=width/2, cy=height/2.'
            )
            tmp_img = cv2.imread(rgb_path)
            if tmp_img is None:
                logger.warning(f'Failed to read image when estimating intrinsic: {rgb_path}, skip.')
                continue
            intrinsic = [1000.0, 1000.0, tmp_img.shape[1] / 2.0, tmp_img.shape[0] / 2.0]

        rgb_tensor, cam_models_stacks, pad_info, label_scale_factor, ori_shape = transform_single(
            rgb_path, intrinsic, cfg.data_basic
        )

        pred_depth = run_inference_hub(
            model=model,
            rgb_tensor=rgb_tensor,
            cam_models_stacks=cam_models_stacks,
            pad_info=pad_info,
            scale_info=label_scale_factor,
            normalize_scale=normalize_scale,
            ori_shape=ori_shape,
        )

        pred_np = pred_depth.detach().cpu().numpy().astype(np.float16)
        save_name = osp.splitext(an['filename'])[0] + '.npy'
        save_path = osp.join(cfg.show_dir, save_name)
        
        # 保存元数据以便后续使用（原始图像尺寸、scale_info等）
        metadata = {
            'ori_shape': ori_shape,  # 原始图像尺寸 [H, W]
            'depth_shape': list(pred_np.shape),  # 当前深度图尺寸 [H, W]
            'scale_info': float(label_scale_factor),
            'normalize_scale': float(normalize_scale),
            'intrinsic': intrinsic,
        }
        metadata_path = osp.join(cfg.show_dir, save_name.replace('.npy', '_meta.npz'))
        
        try:
            np.save(save_path, pred_np)
            np.savez(metadata_path, **metadata)
            num_saved += 1
        except Exception as e:
            logger.error(f'Failed to save depth to {save_path}: {e}')

    logger.info(f'Depth prediction finished. Saved {num_saved} depth maps to {cfg.show_dir}')


if __name__ == '__main__':
    main()


