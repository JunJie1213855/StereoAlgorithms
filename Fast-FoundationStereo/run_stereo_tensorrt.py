# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
TensorRT 双目立体匹配推理

输入: 左右目图像
输出: 视差图可视化
"""

import os
import sys
import argparse

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}')

from omegaconf import OmegaConf
import torch
import imageio
import logging
import yaml
import numpy as np
import cv2
from Utils import (
    set_logging_format, set_seed, vis_disparity,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_dir', default=f'{code_dir}/onnxmodel', type=str,
                        help='Directory containing ONNX model files')
    parser.add_argument('--left_file', default=f'{code_dir}/rect_left.png', type=str,
                        help='Left image path')
    parser.add_argument('--right_file', default=f'{code_dir}/rect_right.png', type=str,
                        help='Right image path')
    parser.add_argument('--out_dir', default=f'{code_dir}/output', type=str,
                        help='Output directory')
    parser.add_argument("--max-disp", default=192, type=int,
                        help= "the maxium of disparity")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    os.system(f'rm -rf {args.out_dir} && mkdir -p {args.out_dir}')

    # 加载配置
    onnx_config_path = os.path.join(os.path.dirname(args.onnx_dir), 'onnx.yaml')
    if os.path.exists(onnx_config_path):
        with open(onnx_config_path, 'r') as ff:
            cfg = dict(yaml.safe_load(ff))
        for k in args.__dict__:
            if args.__dict__[k] is not None:
                cfg[k] = args.__dict__[k]
        args = OmegaConf.create(cfg)

    logging.info(f"args:\n{args}")

    # 加载 TensorRT 模型
    from core.foundation_stereo import TrtRunner
    model = TrtRunner(args, args.onnx_dir + '/feature_runner.engine', args.onnx_dir + '/post_runner.engine')

    # 读取图像
    img0 = imageio.imread(args.left_file)
    img1 = imageio.imread(args.right_file)

    if len(img0.shape) == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]
    H, W = img0.shape[:2]

    # 图像缩放
    if hasattr(args, 'image_size'):
        fx = args.image_size[1] / img0.shape[1]
        fy = args.image_size[0] / img0.shape[0]
        if fx != 1 or fy != 1:
            logging.info(f"Resizing image to {args.image_size}, fx: {fx}, fy: {fy}")
        img0 = cv2.resize(img0, fx=fx, fy=fy, dsize=None)
        img1 = cv2.resize(img1, fx=fx, fy=fy, dsize=None)

    H, W = img0.shape[:2]
    img0_ori = img0.copy()
    img1_ori = img1.copy()

    logging.info(f"Image size: {img0.shape}")
    imageio.imwrite(f'{args.out_dir}/left.png', img0)
    imageio.imwrite(f'{args.out_dir}/right.png', img1)

    # 转换为 tensor
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

    # 推理
    logging.info("Start forward...")
    disp = model.forward(img0, img1)
    logging.info("Forward done")

    # 处理视差图
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)
    if fx != 1:
        disp = disp * fx

    # 保存视差图
    np.save(f'{args.out_dir}/disp.npy', disp)

    # 可视化视差
    vis = vis_disparity(disp, cmap=None, color_map=cv2.COLORMAP_TURBO)
    vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/disp_vis.png', vis)

    # 显示
    s = 1280 / vis.shape[1]
    resized_vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))
    cv2.imshow('disparity', resized_vis[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    logging.info(f"Results saved to {args.out_dir}")
    logging.info(f"  - disp.npy: Raw disparity map")
    logging.info(f"  - disp_vis.png: Disparity visualization")
