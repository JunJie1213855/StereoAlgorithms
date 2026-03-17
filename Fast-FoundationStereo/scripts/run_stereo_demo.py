# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
import argparse, torch, imageio, logging, yaml
import numpy as np
from Utils import (
    AMP_DTYPE, set_logging_format, set_seed, vis_disparity,
)
import cv2


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default=f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth', type=str)
    parser.add_argument('--left_file', default=f'{code_dir}/../rect_left.png', type=str)
    parser.add_argument('--right_file', default=f'{code_dir}/../rect_right.png', type=str)
    parser.add_argument('--out_dir', default=f'{code_dir}/../output', type=str)
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--hiera', default=0, type=int)
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
    parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    os.system(f'rm -rf {args.out_dir} && mkdir -p {args.out_dir}')

    with open(f'{os.path.dirname(args.model_dir)}/cfg.yaml', 'r') as ff:
        cfg: dict = yaml.safe_load(ff)
    for k in args.__dict__:
        if args.__dict__[k] is not None:
            cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp

    model.cuda().eval()

    scale = args.scale

    img0 = imageio.imread(args.left_file)
    img1 = imageio.imread(args.right_file)
    if len(img0.shape) == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]
    H, W = img0.shape[:2]

    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
    H, W = img0.shape[:2]
    img0_ori = img0.copy()
    img1_ori = img1.copy()
    logging.info(f"img0: {img0.shape}")
    imageio.imwrite(f'{args.out_dir}/left.png', img0)
    imageio.imwrite(f'{args.out_dir}/right.png', img1)

    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    logging.info(f"Start forward, 1st time run can be slow due to compilation")
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        if not args.hiera:
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
        else:
            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    logging.info("forward done")
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

    # Save disparity map
    np.save(f'{args.out_dir}/disp.npy', disp)

    # Visualize disparity
    cmap = None
    min_val = None
    max_val = None
    vis = vis_disparity(disp, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)
    vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/disp_vis.png', vis)

    logging.info(f"Results saved to {args.out_dir}")
    logging.info(f"  - disp.npy: Raw disparity map")
    logging.info(f"  - disp_vis.png: Disparity visualization")
