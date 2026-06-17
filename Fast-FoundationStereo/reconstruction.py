# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
立体视觉三维重建算法流程:
1. 使用双目标定参数 (YAML格式) 进行图像校正
2. 使用 FoundationStereo 进行立体匹配
3. 根据视差图和重投影矩阵 Q 进行三维重建
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch
import imageio
import logging
import yaml
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from core.utils.utils import InputPadder
from Utils import (
    AMP_DTYPE, set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
import stereoconfig


def create_point_cloud(points_3d: np.ndarray, image: np.ndarray, filename: str):
    """创建带颜色的点云 PLY 文件"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_points = []

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            x = points_3d[row, col, 0]
            y = points_3d[row, col, 1]
            z = points_3d[row, col, 2]

            # 过滤无效点
            if np.isinf(z) or np.isnan(z) or z <= 0:
                continue

            output_points.append([
                x, y, z,
                image[row, col, 0],
                image[row, col, 1],
                image[row, col, 2]
            ])

    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(output_points)))
        np.savetxt(f, output_points, '%f %f %f %d %d %d')


def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，显示点击位置的深度"""
    if event == cv2.EVENT_LBUTTONDOWN:
        depth = np.sqrt(
            param[y, x, 0] * param[y, x, 0] +
            param[y, x, 1] * param[y, x, 1] +
            param[y, x, 2] * param[y, x, 2]
        )
        print(f'坐标 ({x}, {y}) 的距离为: {depth:.4f} m')


def main(args: argparse.Namespace):
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    # 创建输出目录
    os.system(f'rm -rf {args.out_dir} && mkdir -p {args.out_dir}')

    # 加载模型配置
    with open(f'{os.path.dirname(args.model_dir)}/cfg.yaml', 'r') as ff:
        cfg: dict = yaml.safe_load(ff)

    # 合并命令行参数
    for k in args.__dict__:
        if args.__dict__[k] is not None:
            cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")

    # 加载模型
    logging.info("Loading model...")
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()

    # 加载相机标定参数
    logging.info(f"Loading camera config from {args.param_file}")
    stereo_cam = stereoconfig.stereoCamera(args.param_file)

    # 读取原始图像
    logging.info(f"Loading images from {args.left_img} and {args.right_img}")
    left_img = cv2.imread(args.left_img)
    right_img = cv2.imread(args.right_img)

    if left_img is None or right_img is None:
        raise FileNotFoundError("Failed to load images")

    # 图像校正
    logging.info("Rectifying images...")
    left_rected, right_rected = stereo_cam.rectify(left_img, right_img)

    # 保存校正后的图像
    if args.save_rectified:
        cv2.imwrite(f'{args.out_dir}/rect_left.png', left_rected)
        cv2.imwrite(f'{args.out_dir}/rect_right.png', right_rected)

    # 显示校正后的图像拼接（可选）
    if args.show_rectified:
        cat_img = stereo_cam.cat(left_rected, right_rected)
        cv2.imshow("Rectified Images", cat_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 使用校正后的图像进行立体匹配
    img0 = cv2.cvtColor(left_rected, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(right_rected, cv2.COLOR_BGR2RGB)
    H, W = img0.shape[:2]

    # 图像缩放
    if args.scale != 1:
        img0 = cv2.resize(img0, fx=args.scale, fy=args.scale, dsize=None)
        img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
        H, W = img0.shape[:2]

    img0_ori = img0.copy()
    img1_ori = img1.copy()

    logging.info(f"Image size: {img0.shape}")

    # 保存图像
    imageio.imwrite(f'{args.out_dir}/left.png', img0)
    imageio.imwrite(f'{args.out_dir}/right.png', img1)

    # 转换为 tensor 并 padding
    img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
    img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)

    # 前向推理
    logging.info("Start forward, first run can be slow due to compilation")
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        if not args.hiera:
            disp = model.forward(
                img0_tensor, img1_tensor,
                iters=args.valid_iters,
                test_mode=True,
                optimize_build_volume='pytorch1'
            )
        else:
            disp = model.run_hierachical(
                img0_tensor, img1_tensor,
                iters=args.valid_iters,
                test_mode=True,
                small_ratio=0.5
            )
    logging.info("Forward done")

    # 处理视差图
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

    # 去除不可见点（视差为负）
    if args.remove_invisible:
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx - disp
        invalid = us_right < 0
        disp[invalid] = np.inf

    # 保存视差图
    np.save(f'{args.out_dir}/disp.npy', disp)

    # 可视化视差
    vis = vis_disparity(disp, cmap=None, color_map=cv2.COLORMAP_JET)
    imageio.imwrite(f'{args.out_dir}/disp.png', vis)
    vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/disp_vis.png', vis)

    # 使用 Q 矩阵进行三维重建
    if args.get_pc:
        logging.info("Generating point cloud...")

        # 获取重投影矩阵 Q
        Q = stereo_cam.Q

        # 如果图像缩放了，需要调整 Q 矩阵
        if args.scale != 1:
            # Q 矩阵的缩放: [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
            # 缩放后: [fx*scale, 0, cx*scale, 0, 0, fy*scale, cy*scale, 0, 0, 0, 1, 0]
            Q_scaled = Q.copy()
            Q_scaled[0, 2] *= args.scale
            Q_scaled[1, 2] *= args.scale
            Q_scaled[0, 3] *= args.scale
            Q_scaled[1, 3] *= args.scale
            Q = Q_scaled

        # 使用 cv2.reprojectImageTo3D 获取三维点云
        points_3d = cv2.reprojectImageTo3D(disp, Q)
        np.save(f'{args.out_dir}/points_3d.npy', points_3d)

        # 创建 PLY 点云文件
        create_point_cloud(points_3d, img0, f'{args.out_dir}/cloud.ply')
        logging.info(f"Point cloud saved to {args.out_dir}/cloud.ply")

        # 使用 Open3D 处理点云
        xyz_map = points_3d.transpose(2, 0, 1).reshape(3, -1).T
        img_flat = img0.reshape(-1, 3)
        pcd = toOpen3dCloud(xyz_map, img_flat)

        # 过滤深度范围
        points = np.asarray(pcd.points)
        keep_mask = (points[:, 2] > 0) & (points[:, 2] <= args.zfar)
        keep_ids = np.arange(len(points))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)

        # 保存 Open3D 格式点云
        o3d.io.write_point_cloud(f'{args.out_dir}/cloud_o3d.ply', pcd)

        # 点云去噪（可选）
        if args.denoise_cloud:
            logging.info("Denoising point cloud...")
            pcd = pcd.voxel_down_sample(voxel_size=0.001)
            cl, ind = pcd.remove_radius_outlier(
                nb_points=args.denoise_nb_points,
                radius=args.denoise_radius
            )
            inlier_cloud = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
            pcd = inlier_cloud

        # 可视化
        if args.visualize:
            logging.info("Visualizing point cloud. Press ESC to exit.")
            o3d.visualization.draw_geometries(
                [pcd],
                window_name="Point Cloud",
                point_show_normal=False
            )

    # 使用 cv2 可视化（可选）
    if args.visualize_cv2:
        logging.info("Visualizing with OpenCV. Click on image to see depth.")
        cv2.imshow("rgb image", left_rected)
        cv2.setMouseCallback("rgb image", mouse_callback, points_3d)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    logging.info(f"All results saved to {args.out_dir}")


if __name__ == '__main__':
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument('--model_dir', default=f'{code_dir}/weights/23-36-37/model_best_bp2_serialize.pth', type=str)

    # 输入图像
    # parser.add_argument('--left_img', '-l', default=r"/root/code/C++/TensorRTTemplate/left.png", type=str,
    #                     help='Left image path')
    # parser.add_argument('--right_img', '-r', default=r"/root/code/C++/TensorRTTemplate/right.png", type=str,
    #                     help='Right image path')

    # # 相机标定参数文件 (YAML格式)
    # parser.add_argument('-p', '--param_file', default=f'{code_dir}/param/zed1280.yaml', type=str,
    #                     help='Camera calibration parameters in YAML format')

    parser.add_argument("--left_img","-l",default=r"/root/code/python/StereoMatch/StereoAlgorithms/example/1920x1080/left.jpg")
    parser.add_argument("--right_img","-r",default=r"/root/code/python/StereoMatch/StereoAlgorithms/example/1920x1080/right.jpg")
    parser.add_argument("-p","--param_file",default=r"/root/code/python/StereoMatch/StereoAlgorithms/example/1920x1080/zed.yaml")

    # 输出目录
    parser.add_argument('--out_dir', default=f'{code_dir}/output', type=str)

    # 图像缩放
    parser.add_argument('--scale', default=1, type=float)

    # 模型选项
    parser.add_argument('--hiera', default=False, type=bool)
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
    parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity')

    # 校正选项
    parser.add_argument('--show_rectified', action='store_true', default=True, help='Show rectified images')
    parser.add_argument('--save_rectified', action='store_true', default=False, help='Save rectified images')

    # 点云选项
    parser.add_argument('--remove_invisible', default=True, type=bool)
    parser.add_argument('--get_pc', type=bool, default=True, help='save point cloud output')
    parser.add_argument('--zfar', type=float, default=100, help="max depth to include in point cloud")

    # 点云去噪选项
    parser.add_argument('--denoise_cloud', default=True, type=bool)
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius for outlier removal')

    # 可视化选项
    parser.add_argument('--visualize', action='store_true', default=True, help='Visualize point cloud with Open3D')
    parser.add_argument('--visualize_cv2', action='store_true', default=True, help='Visualize with OpenCV')

    args = parser.parse_args()
    main(args)
