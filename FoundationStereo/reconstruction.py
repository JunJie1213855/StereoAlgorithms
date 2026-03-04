"""
立体视觉三维重建算法流程:
1.使用双目标定算法:
    a.获得参数 K_l,K_r,D_l,D_r,R_l,R_r,P_l,P_r,Q,相机类型，图像尺寸
    b.使用xml或者yaml文件将其保存

2.使用该三维重建方法
    a.修改 param_file 成你自己的xml或者yaml文件的路径
    b.修改输入的图像 left_img,right_img
    c.开始运行

注意: 本文使用FoundationStereo模型进行立体匹配
"""
import torch
import torch.nn.functional as F
import cv2
import time
import stereoconfig
import open3d as o3d
import sys
sys.path.append('core')
sys.path.append('..')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import logging
import trimesh
import imageio
from Utils import set_logging_format, set_seed, depth2xyzmap, toOpen3dCloud


# 设置日志格式
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')


def remove_light(img: np.ndarray):
    """去除图像中的高亮区域"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 225, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return dst


def create_ply(points_3d: np.ndarray, image: np.ndarray, filename):
    """创建PLY点云文件"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_points = []
    max_z = np.max(points_3d[:, :, 2])
    print(f"Max Z: {max_z}")

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            output_points.append(list([
                points_3d[row, col, 0],
                points_3d[row, col, 1],
                points_3d[row, col, 2],
                image[row, col, 0],
                image[row, col, 1],
                image[row, col, 2]
            ]))

    # 写入PLY文件
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
    """鼠标回调函数,用于显示像素点的深度"""
    if event == cv2.EVENT_LBUTTONDOWN:
        depth = np.sqrt(param[y, x, 0] ** 2 + param[y, x, 1] ** 2 + param[y, x, 2] ** 2)
        print(f'坐标 ({x}, {y}) 的距离为: {depth} m')


def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf):
    """可视化视差图"""
    disp = disp.copy()
    H, W = disp.shape[:2]
    invalid_mask = disp >= invalid_thres

    if (invalid_mask == 0).sum() == 0:
        return np.zeros((H, W, 3))

    if min_val is None:
        min_val = disp[invalid_mask == 0].min()
    if max_val is None:
        max_val = disp[invalid_mask == 0].max()

    vis = ((disp - min_val) / (max_val - min_val)).clip(0, 1) * 255
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]

    if invalid_mask.any():
        vis[invalid_mask] = 0

    return vis.astype(np.uint8)


def inference(left_img: np.ndarray, right_img: np.ndarray, model, valid_iters=32, hiera=0):
    """
    使用FoundationStereo模型进行立体匹配推理

    Args:
        left_img: 左图像 (RGB格式)
        right_img: 右图像 (RGB格式)
        model: FoundationStereo模型
        valid_iters: GRU迭代次数
        hiera: 是否使用分层推理

    Returns:
        disp: 视差图 (numpy数组)
    """
    print("Load FoundationStereo model for inference!")
    model.eval()

    with torch.no_grad():
        # 转换为tensor
        left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).float()[None].to(DEVICE)
        right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).float()[None].to(DEVICE)

        # Padding确保尺寸是32的倍数
        padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

        print(f"Input shape: {left_tensor.shape}")

        # 推理
        with torch.cuda.amp.autocast(True):
            if not hiera:
                disp = model.forward(left_tensor, right_tensor, iters=valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(left_tensor, right_tensor, iters=valid_iters, test_mode=True, small_ratio=0.5)

        # 转换回numpy
        disp = disp.cpu().numpy()
        disp = padder.unpad(disp)

    return disp.squeeze()


def main(args: argparse.Namespace):
    """主函数"""

    # 设置日志和种子
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    # 1. 加载模型配置
    print("Loading FoundationStereo model...")
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        raise ValueError("Please provide --ckpt_dir path to FoundationStereo checkpoint. " +
                        "You can download the model from the official repository.")
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")

    # 2. 创建模型
    model = FoundationStereo(args)

    # 3. 加载checkpoint
    if ckpt_dir is not None:
        assert ckpt_dir.endswith(".pth"), "Checkpoint file must be .pth"
        print(f"Loading checkpoint from {ckpt_dir}")
        checkpoint = torch.load(ckpt_dir, map_location=DEVICE)
        logging.info(f"ckpt global_step:{checkpoint.get('global_step', 'N/A')}, epoch:{checkpoint.get('epoch', 'N/A')}")

        target_model = model.module if hasattr(model, 'module') else model
        target_model.load_state_dict(checkpoint['model'])
        print("Checkpoint loaded successfully")

    model.to(DEVICE)
    model.eval()

    # 4. 加载相机参数
    print("Loading stereo camera parameters...")
    config = stereoconfig.stereoCamera(args.param_file)

    # 5. 读取左右图像
    print(f"Loading images...")
    left_img = cv2.imread(args.left_img)
    right_img = cv2.imread(args.right_img)

    if left_img is None or right_img is None:
        raise ValueError("Failed to load images. Please check the image paths.")

    # 6. 畸变矫正
    print("Performing rectification...")
    left_img_rect, right_img_rect = config.rectify(left_img, right_img)

    if args.show_rectified:
        print("Showing rectified images...")
        cat_img = config.cat(left_img_rect, right_img_rect)
        cv2.namedWindow("rectified_stereo", cv2.WINDOW_NORMAL)
        cv2.imshow("rectified_stereo", cat_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 7. 图像缩放
    scale = args.scale
    assert scale <= 1, "scale must be <=1"
    left_img_rect = cv2.resize(left_img_rect, fx=scale, fy=scale, dsize=None)
    right_img_rect = cv2.resize(right_img_rect, fx=scale, fy=scale, dsize=None)
    H, W = left_img_rect.shape[:2]
    left_img_rect_ori = left_img_rect.copy()

    # 转换为RGB
    left_img_rgb = cv2.cvtColor(left_img_rect, cv2.COLOR_BGR2RGB)
    right_img_rgb = cv2.cvtColor(right_img_rect, cv2.COLOR_BGR2RGB)

    # 8. 创建输出目录
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    # 9. 立体匹配获取视差
    print("Computing disparity...")
    disp = inference(left_img_rgb, right_img_rgb, model, valid_iters=args.valid_iters, hiera=args.hiera)

    # 10. 移除不可见点 (右图中对应的点在左图像外)
    if args.remove_invisible:
        print("Removing invisible points...")
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx - disp
        invalid = us_right < 0
        disp[invalid] = np.inf

    # 11. 保存视差图
    file_stem = args.save_name
    vis_disp = vis_disparity(disp)
    plt.imsave(output_directory / f"{file_stem}.png", vis_disp)
    # 保存原始视差图（归一化到0-255）
    disp_valid = disp[disp != np.inf]
    if len(disp_valid) > 0:
        disp_max = np.max(disp_valid)
        if disp_max > 0:
            cv2.imwrite(str(output_directory / f"{file_stem}_raw.png"), (disp / disp_max * 255).astype(np.uint8))
    print(f"Disparity map saved to {output_directory / f'{file_stem}.png'}")

    # 12. 三维重建 - 使用K矩阵和baseline
    print("Reconstructing 3D points...")
    # 从stereoconfig获取baseline，或从Q矩阵计算
    if hasattr(config, 'baseline'):
        baseline = config.baseline
    else:
        # 从Q矩阵计算: baseline = -1/Q[3,2]
        baseline = abs(1.0 / config.Q[3, 2])
    print(f"Baseline: {baseline} m")

    # 使用左相机内参矩阵
    K = config.cam_matrix_left.copy()
    K[:2] *= scale  # 应用缩放

    # 计算深度: depth = fx * baseline / disp
    disp_safe = disp.copy()
    disp_safe[disp_safe <= 0] = np.inf
    depth = K[0, 0] * baseline / disp_safe
    np.save(output_directory / f"{file_stem}_depth.npy", depth)

    # 转换为3D点云
    xyz_map = depth2xyzmap(depth, K)

    # 13. 创建点云
    print("Creating point cloud...")
    # 使用Open3D创建点云
    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), left_img_rect_ori.reshape(-1, 3))

    # 裁剪深度范围
    keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)

    # 保存点云
    ply_path = output_directory / f"{file_stem}.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"Point cloud saved to {ply_path}")

    # 14. 点云去噪
    if args.denoise:
        print("Applying statistical outlier removal...")
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=args.denoise_nb_neighbors,
            std_ratio=args.denoise_std_ratio
        )
        pcd_denoise = pcd.select_by_index(ind)

        # 保存去噪后的点云
        denoise_ply_path = output_directory / f"{file_stem}_denoised.ply"
        o3d.io.write_point_cloud(str(denoise_ply_path), pcd_denoise)
        print(f"Denoised point cloud saved to {denoise_ply_path}")

        # 保存为GLB格式
        points = np.asarray(pcd_denoise.points)
        colors = np.asarray(pcd_denoise.colors) if len(pcd_denoise.colors) > 0 else None
        if colors is not None and colors.dtype != np.uint8:
            colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)

        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
        scene = trimesh.Scene()
        scene.add_geometry(cloud)
        glb_path = output_directory / f"{file_stem}.glb"
        scene.export(str(glb_path))
        print(f"GLB file saved to {glb_path}")
    else:
        pcd_denoise = pcd

    # 15. 可视化
    if args.visualize:
        # 显示RGB图像并添加鼠标回调
        cv2.imshow("RGB Image", left_img_rect)
        cv2.setMouseCallback("RGB Image", mouse_callback, xyz_map)
        print("Click on the RGB image to see depth. Press any key to continue to 3D visualization.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 3D点云可视化
        print("Visualizing point cloud. Press 'Q' to exit.")
        o3d.visualization.draw_geometries([pcd_denoise])

    print("3D reconstruction completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FoundationStereo 3D Reconstruction')

    # 模型参数
    parser.add_argument('--ckpt_dir', help="path to FoundationStereo checkpoint",
                        default="pretrained/model_best_bp2.pth")  # 用户需要提供模型路径
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of GRU iterations during forward pass')
    parser.add_argument('--hiera', type=int, default=0,
                        help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='downsize the image by scale, must be <=1')

    # 输出选项
    parser.add_argument("-sn", "--save_name", default="disparity",
                        help="save name for output files")
    parser.add_argument('--output_directory', help="directory to save output",
                        default="./output/")

    # 图像输入
    parser.add_argument("--left_img", "-l",
                        default=r"/root/code/C++/TensorRTTemplate/left.png",
                        help="path to left image")
    parser.add_argument("--right_img", "-r",
                        default=r"/root/code/C++/TensorRTTemplate/right.png",
                        help="path to right image")
    parser.add_argument("-p", "--param_file", default=r"param/zed1280.yaml",
                        help="path to camera calibration file (K.txt for FoundationStereo, or YAML/XML for stereo calibration)")

    # 可视化选项
    parser.add_argument("--show_rectified", action='store_true', default=True,
                        help="show rectified left and right images")
    parser.add_argument("--visualize", action='store_true', default=True,
                        help="visualize the point cloud")

    # 点云处理选项
    parser.add_argument("--z_far", type=float, default=10.0,
                        help="max depth to clip in point cloud")
    parser.add_argument("--remove_invisible", type=int, default=0,
                        help="remove non-overlapping observations between left and right images from point cloud")

    # 点云去噪选项
    parser.add_argument("--denoise", action='store_true', default=True,
                        help="apply statistical outlier removal to point cloud")
    parser.add_argument("--denoise_nb_neighbors", type=int, default=20,
                        help="number of neighbors for statistical outlier removal")
    parser.add_argument("--denoise_std_ratio", type=float, default=2.0,
                        help="standard deviation ratio for statistical outlier removal")

    args = parser.parse_args()
    main(args)
