"""
立体视觉三维重建算法流程:
1.使用双目标定算法:
    a.获得参数 K_l,K_r,D_l,D_r,R_l,R_r,P_l,P_r,Q,相机类型，图像尺寸
    b.使用xml或者yaml文件将其保存

2.使用该三维重建方法
    a.修改 param_file 成你自己的xml或者yaml文件的路径
    b.修改输入的图像 left_img,right_img
    c.开始运行

注意: 本文使用Selective-IGEV模型进行立体匹配
"""
import torch
import torch.nn.functional as F
import cv2
import time
import stereoconfig
import open3d as o3d
import sys
sys.path.append('core')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import logging
import trimesh


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


def toOpen3dCloud(points, colors=None):
    """转换为Open3D点云格式"""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return cloud


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


def inference(left_img: np.ndarray, right_img: np.ndarray, model):
    """
    使用Selective-IGEV模型进行立体匹配推理

    Args:
        left_img: 左图像 (BGR格式)
        right_img: 右图像 (BGR格式)
        model: Selective-IGEV模型

    Returns:
        disp: 视差图 (numpy数组)
    """
    print("Load Selective-IGEV model for inference!")
    model.eval()

    with torch.no_grad():
        # 转换为RGB
        left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # 转换为tensor
        left_tensor = torch.from_numpy(left_img_rgb).permute(2, 0, 1).float()[None].to(DEVICE)
        right_tensor = torch.from_numpy(right_img_rgb).permute(2, 0, 1).float()[None].to(DEVICE)

        # Padding确保尺寸是32的倍数
        padder = InputPadder(left_tensor.shape, divis_by=32)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

        print(f"Input shape: {left_tensor.shape}")
        # 推理
        start = time.time()
        model(left_tensor, right_tensor)
        end = time.time()
        print(f"Inference time: {end - start :.5f}s")
        disp = model(left_tensor, right_tensor)
        # 转换回numpy
        disp = disp.cpu().numpy()
        disp = padder.unpad(disp)

    return disp.squeeze()


def main(args: argparse.Namespace):
    """主函数"""

    # 1. 加载相机参数
    print("Loading stereo camera parameters...")
    config = stereoconfig.stereoCamera(args.param_file)

    # 2. 读取左右图像
    print(f"Loading images...")
    left_img = cv2.imread(args.left_img)
    right_img = cv2.imread(args.right_img)

    if left_img is None or right_img is None:
        raise ValueError("Failed to load images. Please check the image paths.")

    # 3. 畸变矫正
    print("Performing rectification...")
    left_img_rect, right_img_rect = config.rectify(left_img, right_img)

    if args.show_rectified:
        print("Showing rectified images...")
        cat_img = config.cat(left_img_rect, right_img_rect)
        cv2.namedWindow("rectified_stereo", cv2.WINDOW_NORMAL)
        cv2.imshow("rectified_stereo", cat_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 4. 加载Selective-IGEV模型
    print("Loading Selective-IGEV model...")
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth"), "Checkpoint file must be .pth"
        print(f"Loading checkpoint from {args.restore_ckpt}")
        model.load_state_dict(torch.load(args.restore_ckpt, map_location=DEVICE))
        print("Checkpoint loaded successfully")

    model = model.module
    model.to(DEVICE)
    model.eval()

    # 5. 创建输出目录
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    # 6. 立体匹配获取视差
    print("Computing disparity...")
    disp = inference(left_img_rect, right_img_rect, model)

    # 7. 保存视差图
    file_stem = args.save_name
    vis_disp = vis_disparity(disp)
    plt.imsave(output_directory / f"{file_stem}.png", vis_disp)
    cv2.imwrite(str(output_directory / f"{file_stem}_raw.png"), (disp / disp.max() * 255).astype(np.uint8))
    print(f"Disparity map saved to {output_directory / f'{file_stem}.png'}")

    # 8. 三维重建 - 使用Q矩阵
    print("Reconstructing 3D points...")
    points_3d = cv2.reprojectImageTo3D(disp, config.Q, None, False, cv2.CV_32FC1)

    # 9. 创建点云
    print("Creating point cloud...")
    ply_path = os.path.join(output_directory, f"{file_stem}.ply")
    create_ply(points_3d, left_img_rect, ply_path)
    print(f"Point cloud saved to {ply_path}")

    # 10. 读取点云并进行去噪
    print("Loading point cloud for denoising...")
    pcd = o3d.io.read_point_cloud(ply_path)

    # 统计滤波去噪
    if args.denoise:
        print("Applying statistical outlier removal...")
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=args.denoise_nb_neighbors,
            std_ratio=args.denoise_std_ratio
        )
        pcd_denoise = pcd.select_by_index(ind)

        # 保存去噪后的点云
        denoise_ply_path = os.path.join(output_directory, f"{file_stem}_denoised.ply")
        o3d.io.write_point_cloud(denoise_ply_path, pcd_denoise)
        print(f"Denoised point cloud saved to {denoise_ply_path}")

        # 保存为GLB格式
        points = np.asarray(pcd_denoise.points)
        colors = np.asarray(pcd_denoise.colors) if len(pcd_denoise.colors) > 0 else None
        if colors is not None and colors.dtype != np.uint8:
            colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)

        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
        scene = trimesh.Scene()
        scene.add_geometry(cloud)
        glb_path = os.path.join(output_directory, f"{file_stem}.glb")
        scene.export(glb_path)
        print(f"GLB file saved to {glb_path}")
    else:
        pcd_denoise = pcd

    # 11. 可视化
    if args.visualize:
        # 显示RGB图像并添加鼠标回调
        cv2.imshow("RGB Image", left_img_rect)
        cv2.setMouseCallback("RGB Image", mouse_callback, points_3d)
        print("Click on the RGB image to see depth. Press any key to continue to 3D visualization.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 3D点云可视化
        print("Visualizing point cloud. Press 'Q' to exit.")
        o3d.visualization.draw_geometries([pcd_denoise])

    print("3D reconstruction completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selective-IGEV 3D Reconstruction')

    # 模型参数
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default='pretrained/middlebury_finetune.pth')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during forward pass')
    parser.add_argument('--max_disp', type=int, default=192,
                        help='max disp of geometry encoding volume')

    # 输出选项
    parser.add_argument("-sn", "--save_name", default="disparity",
                        help="save name for output files")
    parser.add_argument('--output_directory', help="directory to save output",
                        default="./output/")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # 图像输入
    parser.add_argument("--left_img", "-l",
                        default=r"./left.png")
    parser.add_argument("--right_img", "-r",
                        default=r"./right.png")
    parser.add_argument("-p", "--param_file", default=r"param/zed1280.yaml",
                        help="path to stereo camera calibration file (YAML/XML)")

    # 可视化选项
    parser.add_argument("--show_rectified", action='store_true', default=True,
                        help="show rectified left and right images")
    parser.add_argument("--visualize", action='store_true', default=True,
                        help="visualize the point cloud")

    # 点云去噪选项
    parser.add_argument("--denoise", action='store_true', default=True,
                        help="apply statistical outlier removal to point cloud")
    parser.add_argument("--denoise_nb_neighbors", type=int, default=20,
                        help="number of neighbors for statistical outlier removal")
    parser.add_argument("--denoise_std_ratio", type=float, default=2.0,
                        help="standard deviation ratio for statistical outlier removal")

    # Selective-IGEV 架构参数
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"],
                        default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2,
                        help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4,
                        help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2,
                        help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true',
                        help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3,
                        help="number of hidden GRU levels")
    parser.add_argument("--test_mode", default=True, help="the mode of this model")

    args = parser.parse_args()
    main(args)
