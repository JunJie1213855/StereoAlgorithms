"""
双目三维重建算法 - 使用 GGEV 立体匹配
流程:
1. 使用双目标定算法获得相机参数
2. 加载 GGEV 立体匹配模型
3. 畸变矫正
4. 计算视差图
5. 三维重建生成点云
"""
import sys
sys.path.append('core_rt')

import argparse
import numpy as np
import torch
from pathlib import Path
from core_rt.ggev_stereo import GGEVStereo
from core_rt.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import time
import open3d as o3d
import stereoconfig

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def movelight(img: np.ndarray):
    """移除图像中的高光区域"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 225, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return dst


def create_ply(points_3d: np.ndarray, image: np.ndarray, filename):
    """创建 PLY 点云文件"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_points = []
    
    print(f"Max depth: {np.max(points_3d[:,:,2])}")
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            """
            x y z r g b
            """
            output_points.append(list([
                points_3d[row, col, 0],
                points_3d[row, col, 1],
                points_3d[row, col, 2],
                image[row, col, 0],
                image[row, col, 1],
                image[row, col, 2]
            ]))
    
    # 写入 PLY 文件
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
    """鼠标回调函数，显示三维点的距离"""
    if event == cv2.EVENT_LBUTTONDOWN:
        depth = np.sqrt(param[y, x, 0] * param[y, x, 0] + 
                        param[y, x, 1] * param[y, x, 1] + 
                        param[y, x, 2] * param[y, x, 2])
        print(f'坐标 ({x}, {y}) 的距离为: {depth} m')


def inference(left_img: np.ndarray, right_img: np.ndarray, args: argparse.Namespace):
    """使用 GGEV 模型进行立体匹配推理"""
    print("Loading GGEV stereo matching model...")
    model = torch.nn.DataParallel(GGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, map_location=DEVICE))
    model = model.module
    model.to(DEVICE)
    model.eval()

    print("Checking output save directory")
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    print("Starting stereo matching...")
    with torch.no_grad():
        # 转换颜色空间
        left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        # 转换为张量
        left_tensor = torch.from_numpy(left_img_rgb).permute(2, 0, 1).float()[None].to(DEVICE)
        right_tensor = torch.from_numpy(right_img_rgb).permute(2, 0, 1).float()[None].to(DEVICE)
        
        # 填充图像
        padder = InputPadder(left_tensor.shape, divis_by=32)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
        for i in range(15):
            _ = model(left_tensor, right_tensor, iters=args.valid_iters, test_mode=True)
        # 计算视差
        start = time.time()
        for i in range(15):
            disp = model(left_tensor, right_tensor, iters=args.valid_iters, test_mode=True)
        end = time.time()
        print(f"Time consumed: {(end - start) / 15:.3f} seconds")
        
        # 处理结果
        disp = disp.cpu().numpy()
        disp = padder.unpad(disp)
        
        # 保存视差图
        file_stem = args.save_name
        plt.imsave(output_directory / f"{file_stem}.png", disp.squeeze(), cmap='jet')
        
        if args.save_numpy:
            np.save(output_directory / f"{file_stem}.npy", disp.squeeze())
    
    print("GGEV stereo matching completed!")
    return disp.squeeze()


def main(args: argparse.Namespace):
    """主函数：执行双目三维重建流程"""
    
    # 1. 加载相机参数
    print("Loading stereo camera parameters...")
    config = stereoconfig.stereoCamera(args.param_file)
    
    # 2. 读取左右图像
    print("Loading images...")
    left_img = cv2.imread(args.left_img)
    right_img = cv2.imread(args.right_img)
    
    if left_img is None or right_img is None:
        raise ValueError("Failed to load images. Please check the image paths.")
    
    # 3. 畸变矫正
    print("Performing distortion correction...")
    left_img, right_img = config.rectify(left_img, right_img)
    
    # 4. 显示矫正后的图像（可选）
    if args.show_rectified:
        print("Displaying rectified images...")
        catimg = config.cat(left_img, right_img)
        cv2.namedWindow("Rectified Images", cv2.WINDOW_NORMAL)
        cv2.imshow("Rectified Images", catimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 5. 计算视差图
    print("Computing disparity map...")
    disp = inference(left_img, right_img, args)
    
    # 6. 三维重建
    print("Performing 3D reconstruction...")
    points_3d = cv2.reprojectImageTo3D(disp, config.Q, None, False, cv2.CV_32FC1)
    
    # 7. 生成点云文件
    print("Creating point cloud file...")
    create_ply(points_3d, left_img, args.ply_path)
    
    # 8. 读取并过滤点云
    print("Loading and filtering point cloud...")
    pcd = o3d.io.read_point_cloud(args.ply_path)
    
    # 统计离群点滤波
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_pcd = pcd.select_by_index(ind)
    
    print(f"Original points: {len(pcd.points)}, After filtering: {len(inlier_pcd.points)}")
    
    # 9. 可视化
    if args.show_3d:
        print("Starting 3D visualization...")
        
        # 显示 RGB 图像并设置鼠标回调
        cv2.imshow("RGB Image", left_img)
        cv2.setMouseCallback("RGB Image", mouse_callback, points_3d)
        
        # 使用 Open3D 显示点云
        o3d.visualization.draw_geometries([inlier_pcd])
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("3D reconstruction completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GGEV Stereo Matching and 3D Reconstruction')
    
    # GGEV 模型参数
    parser.add_argument('--restore_ckpt', help="restore checkpoint", 
                       default='pretrained/kitti.pth')
    parser.add_argument('--save_numpy', action='store_true', 
                       help='save output as numpy arrays')
    parser.add_argument('--mixed_precision', action='store_true', default=False, 
                       help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', 
                       choices=['float16', 'bfloat16', 'float32'], 
                       help='Choose precision type')
    parser.add_argument('--valid_iters', type=int, default=8, 
                       help='number of flow-field updates during forward pass')
    parser.add_argument('--encoder', default='vits', 
                       help="DepthAnything V2 encoder")

    # GGEV 架构参数
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, 
                       help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, 
                       help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, 
                       help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, 
                       help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=1, 
                       help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, 
                       help="max disp range")
    
    # 3D 重建参数
    parser.add_argument('--save_name', default="disparity", 
                       help="save name for output files")
    parser.add_argument('--output_directory', help="directory to save output", 
                       default="./reconstruction_output/")
    parser.add_argument('--show_rectified', action='store_true', 
                       help='show rectified left and right images')
    parser.add_argument('--show_3d', action='store_true', 
                       help='show 3D point cloud visualization')

    # 输入输出路径
    parser.add_argument('--left_img', '-l', 
                       default=r"/mnt/d/dataset/CameraCalib/stereoexample_github/stereoexample_zed/test/1280/left/Explorer_HD720_SN21067_21-14-47.jpg",
                       help="path to left image")
    parser.add_argument('--right_img', '-r', 
                       default=r"/mnt/d/dataset/CameraCalib/stereoexample_github/stereoexample_zed/test/1280/right/Explorer_HD720_SN21067_21-14-47.jpg",
                       help="path to right image")
    parser.add_argument('--param_file', '-p', 
                       default="param/zed1280.yaml",
                       help="path to stereo camera parameters file (xml or yaml)")
    parser.add_argument('--ply_path', 
                       default="./reconstruction_output/3d.ply",
                       help="path to save PLY point cloud file")
    
    args = parser.parse_args()
    
    main(args)