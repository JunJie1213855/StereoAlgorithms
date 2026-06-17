"""
Fast-ACVNet 立体视觉三维重建流程:

1. 双目标定算法 (与 s2m2 一致):
    a. 获得参数 K_l, K_r, D_l, D_r, R_l, R_r, P_l, P_r, Q, 相机类型, 图像尺寸
    b. 使用 xml 或者 yaml 文件将其保存

2. 使用本三维重建流程:
    a. 修改 param_file 成你自己的 xml 或者 yaml 文件路径
    b. 修改输入的图像 left_img, right_img
    c. 开始运行
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

import stereoconfig
from models import __models__

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------------------------------------
# 鼠标回调: 点击图像输出 3D 距离 (m)
# ----------------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 注意: param 传入的是重投影出的 points_3d
        depth = np.sqrt(
            param[y, x, 0] * param[y, x, 0]
            + param[y, x, 1] * param[y, x, 1]
            + param[y, x, 2] * param[y, x, 2]
        )
        print('坐标 ({}, {}) 的距离为: {} m'.format(x, y, depth))


# ----------------------------------------------------------------------------
# 模型加载: 与 predict_stereo.py / test_mid.py 保持一致
# ----------------------------------------------------------------------------
def load_model(args):
    # 模型结构: Fast_ACVNet 或 Fast_ACVNet_plus
    if args.model not in __models__:
        raise ValueError(
            "model 必须是 {} 中的一个, 但得到 '{}'".format(list(__models__.keys()), args.model)
        )

    model = __models__[args.model](args.maxdisp, False)
    model = nn.DataParallel(model)
    model.cuda()

    print('loading model {}'.format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location='cuda:0')
    model.load_state_dict(state_dict['model'])
    model.eval()
    return model


# ----------------------------------------------------------------------------
# 预处理: 与 predict_stereo.py 一致 —— PIL → 右下角裁到 32 倍数 → ToTensor + Normalize
# ----------------------------------------------------------------------------
def preprocess(left_img: np.ndarray, right_img: np.ndarray, factor: int = 32):
    """读入的 BGR/HWC uint8 numpy → (1, 3, H_pad, W_pad) cuda float tensor"""
    left_pil = Image.fromarray(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    right_pil = Image.fromarray(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))

    w, h = left_pil.size
    wi, hi = (w // factor + 1) * factor, (h // factor + 1) * factor

    # 从右下角裁剪, 推理完成后再用 pred_disp[:, hi - h:, wi - w:] 还原
    left_pil = left_pil.crop((w - wi, h - hi, w, h))
    right_pil = right_pil.crop((w - wi, h - hi, w, h))

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    left_tensor = transforms.Compose([transforms.ToTensor(), normalize])(left_pil)
    right_tensor = transforms.Compose([transforms.ToTensor(), normalize])(right_pil)

    left_tensor = left_tensor.unsqueeze(0).cuda()
    right_tensor = right_tensor.unsqueeze(0).cuda()
    return left_tensor, right_tensor, (h, w), (hi, wi)


# ----------------------------------------------------------------------------
# 推理: 输出裁回原图尺寸的视差数组 (H, W) float32
# ----------------------------------------------------------------------------
def inference(left_img: np.ndarray, right_img: np.ndarray, model, args):
    left_t, right_t, (h, w), (hi, wi) = preprocess(left_img, right_img, factor=32)
    print(f"original image size: img_height({h}), img_width({w})")
    print(f"padded   image size: img_height({hi}), img_width({wi})")

    with torch.no_grad():
        with torch.amp.autocast(enabled=args.fp16, device_type=device.type, dtype=torch.float16):
            pred_disp = model(left_t, right_t)[-1]
        # 裁回原图尺寸
        pred_disp = pred_disp[:, hi - h:, wi - w:]

    pred_disp_np = pred_disp.squeeze().cpu().float().numpy().astype(np.float32)
    return pred_disp_np


# ----------------------------------------------------------------------------
# 保存可视化的视差图 (PNG, jet 色表)
# ----------------------------------------------------------------------------
def save_visualization(disp: np.ndarray, output_path: Path, save_name: str):
    output_path.mkdir(parents=True, exist_ok=True)

    valid = disp > 0
    if valid.any():
        d_min = float(disp[valid].min())
        d_max = float(disp[valid].max())
    else:
        d_min, d_max = 0.0, 1.0

    disp_vis = np.zeros_like(disp, dtype=np.uint8)
    disp_vis[valid] = ((disp[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    plt.imsave(output_path / f"{save_name}.png", disp_vis, cmap='jet')
    # _masked 版: 与 s2m2 行为一致, 在无效区域保持 0
    plt.imsave(output_path / f"{save_name}_masked.png", disp_vis, cmap='jet')


# ----------------------------------------------------------------------------
# 生成 ASCII 格式的 PLY 点云 (x y z r g b)
# ----------------------------------------------------------------------------
def create_ply(points_3d: np.ndarray, image: np.ndarray, filename: str):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    flat_points = points_3d.reshape(h * w, 3)
    flat_colors = image.reshape(h * w, 3)
    output_points = np.concatenate([flat_points, flat_colors.astype(flat_points.dtype)], axis=1)
    print("max z =", float(np.max(points_3d[:, :, 2])))

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
        f.write(ply_header % dict(vert_num=output_points.shape[0]))
        np.savetxt(f, output_points, '%f %f %f %d %d %d')


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main(args: argparse.Namespace):
    print("to obtain the parameters!")
    config = stereoconfig.stereoCamera(args.param_file)
    left_img: np.ndarray = cv2.imread(args.left_img)
    right_img: np.ndarray = cv2.imread(args.right_img)

    # 畸变矫正 + 立体校正
    left_img, right_img = config.rectify(left_img, right_img)
    if args.showrectified:
        print("show the rectified pictures !")
        catimg = config.cat(left_img, right_img)
        cv2.namedWindow("catimage", cv2.WINDOW_NORMAL)
        cv2.imshow("catimage", catimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    model = load_model(args)

    # 推理得到视差
    disp: np.ndarray = inference(left_img, right_img, model, args)

    # 保存可视化的视差图
    output_directory = Path(args.output_directory)
    save_visualization(disp, output_directory, args.save_name)
    print(f"disparity map saved to {output_directory}")

    # 三维重建
    points_3d = cv2.reprojectImageTo3D(disp.astype(np.float32), config.Q, None, False, cv2.CV_32FC1)
    # 仅保留有效 (z > 0) 的点
    valid_mask = (points_3d[:, :, 2] > 0) & np.isfinite(points_3d[:, :, 2])
    points_3d[~valid_mask] = 0

    # 保存 PLY
    create_ply(points_3d, left_img, args.ply_path)
    print(f"point cloud saved to {args.ply_path}")

    # 统计滤波 + 可视化
    pcd = o3d.io.read_point_cloud(args.ply_path)
    if len(pcd.points) > 0:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

    cv2.namedWindow("rgb image", cv2.WINDOW_NORMAL)
    cv2.imshow("rgb image", left_img)
    cv2.setMouseCallback("rgb image", mouse_callback, points_3d)
    o3d.visualization.draw_geometries([pcd])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast-ACVNet 3D Reconstruction')
    # 模型与权重
    parser.add_argument('--model', default='Fast_ACVNet', choices=__models__.keys(),
                        help='select a model structure')
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--loadckpt', default='pretrained/FastACV/generalization.ckpt',
                        help='load the weights from a specific checkpoint')
    parser.add_argument('--fp16', action='store_true', help='use FP16 autocast in inference')

    # 立体匹配输入
    # parser.add_argument("-l", "--left_img", default='data/rect_left.png',
    #                     help='left image path (raw, undistortion happens via param_file)')
    # parser.add_argument("-r", "--right_img", default='data/rect_right.png',
    #                     help='right image path')
    # parser.add_argument("-p", "--param_file", default='param/zed1280.yaml',
    #                     help='stereo calibration yaml file')
    parser.add_argument("--left_img","-l",default=r"/root/code/python/StereoMatch/StereoAlgorithms/example/1920x1080/left.jpg")
    parser.add_argument("--right_img","-r",default=r"/root/code/python/StereoMatch/StereoAlgorithms/example/1920x1080/right.jpg")
    parser.add_argument("-p","--param_file",default=r"/root/code/python/StereoMatch/StereoAlgorithms/example/1920x1080/zed.yaml")
    
    parser.add_argument("--showrectified", default=True, help='是否显示矫正的左右图')

    # 输出
    parser.add_argument("-sn", "--save_name", default='disparity',
                        help='file stem for saved disparity.png / _masked.png')
    parser.add_argument('--output_directory', default='./demo-output/',
                        help='directory to save disparity visualization')
    parser.add_argument("--ply_path", default='./data/3d.ply',
                        help='output point cloud PLY path')

    args = parser.parse_args()
    main(args)
