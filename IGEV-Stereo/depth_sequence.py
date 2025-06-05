"""
    视差转深度图,用于稠密重建
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
import os
import cv2
import time


# 转换为深度图
def disp2depth(disp : np.ndarray,Q : np.ndarray):
    points_map = cv2.reprojectImageTo3D(disp,Q)
    depth = points_map[:,:,2]
    return depth

# 深度图可视化
def disp_show(disp:np.ndarray):
    disp_norm = (disp - np.min(disp)) / (np.max(disp) - np.min(disp)) * 255
    disp_vis = np.round(disp_norm).astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_vis,cv2.COLORMAP_INFERNO)
    return disp_vis

def inference(left_img:np.ndarray,right_img:np.ndarray,model):
    with torch.no_grad():
        left_img = cv2.cvtColor(left_img,cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img,cv2.COLOR_BGR2RGB)
        ## 读取与矫正
        left_rected = torch.from_numpy(left_img).permute(2, 0, 1).float()[None].to(DEVICE)
        right_rected = torch.from_numpy(right_img).permute(2, 0, 1).float()[None].to(DEVICE)
        ## 开始运行
        padder = InputPadder(left_rected.shape, divis_by=32)
        left_rected, right_rected = padder.pad(left_rected, right_rected)
        start = time.time()
        disp = model(left_rected, right_rected, iters=args.valid_iters, test_mode=True)
        end = time.time()
        print("the consume time :",end - start)
        disp = disp.cpu().numpy()
        disp = padder.unpad(disp)
    return disp.squeeze()

# 定义鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 如果检测到左键点击
        depth = param[y,x]  # 获取灰度图像中的像素值
        print('坐标 ({}, {}) 的深度值为: {}'.format(x, y, depth))  # 打印灰度值

def main(args:argparse.Namespace):
    ## 参数获取
    print("to obtain the parameters!")
    config = stereoconfig.stereoCamera(args.param_file)
    print("output_directory : ",args.output_directory)
    ## 视频
    left_image_name = os.listdir(args.left_sequence)
    right_image_name = os.listdir(args.right_sequence)
    "图像的绝对路径"
    left_image_path = [os.path.join(args.left_sequence,name) for name in  left_image_name]
    right_image_path = [os.path.join(args.right_sequence,name) for name in  right_image_name]
    ## model
    # 模型
    print("load the igve stereo matching model !")
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.to(DEVICE)
    model.eval()
    # 运行
    for i in tqdm(range(1,len(left_image_path))):
        left_frame = cv2.imread(left_image_path[i + 1000])
        right_frame = cv2.imread(right_image_path[i + 1000])
        ## 依靠参数文件的图像尺寸resize
        if left_frame.shape[0] != config.height:
            left_frame = cv2.resize(left_frame,dsize=(config.width,config.height))
            right_frame = cv2.resize(right_frame,dsize=(config.width,config.height))

        ## 畸变矫正
        left_img,right_img = config.rectify(left_frame,right_frame)
        # if args.showrecitied==1:
        #     print("show the rectified picures !")
        #     cv2.imshow("left rectified",left_img)
        #     cv2.imshow("right rectified",right_img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        
        ## 获取视差
        disp:np.ndarray = inference(left_img,right_img,model)
        ## 深度图
        depth = disp2depth(disp,config.Q)
        if args.rgbd_save ==1:
            ## 将超过阈值 10m(sqare cm) 的深度转换为 0
            # depth[depth > 15000] = 0
            # depth[depth < 600] = 0
            # 检查目录是否存在
            if not os.path.exists(os.path.join(args.output_directory,"rgb")):
                os.mkdir(os.path.join(args.output_directory,"rgb"))
            if not os.path.exists(os.path.join(args.output_directory,"right")):
                os.mkdir(os.path.join(args.output_directory,"right"))
            if not os.path.exists(os.path.join(args.output_directory,"depth")):
                os.mkdir(os.path.join(args.output_directory,"depth"))
            # rgb图像保存
            cv2.imwrite(os.path.join(os.path.join(args.output_directory,"rgb"),f"{i}.jpg"),left_frame)
            # right
            cv2.imwrite(os.path.join(os.path.join(args.output_directory,"right"),f"{i}.jpg"),right_frame)
            # depth图像保存
            cv2.imwrite(os.path.join(os.path.join(args.output_directory,"depth"),f"{i}.pfm"),depth)
        else:
            disp_vis = disp_show(disp)
            # depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
            # depth_vis = np.round(depth_norm).astype(np.uint8)
            depth_vis = disp_show(depth)
            cv2.namedWindow("original image",cv2.WINDOW_NORMAL)
            cv2.namedWindow("depth image",cv2.WINDOW_NORMAL)
            cv2.namedWindow("disparity visualization",cv2.WINDOW_NORMAL)
            cv2.imshow("original image",left_img)
            cv2.imshow("depth image",depth_vis)
            cv2.imshow("disparity visualization",disp_vis)
            # cv2.setMouseCallback("original image",mouse_callback,depth)
            # esc 退出
            if cv2.waitKey(1)==27 :
                break



### 3D 重建
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 预训练模型和一些选项
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained/middlebury.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=128, help="max disp of geometry encoding volume")
    
    ## 添加的选项
    parser.add_argument('--rgbd_save', help="directory to save output", default=1,type=int)
    parser.add_argument('--output_directory', help="directory to save output", default="E:/code/python/IGEV_stereo/IGEV-Stereo/rgbd")
    parser.add_argument("--showrecitied",default=0,help="是否显示矫正的左右图")
    
    parser.add_argument("--param_file",default="./param/euroc.yaml")
    parser.add_argument("-l","--left_sequence",default="E:\\dataset\\slam\\visual\\Euroc\\mav0\\mav0\\cam0\\data")
    parser.add_argument("-r","--right_sequence",default="E:\\dataset\\slam\\visual\\Euroc\\mav0\\mav0\\cam1\\data")

    ## 命令行参数
    args = parser.parse_args()
    main(args)