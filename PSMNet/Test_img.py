from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image
import stereoconfig
import open3d as o3d
import sys
sys.path.append('core')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import os
from pathlib import Path



def inference(left_recitified : np.ndarray,right_recitified:np.ndarray,args:argparse.Namespace):
    # 预处理准备
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])
    # 转换格式
    imgL_o = cv2.cvtColor(left_recitified,cv2.COLOR_BGR2RGB)
    imgR_o = cv2.cvtColor(right_recitified,cv2.COLOR_BGR2RGB)
    # 转换
    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o)
    # 转换为 16n 尺寸
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0
    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0
    # 填充
    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.model == 'stackhourglass':
        print("model : stackhourglass")
        model = stackhourglass(args.maxdisp)
    elif args.model == 'basic':
        print("model : basic")
        model = basic(args.maxdisp)
    else:
        print('no model')
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()
    if args.loadmodel is not None:
        print('load PSMNet')
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # 准备执行
    model.eval()
    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()     

    start = time.time()
    with torch.no_grad():
        disp = model(imgL,imgR)
    end  = time.time()
    print('cost time = %.2f (s)' %(end - start))

    # 转换回来
    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()
    if top_pad !=0 and right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
    elif top_pad ==0 and right_pad != 0:
            img = pred_disp[:,:-right_pad]
    elif top_pad !=0 and right_pad == 0:
            img = pred_disp[top_pad:,:]
    else:
            img = pred_disp
    
    # 保存
    print("check the output save directory")
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)
    file_stem = args.save_name
    ## 用plt的颜色映射表保存
    plt.imsave(output_directory / f"{file_stem}.png", img, cmap='jet')
    return img
    
# 创建点云文件
def creatp_ply(points_3d:np.ndarray,image:np.ndarray, filename):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    output_points = []
    # 生成 ply点云文件
    max_z = np.max(points_3d[:,:,2])
    print(max_z)
    # depth = np.zeros((points_3d.shape[0],points_3d.shape[1]),np.float32)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
                """
                x y z r g b
                """
                output_points.append(list([points_3d[row,col,0],
                                        points_3d[row,col,1],
                                        points_3d[row,col,2],
                                        image[row,col,0],
                                        image[row,col,1],
                                        image[row,col,2]
                                        ])
                                    )
    ## 写入
    ply_header = \
'''ply
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

# 定义鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    # 如果检测到左键点击
    if event == cv2.EVENT_LBUTTONDOWN: 
        # 获取三维点的距离, mm 的需要除以 1000 
        depth = np.sqrt(param[y,x,0] * param[y,x,0] +  param[y,x,1]*param[y,x,1] + param[y,x,2]*param[y,x,2])
        print('坐标 ({}, {}) 的距离为: {}'.format(x, y, depth)," m")  # 打印灰度值

def main(args:argparse.Namespace):
    ## 参数获取
    print("to obtain the parameters!")
    config = stereoconfig.stereoCamera(args.param_file)
    left_img : np.ndarray = cv2.imread(args.left_img)
    right_img : np.ndarray = cv2.imread(args.right_img)

    ## 依靠参数文件的图像尺寸resize
    if left_img.shape[0] != config.height:
        left_img = cv2.resize(left_img,dsize=(config.width,config.height))
        right_img = cv2.resize(right_img,dsize=(config.width,config.height))
    
    ## 畸变矫正
    left_img,right_img = config.rectify(left_img,right_img)
    if args.showrecitied==1:
        print("show the rectified picures !")
        cv2.imshow("left rectified",left_img)
        cv2.imshow("right rectified",right_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    ## 获取视差
    disp:np.ndarray = inference(left_img,right_img,args)
    ## 三维重建
    points_3d = cv2.reprojectImageTo3D(disp, config.Q,None,False,cv2.CV_32FC1)
    ## 创建ply文件
    creatp_ply(points_3d,left_img, args.ply_path)
    ## 读取
    pcd = o3d.io.read_point_cloud(args.ply_path)
    cl,ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                    std_ratio=2.0)
    cv2.imshow("rgb image",left_img)
    cv2.setMouseCallback("rgb image",mouse_callback,points_3d)
    o3d.visualization.draw_geometries([pcd.select_by_index(ind)])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

### 3D 重建
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--KITTI', default='2012',
                        help='KITTI version')
    parser.add_argument('--loadmodel', default='trained/pretrained_model.tar',
                        help='loading model')                                   
    parser.add_argument("-m",'--model', default='stackhourglass',
                        help='select model')
    parser.add_argument("-md",'--maxdisp', type=int, default=128,
                        help='maxium disparity')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    ## 添加的选项
    parser.add_argument("-sn","--save_name",default="disparity")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
    parser.add_argument("--showrecitied",default=1,help="是否显示矫正的左右图")
    
    parser.add_argument("-p","--param_file",default="./param/test.yaml")
    parser.add_argument("-l","--left_img",default = None,help="the input left image")
    parser.add_argument("-r","--right_img",default=None,help="the input right image")
    # 3D 重建点云文件的保存路径
    parser.add_argument("--ply_path",default="./data/3d.ply")
    
    ## 命令行参数
    args = parser.parse_args()
    main(args)