import cv2
import numpy as np
import torch
from torch import nn
import time
import argparse
from matplotlib import pyplot as plt
import os
from pathlib import Path
def creatp_ply(points_3d:np.ndarray,image:np.ndarray, filename :str):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    output_points = []
    # 生成 ply点云文件
    max_z = np.max(points_3d[:,:,2])
    print(max_z)
    # depth = np.zeros((points_3d.shape[0],points_3d.shape[1]),np.float32)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
        # if(mask[row,col,2]!=0):
                # if depth more than 5m ,
                # if points_3d[row,col,2] > 12 or points_3d[row,col,2]< 0.6:
                #     # depth[row,col]  = points_3d[row,col,2]
                #     continue
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
    # cv2.imshow("depth",depth)
    # cv2.waitKey()
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

def Point3DCallBack(event, x, y, flags, param):
    # 如果检测到左键点击
    if event == cv2.EVENT_LBUTTONDOWN: 
        # 获取三维点的距离, mm 的需要除以 1000 
        depth = np.sqrt(param[y,x,0] * param[y,x,0] +  param[y,x,1]*param[y,x,1] + param[y,x,2]*param[y,x,2])
        print('坐标 ({}, {}) 的距离为: {}'.format(x, y, depth)," m")  # 打印灰度值


def inference(left_img:np.ndarray,right_img:np.ndarray,model : nn.Module,args:argparse.Namespace):

    print("check the output save directory")
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    print("begin to caculate !")
    with torch.no_grad():
        left_img = cv2.cvtColor(left_img,cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img,cv2.COLOR_BGR2RGB)
        ## 读取与矫正
        left_rected = torch.from_numpy(left_img).permute(2, 0, 1).float()[None].to('cuda')
        right_rected = torch.from_numpy(right_img).permute(2, 0, 1).float()[None].to('cuda')
        ## 开始运行
        # padder = InputPadder(left_rected.shape, divis_by=32)
        # left_rected, right_rected = padder.pad(left_rected, right_rected)
        start = time.time()
        disp = model(left_rected, right_rected, iters=args.valid_iters, test_mode=True)
        end = time.time()
        print("the consume time :",end - start)
        disp = disp.cpu().numpy()
        # disp = padder.unpad(disp)
        ## 保存
        file_stem = args.save_name
        filename = os.path.join(output_directory, f"{file_stem}.png")
        ## 用plt的颜色映射表保存
        plt.imsave(output_directory / f"{file_stem}.png", disp.squeeze(), cmap='jet')
    
    print("the igev caculation end !")
    return disp.squeeze()