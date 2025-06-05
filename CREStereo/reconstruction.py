import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
from nets import Model
import argparse
import open3d as o3d
import stereoconfig
from matplotlib import pyplot as plt
import os
device = 'cuda'
from pathlib import Path
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

# 定义鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    # 如果检测到左键点击
    if event == cv2.EVENT_LBUTTONDOWN: 
        # 获取三维点的距离, mm 的需要除以 1000 
        depth = np.sqrt(param[y,x,0] * param[y,x,0] +  param[y,x,1]*param[y,x,1] + param[y,x,2]*param[y,x,2])
        print('坐标 ({}, {}) 的距离为: {}'.format(x, y, depth)," m")  # 打印灰度值


	# # disp = cv2.resize(disp,None,fx=2,fy=2)
	# "存储pfm文件"
	# cv2.imwrite("./disp/disp.pfm",disp)
	# disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
	# disp_vis = disp_vis.astype("uint8")
	# print(disp_vis.shape)
	# disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
	# cv2.imwrite("./disp/disp.png",disp_vis)
	# cv2.imshow("output", disp_vis)
	# print(disp_vis.shape)
	# cv2.waitKey(0)
def inference(left_img:np.ndarray, right_img:np.ndarray, model,args:argparse.Namespace):

    #  size 
    in_h, in_w = left_img.shape[:2]
    eval_h,eval_w = (in_h - in_h % 8 ,in_w - in_w % 8)
    t = float(in_w) / float(eval_w)
# # Resize image in case the GPU memory overflows
    assert eval_h%8 == 0, "input height should be divisible by 8"
    assert eval_w%8 == 0, "input width should be divisible by 8"
    imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    print("Model Forwarding...")
    imgL = imgL.transpose(2, 0, 1)
    imgR = imgR.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])

    imgL = torch.tensor(imgL.astype("float32")).to(device)
    imgR = torch.tensor(imgR.astype("float32")).to(device)

    imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
    imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
    start = time.time()
	# print(imgR_dw2.shape)
    with torch.inference_mode():
        pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=args.n_iter, flow_init=None)
        pred_flow = model(imgL, imgR, iters=args.n_iter, flow_init=pred_flow_dw2)
    end = time.time()
    print("cost time",end - start)
    pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()
    disp = cv2.resize(pred_disp, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
    
    ## 保存
    output_directory = Path(args.output_directory)
    if not output_directory.exists():
        output_directory.mkdir(exist_ok=True)
    file_stem = args.save_name
    filename = os.path.join(output_directory, f"{file_stem}.png")
    ## 用plt的颜色映射表保存
    plt.imsave(output_directory / f"{file_stem}.png", disp.squeeze(), cmap='jet')
    return disp 

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
        catimg = config.cat(left_img,right_img)
        cv2.namedWindow("catimage",cv2.WINDOW_NORMAL)
        cv2.imshow("catimage",catimg)
        # cv2.imshow("right rectified",right_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # model
    model = Model(max_disp=args.max_disp, mixed_precision=args.mixed_precision, test_mode=True) 
    model.load_state_dict(torch.load(args.restore_ckpt), strict=True)
    model.to(device)
    model.eval()
    ## 获取视差
    disp:np.ndarray = inference(left_img,right_img,model,args)
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

	


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 预训练模型和一些选项
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="models/crestereo_eth3d.pth")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--max_disp', type=int, default=128, help="max disp of geometry encoding volume")
    parser.add_argument('--n_iter', type=int, default=10, help="max disp of geometry encoding volume")
   
    ## 添加的选项
    parser.add_argument("-sn","--save_name",default="disparity")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
    parser.add_argument("--showrecitied",default=1,help="是否显示矫正的左右图")
    
    # parser.add_argument("-l","--left_img",default="E:\\dataset\\Slam\\visual\\Euroc\\vic\\cam0\\data\\1413393313505760512.png")
    # parser.add_argument("-r","--right_img",default="E:\\dataset\\Slam\\visual\\Euroc\\vic\\cam1\\data\\1413393313505760512.png")
    parser.add_argument("-l","--left_img",default="demo-imgs\\github\\left_1.png")
    parser.add_argument("-r","--right_img",default="demo-imgs\\github\\right_1.png")
    parser.add_argument("-p","--param_file",default=".\\param\\github.yaml")

    parser.add_argument("--ply_path",default="./data/3d.ply")
    
    ## 命令行参数
    args = parser.parse_args()
    main(args)



