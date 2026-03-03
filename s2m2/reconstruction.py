"""
立体视觉三维重建算法流程:
1.使用双目标定算法:
    a.获得参数 K_l,K_r,D_l,D_r,R_l,R_r,P_l,P_r,Q,相机类型，图像尺寸
    b.使用xml或者yaml文件将其保存

2.使用该三维重建方法
    a.修改 param_file 成你自己的xml或者yaml文件的路径
    b.修改输入的图像 left_img,right_img
    c.开始运行
注意的是,本文并不教会大家如何配置环境.
"""
import torch
import torch.nn.functional as F
import cv2
import stereoconfig
import open3d as o3d
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
from model.s2m2 import S2M2 as Model
torch.backends.cudnn.benchmark = True
import os
import cv2
import math
from pathlib import Path
from matplotlib import pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    # 如果检测到左键点击
    if event == cv2.EVENT_LBUTTONDOWN: 
        # 获取三维点的距离, mm 的需要除以 1000 
        depth = np.sqrt(param[y,x,0] * param[y,x,0] +  param[y,x,1]*param[y,x,1] + param[y,x,2]*param[y,x,2])
        print('坐标 ({}, {}) 的距离为: {}'.format(x, y, depth)," m")  # 打印灰度值

# 模型加载
def load_model(args):

    if args.model_type == "S":
        feature_channels = 128
        n_transformer = 1 * 1
    elif args.model_type == "M":
        feature_channels = 192
        n_transformer = 1 * 2
    elif args.model_type == "L":
        feature_channels = 256
        n_transformer = 1 * 3
    elif args.model_type == "XL":
        feature_channels = 384
        n_transformer = 1*3
    else:
        print('model type should be one of [S, M, L, XL]')
        exit(1)


    model_path = 'CH' + str(feature_channels) + 'NTR' + str(n_transformer) + '.pth'
    ckpt_path = os.path.join('pretrained', model_path)

    model = Model(feature_channels=feature_channels,
                  dim_expansion=1,
                  num_transformer=n_transformer,
                  use_positivity=True,
                  refine_iter=args.num_refine
                  )
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.my_load_state_dict(checkpoint['state_dict'])
    return model

def image_pad(img, factor):
    with torch.no_grad():
        H,W = img.shape[-2:]

        H_new = math.ceil(H / factor) * factor
        W_new = math.ceil(W / factor) * factor

        pad_h = H_new - H
        pad_w = W_new - W

        p2d = (pad_w//2, pad_w-pad_w//2, 0, 0)
        img_pad = F.pad(img, p2d, "constant", 0)
        #
        p2d = (0,0, pad_h // 2, pad_h - pad_h // 2)
        img_pad = F.pad(img_pad, p2d, "constant", 0)

        img_pad_down = F.adaptive_avg_pool2d(img_pad, output_size=[H // factor, W // factor])
        img_pad = F.interpolate(img_pad_down, size=[H_new, W_new], mode='bilinear')

        h_s = pad_h // 2
        h_e = (pad_h - pad_h // 2)
        w_s = pad_w // 2
        w_e = (pad_w - pad_w // 2)
        if h_e==0 and w_e==0:
            img_pad[:, :, h_s:, w_s:] = img
        elif h_e==0:
            img_pad[:, :, h_s:, w_s:-w_e] = img
        elif w_e==0:
            img_pad[:, :, h_s:-h_e, w_s:] = img
        else:
            img_pad[:, :, h_s:-h_e, w_s:-w_e] = img

        return img_pad

def image_crop(img, img_shape):
    with torch.no_grad():
        H,W = img.shape[-2:]
        H_new, W_new = img_shape

        crop_h = H - H_new
        if crop_h > 0:
            crop_s = crop_h // 2
            crop_e = crop_h - crop_h // 2
            img = img[:,:,crop_s: -crop_e]

        crop_w = W - W_new
        if crop_w > 0:
            crop_s = crop_w // 2
            crop_e = crop_w - crop_w // 2
            img = img[:,:,:, crop_s: -crop_e]

        return img


def inference(imgL, imgR, model, args):
    left_torch = (torch.from_numpy(imgL).permute(-1, 0, 1).unsqueeze(0)).half().to(device)
    right_torch = (torch.from_numpy(imgR).permute(-1, 0, 1).unsqueeze(0)).half().to(device)

    left_torch_pad = image_pad(left_torch, 32)
    right_torch_pad = image_pad(right_torch, 32)

    img_height, img_width = imgL.shape[:2]
    print(f"original image size: img_height({img_height}), img_width({img_width})")

    img_height_pad, img_width_pad = left_torch_pad.shape[2:]
    print(f"padded image size: img_height({img_height_pad}), img_width({img_width_pad})")
    with torch.no_grad():
        with torch.amp.autocast(enabled=True, device_type=device.type, dtype=torch.float16):
            # print(f"pre-run...")
            # _ = model(left_torch_pad, right_torch_pad)
            # T = 1
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # starter.record()
            # for _ in range(T):
                pred_disp, pred_occ, pred_conf = model(left_torch_pad, right_torch_pad)
            #     ender.record()
            #     # WAIT FOR GPU SYNC
            #     torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)

    # print(F"torch avg inference time:{(curr_time)/T/1000}, FPS:{1000*T/(curr_time)}")

    pred_disp = image_crop(pred_disp, (img_height, img_width))
    pred_occ = image_crop(pred_occ, (img_height, img_width))
    pred_conf = image_crop(pred_conf, (img_height, img_width))
    valid_disp = (((pred_conf).cpu().float() >.1)*((pred_conf).cpu().float() >.01)).squeeze().numpy()

    # 后处理
    pred_disp_np = np.ascontiguousarray(pred_disp.squeeze().cpu().float().numpy()).astype(np.float32)
    pred_disp_np_filt = pred_disp_np * valid_disp
    pred_disp_np_filt[~valid_disp] = 65500

    # 可视化
    d_min = pred_disp.min().item()
    d_max = pred_disp.max().item()
    disp_left_vis = (pred_disp - d_min) / (d_max-d_min) * 255
    disp_left_vis = disp_left_vis.cpu().squeeze().numpy().astype("uint8")

    pred_disp_masked = pred_disp_np * valid_disp
    d_min = np.min(pred_disp_masked)
    d_max = np.max(pred_disp_masked)
    disp_left_vis_masked = (pred_disp_masked - d_min) / (d_max-d_min) * 255
    disp_left_vis_masked = disp_left_vis_masked.astype("uint8")


    ## 保存
    print("check the output save directory")
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    file_stem = args.save_name
    ## 用plt的颜色映射表保存
    plt.imsave(output_directory / f"{file_stem}.png", disp_left_vis, cmap='jet')
    plt.imsave(output_directory / f"{file_stem}_masked.png", disp_left_vis_masked, cmap='jet')
    
    # print(np.unique(pred_disp_np_filt))

    return pred_disp_np_filt

# 创建点云文件
def creatp_ply(points_3d:np.ndarray, image:np.ndarray, filename):
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
                # if points_3d[row,col,2] > 12e+3 or points_3d[row,col,2]< 0.6e+3:
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


def main(args:argparse.Namespace):
    ## 参数获取
    print("to obtain the parameters!")
    config = stereoconfig.stereoCamera(args.param_file)
    left_img : np.ndarray = cv2.imread(args.left_img)
    right_img : np.ndarray = cv2.imread(args.right_img)

    ## 畸变矫正
    left_img, right_img = config.rectify(left_img, right_img)
    if args.showrecitied:
        print("show the rectified picures !")
        catimg = config.cat(left_img, right_img)
        cv2.namedWindow("catimage",cv2.WINDOW_NORMAL)
        cv2.imshow("catimage",catimg)
        # cv2.imshow("right rectified",right_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    model = load_model(args).to(device).eval()
    if args.torch_compile:
        model = torch.compile(model)
    ## 获取视差
    disp:np.ndarray = inference(left_img, right_img, model, args)
    ## 三维重建
    points_3d = cv2.reprojectImageTo3D(disp, config.Q, None, False, cv2.CV_32FC1)
    ## 创建ply文件
    creatp_ply(points_3d, left_img, args.ply_path)
    ## 读取
    pcd = o3d.io.read_point_cloud(args.ply_path)
    cl,ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                    std_ratio=2.0)
    cv2.imshow("rgb image",left_img)
    cv2.setMouseCallback("rgb image", mouse_callback, points_3d)
    o3d.visualization.draw_geometries([pcd.select_by_index(ind)])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

### 3D 重建
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='M', type=str,
                        help='select model type: S,M,L,XL')
    parser.add_argument('--num_refine', default=5, type=int,
                        help='number of local iterative refinement')
    parser.add_argument('--torch_compile', action='store_true', help='torch_compile')
    
    ## 添加的选项
    parser.add_argument("-sn","--save_name",default="disparity")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
    parser.add_argument("--showrecitied",default=True,help="是否显示矫正的左右图")

    parser.add_argument("--left_img","-l",default=r"/mnt/d/dataset/CameraCalib/stereoexample_github/stereoexample_zed/test/1280/left/Explorer_HD720_SN21067_21-14-47.jpg")
    parser.add_argument("--right_img","-r",default=r"/mnt/d/dataset/CameraCalib/stereoexample_github/stereoexample_zed/test/1280/right/Explorer_HD720_SN21067_21-14-47.jpg")
    parser.add_argument("-p","--param_file",default=r"param/zed1280.yaml")
    # 3D 重建点云文件的保存路径
    parser.add_argument("--ply_path",default="./data/3d.ply")
    
    ## 命令行参数
    args = parser.parse_args()
    main(args)