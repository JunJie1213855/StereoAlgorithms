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
from core.monster import Monster
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import time
class NormalizeTensor(object):
    """Normalize a tensor by given mean and std."""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            
        Returns:
            Tensor: Normalized Tensor image.
        """
        # Ensure mean and std have the same number of channels as the input tensor
        Device = tensor.device
        self.mean = self.mean.to(Device)
        self.std = self.std.to(Device)

        # Normalize the tensor
        if self.mean.ndimension() == 1:
            self.mean = self.mean[:, None, None]
        if self.std.ndimension() == 1:
            self.std = self.std[:, None, None]

        return (tensor - self.mean) / self.std


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

            

def inference(left_img:np.ndarray,right_img:np.ndarray,args:argparse.Namespace):
    # 模型
    print("load the monster stereo matching model !")
    model = torch.nn.DataParallel(Monster(args), device_ids=[0])

    assert os.path.exists(args.restore_ckpt)
    checkpoint = torch.load(args.restore_ckpt)
    ckpt = dict()
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    for key in checkpoint:
        ckpt['module.' + key] = checkpoint[key]

    model.load_state_dict(checkpoint, strict=True)
    model = model.module
    model.to(DEVICE)
    model.eval()

    print("check the output save directory")
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    print("begin to caculate !")
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
        ## 保存
        file_stem = args.save_name
        filename = os.path.join(output_directory, f"{file_stem}.png")
        ## 用plt的颜色映射表保存
        plt.imsave(output_directory / f"{file_stem}.png", disp.squeeze(), cmap='jet')
    
    print("the igev caculation end !")
    return disp.squeeze()



def main(args:argparse.Namespace):
    ## 参数获取
    print("to obtain the parameters!")
    config = stereoconfig.stereoCamera(args.param_file)
    left_img : np.ndarray = cv2.imread(args.left_img)
    right_img : np.ndarray = cv2.imread(args.right_img)

    ## 依靠参数文件的图像尺寸resize
    # if left_img.shape[0] != config.height:
    #     left_img = cv2.resize(left_img,dsize=(config.width,config.height))
    #     right_img = cv2.resize(right_img,dsize=(config.width,config.height))

    ## 畸变矫正
    left_img,right_img = config.rectify(left_img,right_img)
    # left_img = cv2.resize(left_img,dsize=(640, 360))
    # right_img = cv2.resize(right_img,dsize=(640, 360))
    if args.showrecitied:
        print("show the rectified picures !")
        catimg = config.cat(left_img,right_img)
        cv2.namedWindow("catimage",cv2.WINDOW_NORMAL)
        cv2.imshow("catimage",catimg)
        # cv2.imshow("right rectified",right_img)
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
    # 预训练模型和一些选项
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="./pretrained/middlebury.pth")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=256, help="max disp of geometry encoding volume")
    

    ## 添加的选项
    parser.add_argument("-sn","--save_name",default="disparity")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
    parser.add_argument("--showrecitied",default=True,help="是否显示矫正的左右图")
    
    parser.add_argument("--left_img","-l",default=r"/mnt/d/dataset/CameraCalib/stereoexample_github/stereoexample_zed/test/1280/left/Explorer_HD720_SN21067_22-54-39.jpg")
    parser.add_argument("--right_img","-r",default=r"/mnt/d/dataset/CameraCalib/stereoexample_github/stereoexample_zed/test/1280/right/Explorer_HD720_SN21067_22-54-39.jpg")
    parser.add_argument("-p","--param_file",default=r"param/zed1280.yaml")
    # 3D 重建点云文件的保存路径
    parser.add_argument("--ply_path",default="./data/3d.ply")
    
    ## 命令行参数
    args = parser.parse_args()
    main(args)