"""
需要放置左右摄像机的视频路径
"""


import sys
sys.path.append('core')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import stereoconfig
import time


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
        disp = model(left_rected, right_rected, iters=4, test_mode=True)
        end = time.time()
        print("the consume time :",end - start)
        disp = disp.cpu().numpy()
        disp = padder.unpad(disp)
    return disp.squeeze()

def demo(args):
    ## 参数获取
    print("to obtain the parameters!")
    config = stereoconfig.stereoCamera(args.param_file)
    # 视频读取
    print("read left and right video")
    left_video = cv2.VideoCapture(args.left_video)
    right_video = cv2.VideoCapture(args.right_video)
    # 视频参数
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
    fps = left_video.get(cv2.CAP_PROP_FPS)  # 帧数
    # 视频写入
    print("create the video writer")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (config.width,config.height))  
    # 模型
    print("load the igve stereo matching model !")
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.to(DEVICE)
    model.eval()
    print("begin to caculate !")
    index = 0
    while 1:
        # 读取帧
        left_ret,left_frame = left_video.read()
        right_ret,right_frame = right_video.read()
        if (left_ret | right_ret) == False:
            print("the image from the video is empty , exit !")
            break
        if left_frame.shape[0] != config.height:
            left_frame = cv2.resize(left_frame,dsize=(config.width,config.height))
            right_frame = cv2.resize(right_frame,dsize=(config.width,config.height))
        # 畸变矫正
        left_frame,right_frame = config.rectify(left_frame,right_frame)
        # 推理
        disp = inference(left_frame,right_frame,model)
        index +=1
        print("the number of frame" , index)
        # 颜色映射
        disp_norm = (disp - np.min(disp)) / (np.max(disp) - np.min(disp)) * 255
        disp_vis = np.round(disp_norm).astype(np.uint8)
        disp_vis = cv2.applyColorMap(disp_vis,cv2.COLORMAP_INFERNO)
        # 写入
        writer.write(disp_vis)
        # 显示
        # cv2.imshow("disparity visualization" , disp_vis)
        # if(cv2.waitKey(1)==27):
        #     break

    # 资源释放
    left_video.release()
    right_video.release()
    writer.release()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained/sceneflow.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')


    parser.add_argument('-l', '--left_video', help="path to  left video", default="E:\\dataset\\calibrate\\stereoMatch\\stereoexample_2\\video\\left.avi")
    parser.add_argument('-r', '--right_video', help="path to  right video", default="E:\\dataset\\calibrate\\stereoMatch\\stereoexample_2\\video\\right.avi")



    parser.add_argument('--output_video', help="directory to save output", default="./demo-output/disp.mp4")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    parser.add_argument("--param_file",default="./param/stereomatch2_copy.yaml")
    
    
    args = parser.parse_args()
    demo(args)
