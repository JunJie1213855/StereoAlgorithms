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
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import time


def load_image(imfile):
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    img:np.ndarray = cv2.imread(imfile).astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img,None,fx=0.25,fy=0.25)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args:argparse.Namespace):
    print("model downloading ! ")
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    # model = torch.load(IGEVStereo(args),torch.device("cpu"))

    
    model = model.module
    model.to(DEVICE)
    model.eval()

    print("check the output save directory")
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    print("begin to caculate !")
    with torch.no_grad():
        print("check the image !")
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        # ground_truth = cv2.imread(args.ground_truth,cv2.IMREAD_GRAYSCALE)
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")
        for i in range(len(left_images)):
            print("path -->  left :",left_images[i],"right :",right_images[i])

        
        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            start = time.time()
            
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            end = time.time()
            print("the consume time :",end - start)
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            print(disp.squeeze().shape)
            # print(ground_truth.shape)
            # l1loss:np.ndarray = np.abs(disp - ground_truth)
            filename = os.path.join(output_directory, "disp.png")
            plt.imsave(filename, disp.squeeze(), cmap='jet')
            # cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(disp.squeeze(), alpha=0.01),cv2.COLORMAP_INFERNO), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # cv2.imwrite(os.path.join(output_directory, "L1loss.png"), cv2.applyColorMap(cv2.convertScaleAbs(l1loss.squeeze(), alpha=0.01),cv2.COLORMAP_INFERNO), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained/pth/middlebury.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')


    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frame", default="./left_rect.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frame", default="./right_rect.png")
    # parser.add_argument("-g", "--ground_truth",help="path to all ground truth frame",default="E:\\dataset\\kitti\\KITTI_2015\\training\\disp_noc_0\\000053_10.png")

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="E:\\dataset\\Slam\\visual\\VBR\\vbr\\left\\1.jpg")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="E:\\dataset\\Slam\\visual\\VBR\\vbr\\right\\1.jpg")

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="E:\\客户\\研究生毕设\\数据集\\Sampler\\FlyingThings3D\\RGB_cleanpass\\left\\0006.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="E:\\客户\\研究生毕设\\数据集\\Sampler\\FlyingThings3D\\RGB_cleanpass\\right\\0006.png")


    # parser.add_argument('--output_path', help="directory to save output", default="E:\\客户\\研究生毕设\\数据集\\IGEV_dataset\\MiddleBury\\2014\\trainingH\\Adirondack\\1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
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
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)
