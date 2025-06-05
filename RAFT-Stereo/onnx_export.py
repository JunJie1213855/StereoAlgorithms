import sys
sys.path.append('core')
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo_ONNX
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
DEVICE = 'cuda'
def demo(args):
    model = torch.nn.DataParallel(RAFTStereo_ONNX(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt,map_location='cpu'),strict=True)
    model = model.module
    model.to('cpu')
    model.eval()
    h,w = args.img_size
    left = torch.rand(1,3,h,w).to('cpu')
    right = torch.rand(1,3,h,w).to('cpu')
    torch.onnx.export(
        model,
        (left,right),
        input_names=["left","right"],
        output_names=["disparity"],
        f = args.onnx_path,
        export_params= True,
        opset_version=16,
        do_constant_folding=True,
        verbose=True
    )



    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="E:\\code\\python\\StereoMatch\\RAFT-Stereo-main\\model\\raftstereo-middlebury.pth")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="E:\\dataset\\Middlebury2021\\traproom1\\im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="E:\\dataset\\Middlebury2021\\traproom1\\im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument("--onnx_path",default="./raft.onnx")
    parser.add_argument("--test_mode",default=True)
    parser.add_argument("--img_size",default=(352,640))
    args = parser.parse_args()

    demo(args)
