import sys
sys.path.append('core')
import os
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.monster import Monster 
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(Monster(args), device_ids=[0])

    checkpoint = torch.load(args.restore_ckpt)
    ckpt = dict()
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    for key in checkpoint:
        ckpt['module.' + key] = checkpoint[key]

    model.load_state_dict(ckpt, strict=True)

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        datasets_path = '/data2/cjd/StereoDatasets/eth3d/two_view_testing'
        for dir_name in os.listdir(datasets_path):
            dir_path = os.path.join(datasets_path, dir_name)
            if not os.path.isdir(dir_path):
                continue

            output_file_path = os.path.join(output_directory, dir_name + '.pfm')
            timing_file_path = os.path.join(output_directory, dir_name + '.txt')

            if os.path.isfile(output_file_path) and os.path.isfile(timing_file_path):
                print('Skipping since output already present: ' + dir_name)
                continue
            
            print('Processing: ' + dir_name)
            
            # Assemble call.
            left_image_path = os.path.join(dir_path, 'im0.png')
            right_image_path = os.path.join(dir_path, 'im1.png')
            image1 = load_image(left_image_path)
            image2 = load_image(right_image_path)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            starter.record()
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            disp = disp.cpu().numpy()
            disp = padder.unpad(disp).squeeze()
            with open(output_file_path, 'wb') as f:
                H, W = disp.shape
                headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
                for header in headers:
                    f.write(str.encode(header))
                array = np.flip(disp, axis=0).astype(np.float32)
                f.write(array.tobytes())

            with open(timing_file_path, 'wb') as f:
                time ='runtime %.2f' % (curr_time / 1000)
                f.write(str.encode(time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="/data2/cjd/mono_fusion/checkpoints/eth3d.pth")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default=None)
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default=None)
    parser.add_argument('--output_directory', help="directory to save output", default='./test_output/eth3d/')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
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

    demo(args)
