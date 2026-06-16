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
import cv2
import time


def load_image(imfile, height=None, width=None):
    img: np.ndarray = cv2.imread(imfile).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if height is not None and width is not None:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    elif height is not None or width is not None:
        h, w = img.shape[:2]
        if height is not None:
            width = int(w * height / h)
        else:
            height = int(h * width / w)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def benchmark(args: argparse.Namespace):
    print("Loading model...")
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.to(DEVICE)
    model.eval()

    print("Loading images...")
    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))

    # Warmup
    print(f"Warmup: running {args.warmup_iters} iterations...")
    with torch.no_grad():
        for i in range(args.warmup_iters):
            image1 = load_image(left_images[0], args.height, args.width)
            image2 = load_image(right_images[0], args.height, args.width)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            print(image1.size())
            _ = model(image1, image2, iters=args.valid_iters, test_mode=True)
    torch.cuda.synchronize()
    print(f"Warmup completed.")

    # Benchmark
    print(f"\nBenchmark: running {args.test_iters} iterations...")
    times = []
    with torch.no_grad():
        for i in tqdm(range(args.test_iters)):
            idx = i % len(left_images)
            image1 = load_image(left_images[idx], args.height, args.width)
            image2 = load_image(right_images[idx], args.height, args.width)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            torch.cuda.synchronize()
            start = time.time()
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)

    times = np.array(times)
    print("\n" + "=" * 50)
    print("Benchmark Results:")
    print("=" * 50)
    print(f"Mean time:   {times.mean():.4f} s")
    print(f"Std time:   {times.std():.4f} s")
    print(f"Min time:   {times.min():.4f} s")
    print(f"Max time:   {times.max():.4f} s")
    print(f"Median time: {np.median(times):.4f} s")
    print(f"FPS:        {1.0 / times.mean():.2f}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained/middlebury.pth')

    parser.add_argument('-l', '--left_imgs', help="path to left images", default="./rect_left.png")
    parser.add_argument('-r', '--right_imgs', help="path to right images", default="./rect_right.png")

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=4, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU layers")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    # Benchmark options
    parser.add_argument('--warmup_iters', type=int, default=10, help='number of warmup iterations')
    parser.add_argument('--test_iters', type=int, default=100, help='number of test iterations')
    parser.add_argument('--height', type=int, default=736, help='target height (default: original)')
    parser.add_argument('--width', type=int, default=1280, help='target width (default: original)')

    args = parser.parse_args()

    benchmark(args)
