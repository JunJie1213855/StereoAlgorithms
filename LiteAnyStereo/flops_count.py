import sys
sys.path.append('core')
import argparse

import torch
from thop import profile, clever_format
from core.liteanystereo import LiteAnyStereo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    args = parser.parse_args()

    # img = torch.randn(1, 3, 544, 960)
    img = torch.randn(1, 3, 384, 1248)
    img = img.cuda()

    model = LiteAnyStereo()
    model.cuda()

    macs, params = profile(model, inputs=(img, img))
    macs, params = clever_format([macs, params], "%.3f")  # to be consistent with neapeak

    print("Input size:", img.size())
    print("Macs:", macs)
    print("params:", params)