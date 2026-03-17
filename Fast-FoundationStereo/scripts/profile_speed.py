# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
import argparse, torch, logging, yaml, time
import numpy as np
from Utils import AMP_DTYPE, set_logging_format, set_seed


if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', default=f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth', type=str)
  parser.add_argument('--hiera', default=0, type=int)
  parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
  parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity')
  parser.add_argument('--warmup', type=int, default=15, help='number of warmup iterations')
  parser.add_argument('--total', type=int, default=30, help='total number of iterations')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.backends.cudnn.benchmark = True
  torch.autograd.set_grad_enabled(False)

  with open(f'{os.path.dirname(args.model_dir)}/cfg.yaml', 'r') as ff:
    cfg:dict = yaml.safe_load(ff)
  for k in args.__dict__:
    if args.__dict__[k] is not None:
      cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
  model.args.valid_iters = args.valid_iters
  model.args.max_disp = args.max_disp
  model.cuda().eval()

  H, W = 480, 640
  img0 = torch.randint(0, 256, (1, 3, H, W), dtype=torch.float32).cuda()
  img1 = torch.randint(0, 256, (1, 3, H, W), dtype=torch.float32).cuda()
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  logging.info(f"Image size: {H}x{W}, warmup: {args.warmup}, total: {args.total}")

  times = []
  with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    for i in range(args.total):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True, optimize_build_volume='triton')
      torch.cuda.synchronize()
      elapsed = time.perf_counter() - t0
      times.append(elapsed)
      logging.info(f"Iter {i:2d}: {elapsed*1000:.1f} ms {'(warmup)' if i < args.warmup else ''}")

  measure_times = times[args.warmup:]
  avg = np.mean(measure_times) * 1000
  logging.info(f"Vanilla Pytorch speed average (after warmup): {avg:.1f}[ms] over {len(measure_times)} iters")
