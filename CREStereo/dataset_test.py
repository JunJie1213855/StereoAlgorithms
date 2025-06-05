import argparse
import os
import shutil
import sys
import time
import logging
from collections import namedtuple
from itertools import repeat

import yaml
from nets import Model
from dataset import CREStereoDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler


def parse_yaml(file_path: str) -> namedtuple:
    """Parse yaml configuration file and return the object in `namedtuple`."""
    with open(file_path, "rb") as f:
        cfg: dict = yaml.safe_load(f)
    args = namedtuple("train_args", cfg.keys())(*cfg.values())
    # save cfg into train_log
    ensure_dir(args.log_dir)
    dst_file = os.path.join(args.log_dir, file_path.split('/')[-1])
    shutil.copy2(file_path, dst_file)    
    return args

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)




def train(args, world_size):
    print(args.training_data_path)
    # datasets
    dataset = CREStereoDataset(args.training_data_path)
    sampler = RandomSampler(dataset, replacement=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size*world_size,
                            num_workers=0, drop_last=True, persistent_workers=False, pin_memory=True)



def main(args):
    # initial info
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    world_size = torch.cuda.device_count()  # number of GPU(s)
    train(args, world_size)

if __name__ == "__main__":
    # train configuration
    args = parse_yaml("E:/code/python/CREStereo/stereodataset/cfgs/KITTI.yaml")
    main(args)