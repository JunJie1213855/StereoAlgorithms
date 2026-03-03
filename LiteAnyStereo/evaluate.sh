#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH


python evaluate_stereo.py --restore_ckpt ./checkpoints/LiteAnyStereo.pth --dataset middlebury_H
python evaluate_stereo.py --restore_ckpt ./checkpoints/LiteAnyStereo.pth --dataset eth3d
python evaluate_stereo.py --restore_ckpt ./checkpoints/LiteAnyStereo.pth --dataset kitti
python evaluate_stereo.py --restore_ckpt ./checkpoints/LiteAnyStereo.pth --dataset drivingstereo

