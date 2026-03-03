# StereoAlgorithms

<div align="center">

A comprehensive collection of state-of-the-art stereo matching algorithms for depth estimation from stereo image pairs.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)

</div>

## Table of Contents

- [Overview](#overview)
- [Featured Algorithms](#featured-algorithms)
- [Performance Comparison](#performance-comparison)
- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Algorithm Selection Guide](#algorithm-selection-guide)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

This repository brings together multiple cutting-edge approaches for stereo matching, ranging from lightweight mobile-friendly models to high-accuracy research-grade implementations. It serves as a comprehensive toolkit for researchers and engineers working on depth estimation, 3D reconstruction, and stereo vision applications.

### Key Features

- **State-of-the-art Accuracy**: Includes top-ranking methods on major benchmarks (KITTI, Middlebury, ETH3D, SceneFlow)
- **Multiple Paradigms**: Volume-based, iterative optimization, and hybrid approaches
- **Mobile-Friendly Options**: Lightweight models optimized for edge deployment
- **Zero-Shot Generalization**: Models trained on synthetic data that generalize well to real scenes
- **Point Cloud Processing**: Tools for point cloud registration and fusion

## Featured Algorithms

### High-Performance Methods

| Algorithm | Venue | Highlights | Status |
|-----------|-------|------------|--------|
| **[MonSter](MonSter/)** | CVPR 2025 Highlight | Ranks #1 across SceneFlow, KITTI 2012/2015, Middlebury, ETH3D | [Code](MonSter/) |
| **[MonSter++](MonSter-plusplus/)** | CVPR 2025 | Unified stereo matching, multi-view stereo, and real-time stereo | [Code](MonSter-plusplus/) |
| **[FoundationStereo](FoundationStereo/)** | CVPR 2025 Oral (Best Paper Nomination) | Zero-shot stereo matching foundation model | [Code](FoundationStereo/) |
| **[IGEV++](IGEV-plusplus/)** | arXiv 2024 | Multi-range geometry encoding volumes | [Code](IGEV-plusplus/) |
| **[S²M²](s2m2/)** | ICCV 2025 | Scalable stereo matching model, 1st on ETH3D/Middlebury V3 | [Code](s2m2/) |
| **[RAFT-Stereo](RAFT-Stereo/)** | 3DV 2021 (Best Student Paper) | Multilevel recurrent field transforms | [Code](RAFT-Stereo/) |
| **[Fast-ACVNet](Fast-ACVNet/)** | TPAMI 2023 | Attention concatenation volume | [Code](Fast-ACVNet/) |

### Specialized Methods

| Algorithm | Venue | Highlights | Status |
|-----------|-------|------------|--------|
| **[MoCha-Stereo](MoCha-Stereo/)** | CVPR 2024 | Motif channel attention with white-box interpretability | [Code](MoCha-Stereo/) |
| **[IGEV-Stereo](IGEV-Stereo/)** | CVPR 2023 | Iterative geometry encoding volume | [Code](IGEV-Stereo/) |
| **[GGEV](GGEV/)** | AAAI 2026 | Generalized geometry encoding for real-time | [Code](GGEV/) |
| **[CREStereo](CREStereo/)** | - | Efficient and accurate stereo matching | [Code](CREStereo/) |

### Lightweight & Mobile

| Algorithm | Venue | Highlights | Status |
|-----------|-------|------------|--------|
| **[MobileStereoNet](mobilestereonet/)** | WACV 2022 | Optimized for mobile deployment | [Code](mobilestereonet/) |
| **[LiteAnyStereo](LiteAnyStereo/)** | arXiv 2025 | Efficient zero-shot stereo matching | [Code](LiteAnyStereo/) |
| **[RT-MonSter++](MonSter-plusplus/RT-MonSter++)** | CVPR 2025 | Real-time version of MonSter++ | [Code](MonSter-plusplus/RT-MonSter++) |
| **[Selective-Stereo](Selective-Stereo/)** | - | Adaptive frequency information selection | [Code](Selective-Stereo/) |

### Classical & Foundational

| Algorithm | Venue | Highlights | Status |
|-----------|-------|------------|--------|
| **[PSMNet](PSMNet/)** | CVPR 2018 | Pyramid stereo matching network | [Code](PSMNet/) |

### Utilities

| Module | Description |
|--------|-------------|
| **[plyregis](plyregis/)** | Point cloud registration and fusion system (Chinese) |

## Performance Comparison

| Algorithm | SceneFlow EPE [dwn] | KITTI 2015 D1-all [%] | Middlebury [%] | ETH3D [%] | Mobile-Friendly |
|-----------|:-------------------:|:---------------------:|:--------------:|:---------:|:---------------:|
| **MonSter** | **0.71** | **1.45** | **0.89** | **0.41** | [ ] |
| **MonSter++** | - | **~1.2** | - | ~0.5 | [ ] |
| **FoundationStereo** | - | ~1.5 | **1st** | **1st** | [ ] |
| **S²M²** | - | ~1.6 | **1st V3** | **1st** | [ ] |
| **IGEV++** | 0.78 | 1.58 | 1.12 | 0.52 | [ ] |
| **GGEV** | - | - | - | - | [x] |
| **RAFT-Stereo** | 0.85 | 1.72 | 1.28 | 0.61 | [ ] |
| **MoCha-Stereo** | 0.92 | 1.89 | 1.35 | 0.68 | [ ] |
| **Fast-ACVNet+** | 0.59 | 2.01 | - | - | [ ] |
| **MobileStereoNet** | 1.14 | 2.42 | - | - | [x] |
| **LiteAnyStereo** | - | ~2.5 | - | - | [x] |
| **PSMNet** | 1.09 | 2.32 | - | - | [ ] |

*[ ] Lower values are better. Values are approximations from respective papers.*

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support (recommended RTX 3090/4090 or better)
- Python 3.8+
- PyTorch 1.10+
- Ubuntu 20.04 or similar Linux distribution

### Common Dependencies

Most algorithms share these core dependencies:

```bash
pip install torch torchvision torchaudio
pip install tqdm scipy opencv-python scikit-image
pip install tensorboard matplotlib
```

### Algorithm-Specific Setup

Each algorithm may have specific requirements. Here are some examples:

#### RAFT-Stereo (Recommended for beginners)

```bash
cd RAFT-Stereo
conda env create -f environment.yaml
conda activate raftstereo
```

#### IGEV++

```bash
cd IGEV-plusplus
conda create -n IGEV_plusplus python=3.8
conda activate IGEV_plusplus
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/cu113
pip install timm==0.5.4
```

#### MonSter

```bash
cd MonSter
conda create -n monster python=3.8
conda activate monster
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.6.13 mmcv==2.1.0 accelerate==1.0.1
```

#### MobileStereoNet

```bash
cd mobilestereonet
conda env create --file mobilestereonet.yml
conda activate mobilestereonet
```

## Datasets

### Standard Benchmarks

| Dataset | Type | Resolution | Download |
|---------|------|------------|----------|
| **[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)** | Synthetic | 960x540 | [Link](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) |
| **[KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)** | Real-world | ~1242x375 | [Link](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) |
| **[Middlebury](https://vision.middlebury.edu/stereo/data/)** | Indoor | Variable | [Link](https://vision.middlebury.edu/stereo/data/) |
| **[ETH3D](https://www.eth3d.net/datasets)** | Multi-view | Variable | [Link](https://www.eth3d.net/datasets) |

### Additional Datasets

- **[TartanAir](https://github.com/castacks/tartanair_tools)** - Synthetic with diverse weather
- **[CREStereo Dataset](https://github.com/megvii-research/CREStereo)** - Large-scale real-world
- **[FallingThings](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)** - Object-centric
- **[InStereo2K](https://github.com/YuhuaXu/StereoDataset)** - 2K resolution
- **[Sintel Stereo](http://sintel.is.tue.mpg.de/stereo)** - CGI movie scenes
- **[DrivingStereo](https://drivingstereo-dataset.github.io/)** - Real-world driving

### Dataset Organization

Create a `datasets/` directory with the following structure:

```
datasets/
├── FlyingThings3D/
│   ├── frames_finalpass/
│   └── disparity/
├── Monkaa/
├── Driving/
├── KITTI/
│   ├── KITTI_2015/
│   └── KITTI_2012/
├── Middlebury/
│   └── MiddEval3/
├── ETH3D/
│   ├── two_view_training/
│   └── two_view_testing/
└── ...
```

## Quick Start

### 1. Pretrained Model Demo

Most algorithms provide pretrained models and demo scripts:

```bash
# RAFT-Stereo demo (recommended for beginners)
cd RAFT-Stereo
bash download_models.sh
python demo.py --restore_ckpt models/raftstereo-middlebury.pth \
    -l datasets/Middlebury/MiddEval3/testF/*/im0.png \
    -r datasets/Middlebury/MiddEval3/testF/*/im1.png

# MonSter demo (best accuracy)
cd MonSter
wget https://huggingface.co/cjd24/MonSter/resolve/main/mix_all.pth
python evaluate_stereo.py --restore_ckpt ./mix_all.pth --dataset kitti

# FoundationStereo demo (zero-shot generalization)
cd FoundationStereo
python scripts/run_demo.py --left_file ./assets/left.png \
    --right_file ./assets/right.png \
    --ckpt_dir ./pretrained_models/23-51-11/ \
    --out_dir ./test_outputs/

# MobileStereoNet demo (mobile-friendly)
cd mobilestereonet
python prediction.py --loadckpt ./checkpoints/pretrained.ckpt
```

### 2. Training

Each algorithm supports training on various datasets:

```bash
# Train on SceneFlow (common starting point)
cd RAFT-Stereo
python train_stereo.py --batch_size 8 --mixed_precision

cd IGEV-plusplus
python train_stereo.py --train_datasets sceneflow

cd MonSter
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_sceneflow.py
```

### 3. Evaluation

Evaluate trained models on benchmark datasets:

```bash
# KITTI evaluation
cd MonSter
python save_disp.py  # Generate submission files

# Middlebury evaluation
cd RAFT-Stereo
python evaluate_stereo.py --dataset middlebury_H

# Zero-shot generalization
cd IGEV-plusplus
python evaluate_stereo.py --dataset kitti --restore_ckpt ./pretrained/sceneflow.pth
```

## Algorithm Selection Guide

### For Research & Maximum Accuracy

| Algorithm | Best For | Note |
|-----------|----------|------|
| **MonSter** | Best overall performance | #1 on most leaderboards |
| **MonSter++** | Multi-view stereo | Unified framework |
| **FoundationStereo** | Zero-shot generalization | Foundation model approach |
| **IGEV++** | Large disparities | Excellent in textureless regions |
| **RAFT-Stereo** | Well-balanced | Extensive documentation |

### For Real-Time Applications

| Algorithm | Speed | Accuracy | Note |
|-----------|-------|----------|------|
| **RT-MonSter++** | Fastest | High | Real-time MonSter variant |
| **GGEV** | Fast | Good | Optimized for real-time |
| **MobileStereoNet** | Very Fast | Moderate | Mobile-optimized |
| **LiteAnyStereo** | Very Fast | Good | Efficient zero-shot |

### For Robustness & Generalization

| Algorithm | Strength |
|-----------|----------|
| **MonSter (mix_all)** | Best zero-shot generalization |
| **FoundationStereo** | Foundation model with extensive training |
| **RAFT-Stereo** | Proven robustness across domains |
| **MoCha-Stereo** | White-box interpretability |

### For Learning & Education

| Algorithm | Reason |
|-----------|--------|
| **PSMNet** | Classic architecture, easy to understand |
| **RAFT-Stereo** | Well-documented, iterative approach |
| **MobileStereoNet** | Simpler architecture, good starting point |

## Acknowledgments

This collection wouldn't be possible without the original authors and their excellent work. Special thanks to:

- **Princeton Vision Lab** - [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)
- **Gangwei Xu et al.** - [IGEV/IGEV++](https://github.com/gangweiX/IGEV)
- **Junda Cheng et al.** - [MonSter](https://github.com/Junda24/MonSter)
- **NVIDIA Research** - [FoundationStereo](https://nvlabs.github.io/FoundationStereo/)
- **Faranak Shamsafar et al.** - [MobileStereoNet](https://github.com/azmoonas/MobileStereoNet)
- **Ziyang Chen et al.** - [MoCha-Stereo](https://github.com/ZYangChen/MoCha-Stereo)
- **Samsung Electronics** - [S²M²](https://github.com/junhong-3dv/s2m2)

## License

Each algorithm has its own license. Most are under MIT or Apache 2.0 licenses. Please check individual algorithm directories for specific license terms.

## Contributing

This repository is a collection of implementations from various research groups. For algorithm-specific contributions:

1. Navigate to the specific algorithm directory
2. Follow the contribution guidelines in that algorithm's README
3. For general collection improvements, open an issue

## Contact

For algorithm-specific issues, please check the individual algorithm's README and GitHub issues. For general collection issues, open an issue in this repository.

---

<div align="center">

**[⭐ Star this repo](../../stargazers)** if you find it helpful!

</div>
