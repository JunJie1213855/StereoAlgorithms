# StereoAlgorithms

A comprehensive collection of state-of-the-art stereo matching algorithms implementations. This repository brings together multiple cutting-edge approaches for depth estimation from stereo image pairs, ranging from lightweight mobile-friendly models to high-accuracy research-grade implementations.

## 🚀 Featured Algorithms

### 📊 High-Performance Methods
- **[MonSter](MonSter/)** ⭐ (CVPR 2025 Highlight) - Marry Monodepth to Stereo Unleashes Power. Ranks #1 across SceneFlow, KITTI 2012/2015, Middlebury, and ETH3D benchmarks.
- **[IGEV++](IGEV-plusplus/)** - Iterative Multi-range Geometry Encoding Volumes for Stereo Matching with excellent performance in textureless regions.
- **[RAFT-Stereo](RAFT-Stereo/)** ⭐ (3DV 2021 Best Student Paper) - Multilevel Recurrent Field Transforms for Stereo Matching.

### 🎯 Specialized Methods
- **[MoCha-Stereo](MoCha-Stereo/)** ⭐ (CVPR 2024) - Motif Channel Attention Network for Stereo Matching with white-box interpretability.
- **[IGEV-Stereo](IGEV-Stereo/)** - Iterative Geometry Encoding Volume for Stereo Matching.
- **[CREStereo](CREStereo/)** - Efficient and accurate stereo matching network.

### 📱 Lightweight & Mobile
- **[MobileStereoNet](mobilestereonet/)** ⭐ (WACV 2022) - Towards Lightweight Deep Networks for Stereo Matching, optimized for mobile deployment.
- **[Selective-Stereo](Selective-Stereo/)** - Efficient selective stereo matching approaches.

### 🏛️ Classical & Foundational
- **[PSMNet](PSMNet/)** - Pyramid Stereo Matching Network, a foundational deep learning approach.

## 📈 Performance Comparison

| Algorithm | SceneFlow EPE ↓ | KITTI 2015 | Middlebury | ETH3D | Mobile-Friendly |
|-----------|----------------|------------|------------|-------|-----------------|
| **MonSter** | **0.71** | **1.45** | **0.89** | **0.41** | ❌ |
| **IGEV++** | 0.78 | 1.58 | 1.12 | 0.52 | ❌ |
| **RAFT-Stereo** | 0.85 | 1.72 | 1.28 | 0.61 | ❌ |
| **MoCha-Stereo** | 0.92 | 1.89 | 1.35 | 0.68 | ❌ |
| **MobileStereoNet** | 1.14 | 2.42 | - | - | ✅ |

*Lower EPE (End-Point Error) is better. Values are approximations from respective papers.*

## 🛠️ Installation

### Prerequisites
- NVIDIA GPU with CUDA support (recommended RTX 3090 or better)
- Python 3.10
- PyTorch 2.3.0
- Ubuntu 20.04
### Common Dependencies
Most algorithms share these core dependencies:
```bash
pip install torch==2.3.0 torchvision==0.18.0
pip install tqdm scipy opencv-python scikit-image
pip install tensorboard matplotlib
```

### Algorithm-Specific Setup
Each algorithm has its own requirements file:

```bash
# For RAFT-Stereo (recommended starting point)
cd RAFT-Stereo
conda env create -f environment.yaml
conda activate raftstereo

# For IGEV++/IGEV-Stereo
cd IGEV-plusplus  # or IGEV-Stereo
pip install -r requirements.txt
pip install timm==0.5.4

# For MonSter
cd MonSter
pip install torch==2.0.1 torchvision==0.15.2
pip install timm==0.6.13 mmcv==2.1.0 accelerate==1.0.1

# For MobileStereoNet
cd mobilestereonet
conda env create --file mobilestereonet.yml
conda activate mobilestereonet
```

## 📊 Datasets

### Standard Benchmarks
- **[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)** - Large-scale synthetic dataset
- **[KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)** - Real-world driving scenarios
- **[Middlebury](https://vision.middlebury.edu/stereo/data/)** - High-resolution indoor scenes
- **[ETH3D](https://www.eth3d.net/datasets)** - Multi-view stereo benchmark

### Additional Datasets
- **TartanAir** - Synthetic dataset with diverse weather conditions
- **CREStereo Dataset** - Large-scale real-world stereo pairs
- **FallingThings** - Synthetic object-centric dataset
- **InStereo2K** - 2K resolution stereo dataset
- **Sintel Stereo** - Computer graphics movie scenes
- **DrivingStereo** - Real-world driving stereo pairs

### Dataset Organization
Create a `datasets/` directory with the following structure:
```
datasets/
├── FlyingThings3D/
├── Monkaa/
├── Driving/
├── KITTI/
├── Middlebury/
├── ETH3D/
└── [other datasets]/
```

## 🚀 Quick Start

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

# MobileStereoNet demo (mobile-friendly)
cd mobilestereonet
# Download pretrained model from README links
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

## 📱 Algorithm Selection Guide

### For Research & Maximum Accuracy
- **MonSter** - Best overall performance across all benchmarks
- **IGEV++** - Excellent for textureless regions and large disparities
- **RAFT-Stereo** - Well-balanced performance with extensive documentation

### For Real-time Applications
- **MobileStereoNet** - Optimized for mobile and edge devices
- **RAFT-Stereo (real-time variant)** - Faster implementation with mixed precision
- **IGEV++ RT** - Real-time version of IGEV++

### For Robustness & Generalization
- **MonSter (mix_all model)** - Best zero-shot generalization
- **RAFT-Stereo** - Proven robustness across domains
- **MoCha-Stereo** - White-box interpretability for debugging


## 🤝 Contributing

This repository is a collection of implementations from various research groups. For algorithm-specific contributions:

1. Navigate to the specific algorithm directory
2. Follow the contribution guidelines in that algorithm's README
3. For general collection improvements, open an issue

## 📄 License

Each algorithm has its own license:
- Most are under MIT or Apache 2.0 licenses
- Check individual algorithm directories for specific license terms

## 🙏 Acknowledgments

This collection wouldn't be possible without the original authors and their excellent work:

- [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) - Princeton Vision Lab
- [IGEV/IGEV++](https://github.com/gangweiX/IGEV) - Gangwei Xu et al.
- [MonSter](https://github.com/Junda24/MonSter) - Junda Cheng et al.
- [MobileStereoNet](https://github.com/azmoonas/MobileStereoNet) - Faranak Shamsafar et al.
- [MoCha-Stereo](https://github.com/ZYangChen/MoCha-Stereo) - Ziyang Chen et al.

## 📞 Support

For algorithm-specific issues:
- Check the individual algorithm's README and GitHub issues
- Contact the original authors as listed in their papers

For general collection issues:
- Open an issue in this repository
- Check the FAQ sections in individual algorithm READMEs

---

**⭐ If you find this collection useful, please consider starring the repository and citing the original papers!**
