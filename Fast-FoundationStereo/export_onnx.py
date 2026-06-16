# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
导出 FoundationStereo 模型为 ONNX 格式

采用分阶段导出策略（参考 make_onnx.py）:
1. 特征提取器 (Feature Runner) - 导出为 ONNX
2. 后处理 (Post Runner) - 导出为 ONNX
3. GWC Volume 构建 - 需要在推理时使用 PyTorch/TensorRT 实现
"""

import os
import sys
import argparse
import logging

# 禁用 TorchDynamo 避免编译问题
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import yaml
import torch
import numpy as np
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/')

from Utils import AMP_DTYPE, set_logging_format, set_seed
from core.foundation_stereo import FastFoundationStereo, TrtFeatureRunner, TrtPostRunner, build_gwc_volume_triton


def export_feature_and_post(args):
    """分阶段导出特征提取器和后处理器"""
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    # 加载模型
    logging.info(f"Loading model from {args.model_dir}")
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp

    # 确保输入尺寸能被 32 整除
    H, W = args.height, args.width
    assert H % 32 == 0 and W % 32 == 0, "height and width must be divisible by 32"

    # 创建特征提取器和后处理器
    feature_runner = TrtFeatureRunner(model)
    post_runner = TrtPostRunner(model)
    feature_runner.eval()
    post_runner.eval()

    # 创建随机输入
    left_img = torch.randn(1, 3, H, W)
    right_img = torch.randn(1, 3, H, W)

    logging.info(f"Exporting ONNX model: {H}x{W} input")
    logging.info(f"  - valid_iters: {args.valid_iters}")
    logging.info(f"  - max_disp: {args.max_disp}")

    # 创建输出目录
    output_dir = os.path.dirname(args.save_path)
    os.makedirs(output_dir, exist_ok=True)

    # 导出特征提取器
    feature_path = args.save_path.replace('.onnx', '_feature.onnx')
    logging.info(f"Exporting feature runner to {feature_path}")
    torch.onnx.export(
        feature_runner,
        (left_img, right_img),
        feature_path,
        opset_version=args.opset_version,
        input_names=['left', 'right'],
        output_names=[
            'features_left_04', 'features_left_08', 'features_left_16', 'features_left_32',
            'features_right_04', 'stem_2x'
        ],
        do_constant_folding=True,
        verbose=args.verbose,
    )
    logging.info(f"Feature runner exported to: {feature_path}")

    # 运行一次前向传播获取中间输出
    features_left_04, features_left_08, features_left_16, features_left_32, features_right_04, stem_2x = \
        feature_runner(left_img, right_img)

    # 构建 GWC Volume
    gwc_volume = build_gwc_volume_triton(
        features_left_04.half(),
        features_right_04.half(),
        args.max_disp // 4,
        model.cv_group
    )

    # 获取后处理输出用于验证
    disp = post_runner(
        features_left_04.float(), features_left_08.float(), features_left_16.float(),
        features_left_32.float(), features_right_04.float(), stem_2x.float(), gwc_volume.float()
    )

    # 导出后处理器
    post_path = args.save_path.replace('.onnx', '_post.onnx')
    logging.info(f"Exporting post runner to {post_path}")
    torch.onnx.export(
        post_runner,
        (features_left_04, features_left_08, features_left_16, features_left_32,
         features_right_04, stem_2x, gwc_volume),
        post_path,
        opset_version=args.opset_version,
        input_names=[
            'features_left_04', 'features_left_08', 'features_left_16', 'features_left_32',
            'features_right_04', 'stem_2x', 'gwc_volume'
        ],
        output_names=['disparity'],
        do_constant_folding=True,
        verbose=args.verbose,
    )
    logging.info(f"Post runner exported to: {post_path}")

    # 保存配置信息
    config_path = args.save_path.replace('.onnx', '_config.yaml')
    config_dict = {
        'valid_iters': args.valid_iters,
        'max_disp': args.max_disp,
        'height': H,
        'width': W,
        'feature_output_names': [
            'features_left_04', 'features_left_08', 'features_left_16', 'features_left_32',
            'features_right_04', 'stem_2x'
        ],
        'post_input_names': [
            'features_left_04', 'features_left_08', 'features_left_16', 'features_left_32',
            'features_right_04', 'stem_2x', 'gwc_volume'
        ],
        'output_names': ['disparity'],
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    logging.info(f"Config saved to: {config_path}")

    logging.info("=" * 50)
    logging.info("ONNX export completed!")
    logging.info(f"  - Feature runner: {feature_path}")
    logging.info(f"  - Post runner: {post_path}")
    logging.info(f"  - Config: {config_path}")
    logging.info("=" * 50)

    # 验证导出的模型
    if args.verify:
        logging.info("Verifying ONNX models...")
        try:
            import onnx
            # 验证特征提取器
            onnx_model = onnx.load(feature_path)
            onnx.checker.check_model(onnx_model)
            logging.info(f"  - Feature runner verified!")

            # 验证后处理器
            onnx_model = onnx.load(post_path)
            onnx.checker.check_model(onnx_model)
            logging.info(f"  - Post runner verified!")
        except Exception as e:
            logging.warning(f"  - Verification failed: {e}")


def export_full_model(args):
    """尝试导出完整模型（可能因 Unfold 操作失败）"""
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    # 加载模型
    logging.info(f"Loading model from {args.model_dir}")
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.eval()

    # 确保输入尺寸能被 32 整除
    H, W = args.height, args.width
    assert H % 32 == 0 and W % 32 == 0, "height and width must be divisible by 32"

    # 创建简化包装类
    class SimpleFoundationStereo(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self._model = model
            self.args = model.args

        def forward(self, left, right):
            return self._model.forward(
                left, right,
                iters=self.args.valid_iters,
                test_mode=True,
                optimize_build_volume=False  # 禁用复杂体积
            )

    onnx_model = SimpleFoundationStereo(model)
    onnx_model.eval()

    # 创建随机输入
    left_input = torch.randn(1, 3, H, W)
    right_input = torch.randn(1, 3, H, W)

    logging.info(f"Exporting full model: {H}x{W}")

    # 导出
    torch.onnx.export(
        onnx_model,
        (left_input, right_input),
        args.save_path,
        opset_version=args.opset_version,
        input_names=['left', 'right'],
        output_names=['disparity'],
        do_constant_folding=True,
        verbose=args.verbose,
    )
    logging.info(f"Full model exported to: {args.save_path}")


def main():
    parser = argparse.ArgumentParser(description='Export FoundationStereo to ONNX')

    # 模型参数
    parser.add_argument('--model_dir', type=str,
                        default=f'{code_dir}/weights/23-36-37/model_best_bp2_serialize.pth',
                        help='Path to model weights')
    parser.add_argument('--save_path', type=str,
                        default=f'{code_dir}/weights/foundation_stereo.onnx',
                        help='Path to save ONNX model')

    # 输入尺寸
    parser.add_argument('--height', type=int, default=448,
                        help='Input height (must be divisible by 32)')
    parser.add_argument('--width', type=int, default=640,
                        help='Input width (must be divisible by 32)')

    # 模型配置
    parser.add_argument('--valid_iters', type=int, default=8,
                        help='Number of flow-field updates during forward pass')
    parser.add_argument('--max_disp', type=int, default=192,
                        help='Maximum disparity')

    # 导出选项
    parser.add_argument('--opset_version', type=int, default=17,
                        help='ONNX opset version')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify ONNX model after export')
    parser.add_argument('--full_model', action='store_true',
                        help='Export full model (may fail due to Unfold)')

    args = parser.parse_args()

    if args.full_model:
        export_full_model(args)
    else:
        export_feature_and_post(args)


if __name__ == '__main__':
    main()
