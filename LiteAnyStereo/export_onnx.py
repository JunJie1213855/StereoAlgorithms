"""
LiteAnyStereo模型ONNX导出程序

使用方法:
    # 基础导出
    python export_onnx.py

    # 自定义参数导出
    python export_onnx.py --max_disp 192 --img_height 480 --img_width 640

    # 动态尺寸导出
    python export_onnx.py --dynamic_shape --opset_version 17

问题解析:
    错误 "indices element out of data bounds" 是因为导出时非张量参数(max_disp, test_mode)
    被错误地暴露为模型输入。解决方案是创建包装类固定这些参数。
"""
import torch
import torch.nn as nn
import torch.onnx
import argparse
import sys
import os
from pathlib import Path

sys.path.append('core')
from core.liteanystereo import LiteAnyStereo


class LiteAnyStereoONNXWrapper(nn.Module):
    """
    LiteAnyStereo ONNX导出包装类

    将额外的参数(max_disp, test_mode)固定为模型属性,
    只暴露left和right两个图像输入
    """
    def __init__(self, model, max_disp=192):
        super(LiteAnyStereoONNXWrapper, self).__init__()
        self.model = model
        self.max_disp = max_disp

    def forward(self, left, right):
        """
        只接受左右图像作为输入,固定max_disp和test_mode参数

        Args:
            left: 左图像 (B, 3, H, W), 范围 [0, 255]
            right: 右图像 (B, 3, H, W), 范围 [0, 255]

        Returns:
            disparity: 视差图 (B, 1, H, W)
        """
        return self.model(left, right, max_disp=self.max_disp, test_mode=True)


def export_to_onnx(model, output_path, args):
    """
    导出PyTorch模型到ONNX格式

    Args:
        model: PyTorch模型
        output_path: 输出ONNX文件路径
        args: 命令行参数
    """
    model.eval()

    # 创建包装模型,固定max_disp参数
    wrapped_model = LiteAnyStereoONNXWrapper(model, max_disp=args.max_disp)
    wrapped_model.eval()

    # 创建示例输入 - 图像范围 [0, 255]
    dummy_left = torch.randn(1, 3, args.img_height, args.img_width).cuda() * 255
    dummy_right = torch.randn(1, 3, args.img_height, args.img_width).cuda() * 255

    # 定义输入输出名称
    input_names = ['left_image', 'right_image']
    output_names = ['disparity']

    # 设置动态轴 (如果启用动态尺寸)
    dynamic_axes = None
    if args.dynamic_shape:
        dynamic_axes = {
            'left_image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'right_image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'disparity': {0: 'batch_size', 2: 'height', 3: 'width'}
        }

    print("=" * 60)
    print("ONNX Export Configuration:")
    print("=" * 60)
    print(f"  Output Path:        {output_path}")
    print(f"  Input Shape:        (1, 3, {args.img_height}, {args.img_width})")
    print(f"  Input Range:        [0, 255]")
    print(f"  Max Disparity:      {args.max_disp} (embedded)")
    print(f"  Opset Version:      {args.opset_version}")
    print(f"  Dynamic Shape:      {args.dynamic_shape}")
    print(f"  Simplify:           {args.simplify}")
    print("=" * 60)

    # 导出模型
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,                      # 使用包装后的模型
            (dummy_left, dummy_right),          # 只传递图像输入
            output_path,
            export_params=True,                 # 存储训练好的参数权重
            opset_version=args.opset_version,   # ONNX版本
            do_constant_folding=args.constant_folding,  # 执行常量折叠优化
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )

    print(f"\n✓ Model exported successfully to: {output_path}")

    # 验证ONNX模型
    if args.verify:
        print("\nVerifying ONNX model...")
        try:
            import onnx
            import onnxruntime as ort

            # 加载并检查ONNX模型
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model check passed!")

            # 打印模型输入输出
            print("\nModel Inputs:")
            for inp in onnx_model.graph.input:
                print(f"  {inp.name}: {inp.type}")
            print("\nModel Outputs:")
            for out in onnx_model.graph.output:
                print(f"  {out.name}: {out.type}")

            # 使用ONNXRuntime推理测试
            print("\nTesting inference with ONNXRuntime...")
            sess = ort.InferenceSession(output_path)

            # 准备输入
            ort_inputs = {
                'left_image': dummy_left.cpu().numpy(),
                'right_image': dummy_right.cpu().numpy()
            }

            # 运行推理
            ort_outputs = sess.run(None, ort_inputs)
            print(f"✓ ONNXRuntime inference successful!")
            print(f"  Output shape: {ort_outputs[0].shape}")

            # 与PyTorch输出对比
            with torch.no_grad():
                torch_output = wrapped_model(dummy_left, dummy_right)

            torch_output = torch_output.cpu().numpy()
            diff = abs(torch_output - ort_outputs[0]).max()
            print(f"  Max difference: {diff:.6f}")

            if diff < 1e-3:
                print("✓ Output verification passed!")
            else:
                print(f"⚠ Warning: Output difference is large ({diff:.6f})")

        except ImportError:
            print("⚠ Warning: onnx or onnxruntime not installed. Skipping verification.")
        except Exception as e:
            print(f"⚠ Verification failed: {e}")
            import traceback
            traceback.print_exc()

    # 简化ONNX模型 (如果启用)
    if args.simplify:
        try:
            import onnx
            import onnxsim
            print("\nSimplifying ONNX model...")
            simplified_path = output_path.replace('.onnx', '_simplified.onnx')

            onnx_model = onnx.load(output_path)
            model_sim, check = onnxsim.simplify(onnx_model)

            if check:
                onnx.save(model_sim, simplified_path)
                print(f"✓ Simplified model saved to: {simplified_path}")

                # 显示简化前后的模型大小对比
                original_size = os.path.getsize(output_path) / (1024 * 1024)
                simplified_size = os.path.getsize(simplified_path) / (1024 * 1024)
                reduction = (1 - simplified_size / original_size) * 100
                print(f"  Original size:  {original_size:.2f} MB")
                print(f"  Simplified size: {simplified_size:.2f} MB")
                print(f"  Reduction:      {reduction:.1f}%")
            else:
                print("⚠ Simplification check failed, keeping original model")

        except ImportError:
            print("⚠ Warning: onnx-simplifier not installed. Run: pip install onnx-sim")
        except Exception as e:
            print(f"⚠ Simplification failed: {e}")
            import traceback
            traceback.print_exc()


def main(args):
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 加载模型
    print(f"\nLoading LiteAnyStereo model...")
    model = LiteAnyStereo()

    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        target_model = model.module if hasattr(model, 'module') else model
        target_model.load_state_dict(checkpoint, strict=True)
        print("✓ Checkpoint loaded successfully")
    else:
        print("⚠ Warning: No checkpoint provided, using random weights")

    model = model.to(device)
    model.eval()

    # 导出ONNX
    output_path = output_dir / args.output_name
    export_to_onnx(model, str(output_path), args)

    print("\n" + "=" * 60)
    print("Export completed!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export LiteAnyStereo to ONNX')

    # 模型参数
    parser.add_argument('--checkpoint', '-c', type=str,
                        default='./pretrained/LiteAnyStereo.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--max_disp', type=int, default=192,
                        help='Maximum disparity value (will be embedded in ONNX)')

    # 导出配置
    parser.add_argument('--output_dir', '-o', type=str, default='./onnx_models',
                        help='Output directory for ONNX models')
    parser.add_argument('--output_name', '-n', type=str, default='liteanystereo.onnx',
                        help='Name of the output ONNX file')

    # 输入尺寸
    parser.add_argument('--img_height', '-H', type=int, default=736,
                        help='Input image height for export')
    parser.add_argument('--img_width', '-W', type=int, default=1280,
                        help='Input image width for export')

    # ONNX导出选项
    parser.add_argument('--opset_version', type=int, default=17,
                        help='ONNX opset version (recommended: 11-17)')
    parser.add_argument('--dynamic_shape', action='store_true',
                        help='Enable dynamic input/output shapes')
    parser.add_argument('--constant_folding', action='store_true', default=True,
                        help='Apply constant folding optimization')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model (requires onnx-sim)')
    parser.add_argument('--verify', '-v', action='store_true', default=True,
                        help='Verify ONNX model after export')

    args = parser.parse_args()
    main(args)
