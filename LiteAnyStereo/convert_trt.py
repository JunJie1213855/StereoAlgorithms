"""
LiteAnyStereo TensorRT Engine 转换工具

使用方法:
    # 基础转换 (FP16)
    python convert_trt.py

    # 指定ONNX模型路径
    python convert_trt.py --onnx ./onnx_models/liteanystereo.onnx

    # FP32精度
    python convert_trt.py --precision fp32

    # 动态尺寸转换
    python convert_trt.py --dynamic_shape --min_shape 1x3x480x640 --opt_shape 1x3x736x1280 --max_shape 1x3x1080x1920

    # 使用INT8量化 (需要校准数据)
    python convert_trt.py --precision int8 --calibration_dataset ./datasets/

    # 启用TF32
    python convert_trt.py --precision tf32

    # 仅构建推理引擎 (不包含优化profile)
    python convert_trt.py --build_for_inference
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
# from cuda import cuda, cudart
import pycuda
import pycuda.autoinit
# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_tensorrt_version():
    """获取TensorRT版本"""
    return trt.__version__


def check_cuda_available():
    """检查CUDA是否可用"""
    try:
        cudaInit = cuda.cuInit(0)
        cuDeviceGetCount = cuda.cuDeviceGetCount()
        print(f"[OK] CUDA initialized successfully")
        return True
    except Exception as e:
        print(f"[ERROR] CUDA not available: {e}")
        return False


class TensorRTConverter:
    """TensorRT ONNX to Engine 转换器"""

    def __init__(self, onnx_path: str, output_dir: str = "./trt_models"):
        self.onnx_path = Path(onnx_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 生成输出engine路径
        self.engine_path = self.output_dir / f"{self.onnx_path.stem}.engine"

    def build_engine(
        self,
        precision: str = "fp16",
        dynamic_shape: bool = False,
        min_shape: tuple = (1, 3, 480, 640),
        opt_shape: tuple = (1, 3, 736, 1280),
        max_shape: tuple = (1, 3, 1080, 1920),
        workspace_size: int = 4 * (1 << 30),  # 4GB
        calibration_dataset: str = None,
        build_for_inference: bool = False,
    ) -> str:
        """
        构建TensorRT引擎

        Args:
            precision: 精度模式 "fp16", "fp32", "tf32", "int8"
            dynamic_shape: 是否启用动态尺寸
            min_shape: 最小输入尺寸
            opt_shape: 最优输入尺寸
            max_shape: 最大输入尺寸
            workspace_size: TensorRT工作空间大小 (字节)
            calibration_dataset: INT8校准数据集路径
            build_for_inference: 是否仅为推理构建

        Returns:
            生成的engine文件路径
        """
        print("=" * 60)
        print("TensorRT Build Configuration:")
        print("=" * 60)
        print(f"  ONNX Model:      {self.onnx_path}")
        print(f"  Output Engine:   {self.engine_path}")
        print(f"  Precision:       {precision}")
        print(f"  Dynamic Shape:   {dynamic_shape}")
        if dynamic_shape:
            print(f"  Min Shape:       {min_shape}")
            print(f"  Opt Shape:       {opt_shape}")
            print(f"  Max Shape:       {max_shape}")
        print(f"  Workspace:       {workspace_size / (1<<30):.1f} GB")
        if precision == "int8":
            print(f"  Calibration:     {calibration_dataset}")
        print(f"  Build for Inference: {build_for_inference}")
        print("=" * 60)

        # 创建builder和网络
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()

        # 设置工作空间大小
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        # 解析ONNX模型
        parser = trt.OnnxParser(network, TRT_LOGGER)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("ERROR: Failed to parse ONNX model!")
                for error in range(parser.num_errors):
                    print(f"  Error {error}: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")

        # 获取输入输出信息
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        print(f"\nModel I/O:")
        for inp in inputs:
            print(f"  Input:  {inp.name} - {inp.shape}")
        for out in outputs:
            print(f"  Output: {out.name} - {out.shape}")

        # 设置精度模式
        if precision in ["fp16", "tf32"]:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print(f"\n[OK] FP16 enabled")
            else:
                print(f"[WARNING] FP16 not supported on this platform")

        if precision == "tf32":
            if builder.platform_has_fast_tf32:
                config.set_flag(trt.BuilderFlag.TF32)
                print(f"[OK] TF32 enabled")
            else:
                print(f"[WARNING] TF32 not supported on this platform")

        if precision == "int8":
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                print(f"[OK] INT8 enabled")
                if calibration_dataset:
                    # 使用校准器
                    config.int8_calibrator = self._create_calibrator(
                        calibration_dataset, inputs[0].shape if not dynamic_shape else opt_shape
                    )
                else:
                    print("[WARNING] INT8 requires calibration dataset!")
            else:
                print(f"[WARNING] INT8 not supported on this platform")

        # 配置优化profile (动态形状)
        if dynamic_shape:
            profile = builder.create_optimization_profile()

            for inp in inputs:
                profile.set_shape(
                    inp.name,
                    min=min_shape,
                    opt=opt_shape,
                    max=max_shape
                )
                print(f"\n[OK] Dynamic shape set for {inp.name}")
                print(f"    min: {min_shape}, opt: {opt_shape}, max: {max_shape}")

            config.add_optimization_profile(profile)

        # 构建引擎
        print("\nBuilding TensorRT engine...")
        if build_for_inference:
            # 仅构建推理引擎
            engine_bytes = builder.build_serialized_network(network, config)
        else:
            # 构建包含优化profile的引擎
            engine_bytes = builder.build_serialized_network(network, config)

        if engine_bytes is None:
            raise RuntimeError("Failed to build TensorRT engine!")

        # 保存engine文件
        with open(self.engine_path, 'wb') as f:
            f.write(engine_bytes)

        file_size = self.engine_path.stat().st_size / (1024 * 1024)
        print(f"\n[OK] Engine saved to: {self.engine_path}")
        print(f"     Engine size:   {file_size:.2f} MB")

        return str(self.engine_path)

    def _create_calibrator(self, calibration_dataset: str, input_shape: tuple):
        """创建INT8校准器"""
        print(f"[INFO] Creating INT8 calibrator with dataset: {calibration_dataset}")

        # 收集校准图像
        calib_images = []
        dataset_path = Path(calibration_dataset)

        if dataset_path.is_dir():
            # 查找图像文件
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.ppm']:
                calib_images.extend(list(dataset_path.glob(ext)))

        # 如果没有找到图像，使用随机数据
        if len(calib_images) == 0:
            print(f"[WARNING] No images found in {calibration_dataset}, using random data")
            calib_images = None

        class DatasetCalibrator(trt.IInt8LegacyCalibrator):
            def __init__(self, images, input_shape):
                super().__init__()
                self.images = images
                self.input_shape = input_shape
                self.batch_size = input_shape[0]
                self.current_index = 0

                # 预分配内存
                self.device_input = cuda.cuMemAlloc(np.prod(input_shape) * 4)[1]

            def get_batch(self, names):
                if self.images is None or self.current_index >= len(self.images):
                    return None

                # 加载图像
                import cv2
                batch_data = []

                for i in range(self.batch_size):
                    if self.current_index >= len(self.images):
                        break

                    img_path = self.images[self.current_index]
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
                        img = img.transpose(2, 0, 1).flatten() / 255.0
                        batch_data.append(img)

                    self.current_index += 1

                if len(batch_data) == 0:
                    return None

                # 填充剩余批次
                while len(batch_data) < self.batch_size:
                    batch_data.append(np.zeros((3, self.input_shape[2], self.input_shape[3]), dtype=np.float32))

                batch_array = np.stack(batch_data).astype(np.float32)

                # 拷贝到设备
                cuda.cuMemcpyHtoD(self.device_input, batch_array, batch_array.nbytes)

                return [int(self.device_input)]

            def get_batch_size(self):
                return self.batch_size

            def get_legacy_calibrator_name(self):
                return "LiteAnyStereoCalibrator"

        return DatasetCalibrator(calib_images, input_shape)


def convert_onnx_to_trt(
    onnx_path: str,
    output_dir: str = "./trt_models",
    precision: str = "fp16",
    dynamic_shape: bool = False,
    min_shape: tuple = (1, 3, 480, 640),
    opt_shape: tuple = (1, 3, 736, 1280),
    max_shape: tuple = (1, 3, 1080, 1920),
    workspace_size: int = 4 * (1 << 30),
    calibration_dataset: str = None,
    build_for_inference: bool = False,
) -> str:
    """
    将ONNX模型转换为TensorRT引擎

    Args:
        onnx_path: ONNX模型路径
        output_dir: 输出目录
        precision: 精度 "fp16", "fp32", "tf32", "int8"
        dynamic_shape: 启用动态形状
        min_shape: 最小输入形状
        opt_shape: 最优输入形状
        max_shape: 最大输入形状
        workspace_size: 工作空间大小 (字节)
        calibration_dataset: INT8校准数据集路径
        build_for_inference: 仅构建推理引擎

    Returns:
        engine文件路径
    """
    converter = TensorRTConverter(onnx_path, output_dir)
    return converter.build_engine(
        precision=precision,
        dynamic_shape=dynamic_shape,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        workspace_size=workspace_size,
        calibration_dataset=calibration_dataset,
        build_for_inference=build_for_inference,
    )


def main(args):
    """主函数"""
    print("=" * 60)
    print("LiteAnyStereo TensorRT Converter")
    print(f"TensorRT Version: {get_tensorrt_version()}")
    print("=" * 60)

    # 检查CUDA
    if not check_cuda_available():
        print("ERROR: CUDA is required for TensorRT conversion")
        sys.exit(1)

    # 默认ONNX路径
    if args.onnx is None:
        default_onnx = Path("./onnx_models/liteanystereo.onnx")
        if default_onnx.exists():
            args.onnx = str(default_onnx)
        else:
            # 尝试其他常见路径
            for p in ["liteanystereo.onnx", "../LiteAnyStereo.onnx"]:
                if Path(p).exists():
                    args.onnx = p
                    break
            else:
                print(f"ERROR: ONNX model not found. Please specify with --onnx")
                sys.exit(1)

    # 解析形状参数
    min_shape = tuple(map(int, args.min_shape.split('x')))
    opt_shape = tuple(map(int, args.opt_shape.split('x')))
    max_shape = tuple(map(int, args.max_shape.split('x')))

    # 执行转换
    try:
        engine_path = convert_onnx_to_trt(
            onnx_path=args.onnx,
            output_dir=args.output_dir,
            precision=args.precision,
            dynamic_shape=args.dynamic_shape,
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            workspace_size=args.workspace * (1 << 30),
            calibration_dataset=args.calibration_dataset,
            build_for_inference=args.build_for_inference,
        )

        print("\n" + "=" * 60)
        print("Conversion completed successfully!")
        print(f"Engine saved to: {engine_path}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LiteAnyStereo ONNX to TensorRT Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 模型路径
    parser.add_argument("--onnx", "-i", type=str, default=None,
                        help="Path to input ONNX model (default: ./onnx_models/liteanystereo.onnx)")
    parser.add_argument("--output_dir", "-o", type=str, default="./trt_models",
                        help="Output directory for TensorRT engine (default: ./trt_models)")

    # 精度设置
    parser.add_argument("--precision", "-p", type=str,
                        choices=["fp16", "fp32", "tf32", "int8"],
                        default="fp16",
                        help="TensorRT precision mode (default: fp16)")

    # 动态形状
    parser.add_argument("--dynamic_shape", "-d", action="store_true",
                        help="Enable dynamic input shapes")
    parser.add_argument("--min_shape", type=str, default="1x3x480x640",
                        help="Minimum input shape (NCHW format)")
    parser.add_argument("--opt_shape", type=str, default="1x3x736x1280",
                        help="Optimal input shape for optimization (NCHW format)")
    parser.add_argument("--max_shape", type=str, default="1x3x1080x1920",
                        help="Maximum input shape (NCHW format)")

    # INT8 校准
    parser.add_argument("--calibration_dataset", "-c", type=str, default=None,
                        help="Path to calibration dataset for INT8 quantization")

    # 构建选项
    parser.add_argument("--workspace", type=int, default=4,
                        help="Workspace size in GB (default: 4)")
    parser.add_argument("--build_for_inference", action="store_true",
                        help="Build engine only for inference (without optimization profiles)")

    args = parser.parse_args()
    main(args)
