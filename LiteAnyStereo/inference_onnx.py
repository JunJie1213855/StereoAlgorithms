"""
ONNX模型推理与性能测试程序

使用方法:
    # 基础推理
    python inference_onnx.py --left_img left.jpg --right_img right.jpg --onnx_model liteanystereo.onnx

    # 性能测试 (预热+多次测量)
    python inference_onnx.py --left_img left.jpg --right_img right.jpg --onnx_model liteanystereo.onnx \
        --benchmark --warmup 5 --runs 50 --use_gpu

    # 保存视差图
    python inference_onnx.py --left_img left.jpg --right_img right.jpg --onnx_model liteanystereo.onnx --save_dir ./output
"""
import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from collections import defaultdict


def preprocess_image(img_path, target_size=None):
    """
    预处理图像

    Args:
        img_path: 图像路径
        target_size: 目标尺寸 (height, width)，None表示保持原尺寸

    Returns:
        img: BGR格式的图像
        img_rgb: RGB格式的图像 (模型输入)
        original_shape: 原始图像尺寸
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    original_shape = img.shape[:2]

    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb, original_shape


def pad_to_multiple(img, multiple=32):
    """
    将图像padding到指定倍数

    Args:
        img: 输入图像 (H, W, C)
        multiple: 倍数

    Returns:
        img_padded: padding后的图像
        pad_info: padding信息 (top, bottom, left, right)
    """
    H, W = img.shape[:2]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    pad_info = (top, bottom, left, right)

    return img_padded, pad_info


def unpad(img, pad_info):
    """移除padding"""
    top, bottom, left, right = pad_info
    H, W = img.shape[:2]
    return img[top:H-bottom, left:W-right]


def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf):
    """可视化视差图"""
    disp = disp.copy()
    H, W = disp.shape[:2]
    invalid_mask = disp >= invalid_thres

    if (invalid_mask == 0).sum() == 0:
        return np.zeros((H, W, 3))

    if min_val is None:
        min_val = disp[invalid_mask == 0].min()
    if max_val is None:
        max_val = disp[invalid_mask == 0].max()

    vis = ((disp - min_val) / (max_val - min_val)).clip(0, 1) * 255
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]

    if invalid_mask.any():
        vis[invalid_mask] = 0

    return vis.astype(np.uint8)


def print_histogram(times, bins=20):
    """
    打印推理时间的直方图

    Args:
        times: 推理时间列表 (ms)
        bins: 直方图柱子数量
    """
    if len(times) == 0:
        return

    times = np.array(times)
    hist, edges = np.histogram(times, bins=bins)

    max_count = hist.max()
    max_width = 50

    print("\n" + "=" * 60)
    print("Inference Time Histogram (ms)")
    print("=" * 60)

    for i in range(len(hist)):
        count = hist[i]
        bar_width = int(count / max_count * max_width) if max_count > 0 else 0
        bar = "█" * bar_width
        print(f"{edges[i]:6.1f}-{edges[i+1]:6.1f} | {bar} {count}")

    print("=" * 60)


def print_statistics(times, label="Inference"):
    """
    打印统计信息

    Args:
        times: 推理时间列表 (ms)
        label: 标签
    """
    if len(times) == 0:
        return

    times = np.array(times)

    print(f"\n{'='*60}")
    print(f"{label} Statistics")
    print(f"{'='*60}")
    print(f"  Total runs:        {len(times)}")
    print(f"  Mean:              {times.mean():.2f} ms")
    print(f"  Median:            {np.median(times):.2f} ms")
    print(f"  Std Dev:           {times.std():.2f} ms")
    print(f"  Min:               {times.min():.2f} ms")
    print(f"  Max:               {times.max():.2f} ms")
    print(f"  Range:             {times.max() - times.min():.2f} ms")
    print(f"  CV (变异系数):     {times.std()/times.mean()*100:.2f}%")
    print(f"  FPS:               {1000/times.mean():.2f}")
    print(f"{'='*60}")

    # 打印百分位数
    percentiles = [50, 90, 95, 99]
    print(f"Percentiles:")
    for p in percentiles:
        print(f"  P{p}:               {np.percentile(times, p):.2f} ms")


def run_benchmark(session, inputs, warmup_runs=5, benchmark_runs=50, print_warmup=False):
    """
    运行性能测试

    Args:
        session: ONNX Runtime推理会话
        inputs: 输入数据字典
        warmup_runs: 预热次数
        benchmark_runs: 测试次数
        print_warmup: 是否打印预热时间

    Returns:
        warmup_times: 预热时间列表 (ms)
        benchmark_times: 测试时间列表 (ms)
    """
    warmup_times = []
    benchmark_times = []

    # 预热
    if warmup_runs > 0:
        print(f"\nWarming up ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            start = time.perf_counter()
            _ = session.run(None, inputs)
            end = time.perf_counter()
            elapsed = (end - start) * 1000
            warmup_times.append(elapsed)
            if print_warmup:
                print(f"  Warmup {i+1}/{warmup_runs}: {elapsed:.2f} ms")

    # 同步GPU (确保测量准确)
    if 'CUDAExecutionProvider' in session.get_providers():
        try:
            import onnxruntime
            # 尝试同步CUDA
            session.run(None, inputs)  # 额外一次同步
        except:
            pass

    # 性能测试
    print(f"\nRunning benchmark ({benchmark_runs} runs)...")
    for i in range(benchmark_runs):
        start = time.perf_counter()
        _ = session.run(None, inputs)
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        benchmark_times.append(elapsed)

        # 实时进度显示
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Progress: {i+1}/{benchmark_runs} runs completed")

    return warmup_times, benchmark_times


def main(args):
    import onnxruntime as ort

    # 设置执行提供者
    if args.use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    print("=" * 60)
    print("ONNX Model Inference & Benchmark")
    print("=" * 60)
    print(f"  Model:         {args.onnx_model}")
    print(f"  Left Image:    {args.left_img}")
    print(f"  Right Image:   {args.right_img}")
    print(f"  Providers:     {providers}")
    print(f"  Benchmark:     {args.benchmark}")
    print("=" * 60)

    # 加载ONNX模型
    print("\nLoading ONNX model...")
    session = ort.InferenceSession(args.onnx_model, providers=providers)

    # 获取提供者信息
    available_providers = ort.get_available_providers()
    enabled_providers = session.get_providers()
    print(f"Available providers: {available_providers}")
    print(f"Enabled providers:   {enabled_providers}")

    # 获取GPU信息
    if 'CUDAExecutionProvider' in enabled_providers:
        try:
            cuda_provider_info = session.get_provider_options()
            print(f"CUDA Provider Info:  {cuda_provider_info}")
        except:
            pass

    # 获取输入输出信息
    input_info = session.get_inputs()
    output_info = session.get_outputs()

    print("\nModel I/O:")
    for info in input_info:
        print(f"  Input:  {info.name}, shape: {info.shape}, dtype: {info.type}")
    for info in output_info:
        print(f"  Output: {info.name}, shape: {info.shape}, dtype: {info.type}")

    # 读取并预处理图像
    print(f"\nLoading images...")

    # 解析target_size
    target_size = None
    if args.target_size:
        if isinstance(args.target_size, str):
            target_size = tuple(map(int, args.target_size.split(',')))
        else:
            target_size = tuple(args.target_size)

    left_img, left_rgb, left_shape = preprocess_image(args.left_img, target_size=target_size)
    right_img, right_rgb, right_shape = preprocess_image(args.right_img, target_size=target_size)

    # Padding到32的倍数
    print(f"Original size: {left_img.shape[:2]}")
    left_rgb_pad, left_pad_info = pad_to_multiple(left_rgb, 32)
    right_rgb_pad, right_pad_info = pad_to_multiple(right_rgb, 32)
    print(f"Padded size: {left_rgb_pad.shape[:2]}")

    # 准备输入数据
    # 模型期望: (B, C, H, W), 范围 [0, 255]
    left_input = left_rgb_pad.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
    right_input = right_rgb_pad.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

    inputs = {
        'left_image': left_input,
        'right_image': right_input
    }

    # 性能测试
    if args.benchmark:
        warmup_times, benchmark_times = run_benchmark(
            session, inputs,
            warmup_runs=args.warmup,
            benchmark_runs=args.runs,
            print_warmup=args.verbose
        )

        # 打印预热统计
        if args.verbose and warmup_times:
            print_statistics(warmup_times, "Warmup")

        # 打印性能测试统计
        print_statistics(benchmark_times, "Benchmark")

        # 打印直方图
        print_histogram(benchmark_times, bins=args.hist_bins)

        # 打印稳定性分析
        print("\nStability Analysis:")
        mean = np.mean(benchmark_times)
        std = np.std(benchmark_times)
        cv = std / mean * 100

        if cv < 5:
            stability = "Excellent (Very Stable)"
        elif cv < 10:
            stability = "Good (Stable)"
        elif cv < 20:
            stability = "Fair (Moderate Variance)"
        else:
            stability = "Poor (High Variance)"

        print(f"  Stability Rating:  {stability}")
        print(f"  CV:               {cv:.2f}% (Lower is better)")

    # 单次推理获取结果
    print("\nRunning inference for output...")
    start = time.time()
    outputs = session.run(None, inputs)
    end = time.time()

    print(f"Inference time: {(end - start) * 1000:.1f} ms")

    # 处理输出
    disparity = outputs[0].squeeze()

    # 移除padding
    disparity = unpad(disparity, left_pad_info)

    print(f"Output disparity shape: {disparity.shape}")
    print(f"Disparity range: [{disparity.min():.2f}, {disparity.max():.2f}]")

    # 可视化
    vis_disp = vis_disparity(disparity)

    # 保存结果
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True)

        # 保存视差图
        disp_path = save_dir / "disparity_color.png"
        cv2.imwrite(str(disp_path), vis_disp)
        print(f"\nSaved colored disparity to: {disp_path}")

        # 保存原始视差
        raw_disp_path = save_dir / "disparity_raw.png"
        cv2.imwrite(str(raw_disp_path), (disparity / np.abs(disparity).max() * 255).astype(np.uint8))
        print(f"Saved raw disparity to: {raw_disp_path}")

        # 保存性能报告
        if args.benchmark:
            report_path = save_dir / "benchmark_report.txt"
            with open(report_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("ONNX Model Benchmark Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Model: {args.onnx_model}\n")
                f.write(f"Providers: {enabled_providers}\n")
                f.write(f"Image size: {left_rgb_pad.shape[:2]}\n")
                f.write(f"Warmup runs: {args.warmup}\n")
                f.write(f"Benchmark runs: {args.runs}\n\n")
                f.write("Statistics:\n")
                f.write(f"  Mean:     {np.mean(benchmark_times):.2f} ms\n")
                f.write(f"  Median:   {np.median(benchmark_times):.2f} ms\n")
                f.write(f"  Std Dev:  {np.std(benchmark_times):.2f} ms\n")
                f.write(f"  Min:      {np.min(benchmark_times):.2f} ms\n")
                f.write(f"  Max:      {np.max(benchmark_times):.2f} ms\n")
                f.write(f"  FPS:      {1000/np.mean(benchmark_times):.2f}\n")
                f.write(f"  CV:       {np.std(benchmark_times)/np.mean(benchmark_times)*100:.2f}%\n\n")
                f.write(f"Stability: {stability}\n")
            print(f"Saved benchmark report to: {report_path}")

    # 显示结果
    if args.show:
        # 拼接左右图像
        cat_img = np.concatenate([left_img, right_img], axis=1)
        display_img = cv2.resize(cat_img, (cat_img.shape[1]//2, cat_img.shape[0]//2))
        cv2.imshow("Stereo Images", display_img)

        disp_display = cv2.resize(vis_disp, (vis_disp.shape[1]//2, vis_disp.shape[0]//2))
        cv2.imshow("Disparity", disp_display)

        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ONNX Model Inference & Benchmark for LiteAnyStereo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python inference_onnx.py -l left.jpg -r right.jpg

  # Benchmark with GPU (50 runs)
  python inference_onnx.py -l left.jpg -r right.jpg --benchmark --use_gpu

  # Extensive benchmark (100 runs, 10 warmup)
  python inference_onnx.py -l left.jpg -r right.jpg --benchmark --runs 100 --warmup 10
        """
    )

    # 输入参数
    parser.add_argument('--onnx_model', '-m', type=str, default="onnx_models/liteanystereo.onnx",
                        help='Path to the ONNX model file')
    parser.add_argument('--left_img', '-l', type=str, default="rect_left.png",
                        help='Path to the left image')
    parser.add_argument('--right_img', '-r', type=str, default="rect_right.png",
                        help='Path to the right image')
    parser.add_argument("--target-size", type=str, default=None,
                        help="Target image size (e.g., '736,1280' or 736,1280)")

    # 性能测试参数
    parser.add_argument('--benchmark', '-b', action='store_true',
                        help='Enable benchmarking mode')
    parser.add_argument('--warmup', '-w', type=int, default=5,
                        help='Number of warmup runs before benchmark (default: 5)')
    parser.add_argument('--runs', '-n', type=int, default=50,
                        help='Number of benchmark runs (default: 50)')
    parser.add_argument('--hist-bins', type=int, default=20,
                        help='Number of histogram bins (default: 20)')

    # 输出选项
    parser.add_argument('--save_dir', '-o', type=str, default=None,
                        help='Directory to save output results')
    parser.add_argument('--show', '-s', action='store_false', default=True,
                        help='Show visualization')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print verbose output including warmup times')

    # GPU选项
    parser.add_argument('--use_gpu', '-g', action='store_false', default=True,
                        help='Use GPU for inference (requires CUDA-enabled ONNXRuntime)')

    args = parser.parse_args()
    main(args)
