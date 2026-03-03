#!/usr/bin/env python3
"""
示例脚本：演示如何使用点云配准融合系统的各个模块
"""
import os
import sys
import numpy as np
import open3d as o3d

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.preprocessing import PointCloudPreprocessor
from core.registration import PointCloudRegistration, pairwise_registration
from core.fusion import PointCloudFusion, fuse_point_clouds
from utils.visualization import PointCloudVisualizer, quick_visualize
from utils.evaluation import RegistrationEvaluator, FusionEvaluator


def create_sample_point_clouds():
    """创建示例点云用于演示"""
    print("创建示例点云...")
    
    # 创建一个简单的立方体点云
    cube1 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    pcd1 = cube1.sample_points_poisson_disk(number_of_points=1000)
    
    # 创建另一个立方体，稍微偏移
    cube2 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    cube2.translate([0.1, 0.1, 0.1])
    # 使用旋转矩阵而不是角度
    R = cube2.get_rotation_matrix_from_xyz((0.1, 0, 0))
    cube2.rotate(R)
    pcd2 = cube2.sample_points_poisson_disk(number_of_points=1000)
    
    # 为点云添加颜色
    pcd1.paint_uniform_color([1, 0, 0])  # 红色
    pcd2.paint_uniform_color([0, 1, 0])  # 绿色
    
    return [pcd1, pcd2]


def example_basic_usage():
    """基本使用示例"""
    print("=== 示例1：基本使用 ===")
    
    # 1. 创建示例点云
    point_clouds = create_sample_point_clouds()
    
    # 2. 预处理
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    processed_clouds = [preprocessor.preprocess_pipeline(pcd) for pcd in point_clouds]
    
    # 3. 配准
    print("\n执行点云配准...")
    result = pairwise_registration(processed_clouds[0], processed_clouds[1], 
                                  method='coarse_to_fine')
    print(f"配准结果 - Fitness: {result['fitness']:.4f}, RMSE: {result['inlier_rmse']:.4f}")
    
    # 4. 应用变换
    aligned_pcd = processed_clouds[1].transform(result['transformation'])
    
    # 5. 融合
    print("\n执行点云融合...")
    fused = fuse_point_clouds([processed_clouds[0], aligned_pcd], method='voxel')
    print(f"融合完成，最终点数: {len(fused.points)}")
    
    # 6. 可视化
    print("\n可视化结果...")
    visualizer = PointCloudVisualizer()
    visualizer.visualize_multiple_point_clouds([processed_clouds[0], aligned_pcd, fused],
                                             names=["原始点云1", "配准后的点云2", "融合结果"])


def example_preprocessing():
    """预处理示例"""
    print("=== 示例2：预处理功能 ===")
    
    # 创建带噪声的点云
    pcd = create_sample_point_clouds()[0]
    
    # 添加一些噪声
    points = np.asarray(pcd.points)
    noise = np.random.normal(0, 0.01, points.shape)
    noisy_points = points + noise
    pcd_noisy = o3d.geometry.PointCloud()
    pcd_noisy.points = o3d.utility.Vector3dVector(noisy_points)
    pcd_noisy.colors = pcd.colors
    
    print(f"原始点云: {len(pcd.points)} 点")
    print(f"噪声点云: {len(pcd_noisy.points)} 点")
    
    # 预处理
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    
    # 下采样
    downsampled = preprocessor.voxel_down_sample(pcd_noisy)
    print(f"下采样后: {len(downsampled.points)} 点")
    
    # 离群点滤波
    filtered, _ = preprocessor.remove_statistical_outliers(downsampled)
    print(f"滤波后: {len(filtered.points)} 点")
    
    # 法向量估计
    with_normals = preprocessor.estimate_normals(filtered)
    print(f"法向量估计完成: {len(with_normals.points)} 点, 有法向量: {with_normals.has_normals()}")
    
    # 可视化对比
    visualizer = PointCloudVisualizer()
    visualizer.compare_before_after(pcd_noisy, with_normals, 
                                   title_before="原始噪声点云", 
                                   title_after="预处理后")


def example_registration_methods():
    """不同配准方法比较"""
    print("=== 示例3：配准方法比较 ===")
    
    # 创建示例点云
    point_clouds = create_sample_point_clouds()
    
    # 预处理
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    processed_clouds = [preprocessor.preprocess_pipeline(pcd) for pcd in point_clouds]
    
    source = processed_clouds[0]
    target = processed_clouds[1]
    
    methods = ['icp', 'ndt', 'coarse_to_fine']
    results = {}
    
    for method in methods:
        print(f"\n使用 {method} 方法配准...")
        result = pairwise_registration(source, target, method=method)
        results[method] = result
        print(f"  - Fitness: {result['fitness']:.4f}")
        print(f"  - RMSE: {result['inlier_rmse']:.4f}")
    
    # 可视化最佳结果
    best_method = max(results.keys(), key=lambda k: results[k]['fitness'])
    print(f"\n最佳方法: {best_method}")
    
    # 应用最佳变换并可视化
    transformation = results[best_method]['transformation']
    aligned = source.transform(transformation)
    
    visualizer = PointCloudVisualizer()
    visualizer.visualize_registration_result(source, target, aligned, transformation)


def example_fusion_methods():
    """不同融合方法比较"""
    print("=== 示例4：融合方法比较 ===")
    
    # 创建示例点云
    point_clouds = create_sample_point_clouds()
    
    # 预处理
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    processed_clouds = [preprocessor.preprocess_pipeline(pcd) for pcd in point_clouds]
    
    # 配准
    result = pairwise_registration(processed_clouds[0], processed_clouds[1])
    aligned = processed_clouds[1].transform(result['transformation'])
    
    fusion_methods = ['simple', 'voxel', 'statistical', 'color_aware']
    fusion_results = {}
    
    for method in fusion_methods:
        print(f"\n使用 {method} 方法融合...")
        fusion = PointCloudFusion(voxel_size=0.03)
        fused = fusion.smart_fusion([processed_clouds[0], aligned], method=method)
        fusion_results[method] = fused
        print(f"  - 融合后点数: {len(fused.points)}")
    
    # 可视化不同融合结果
    print("\n可视化融合结果...")
    visualizer = PointCloudVisualizer()
    
    for method, fused in fusion_results.items():
        print(f"显示 {method} 融合结果...")
        visualizer.visualize_single_point_cloud(fused, window_name=f"{method} 融合结果")


def example_evaluation():
    """评估示例"""
    print("=== 示例5：质量评估 ===")
    
    # 创建和处理点云
    point_clouds = create_sample_point_clouds()
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    processed_clouds = [preprocessor.preprocess_pipeline(pcd) for pcd in point_clouds]
    
    # 配准
    registration = PointCloudRegistration()
    result = registration.coarse_to_fine_registration(processed_clouds[0], processed_clouds[1])
    aligned = processed_clouds[1].transform(result['transformation'])
    
    # 评估配准质量
    print("评估配准质量...")
    reg_evaluator = RegistrationEvaluator()
    eval_result = reg_evaluator.evaluate_registration_result(
        processed_clouds[0], processed_clouds[1], result['transformation'])
    
    print(f"配准评估结果:")
    print(f"  - Fitness: {eval_result['fitness']:.4f}")
    print(f"  - RMSE: {eval_result['inlier_rmse']:.4f}")
    print(f"  - 重叠率: {eval_result['overlap_ratio']:.4f}")
    print(f"  - 平均距离: {eval_result['mean_distance']:.4f}")
    
    # 融合和评估
    fusion = PointCloudFusion(voxel_size=0.03)
    fused = fusion.smart_fusion([processed_clouds[0], aligned], method='voxel')
    
    print("\n评估融合质量...")
    fusion_evaluator = FusionEvaluator()
    fusion_result = fusion_evaluator.evaluate_fusion_result(processed_clouds, fused)
    
    print(f"融合评估结果:")
    print(f"  - 原始点云总数: {fusion_result['total_original_points']}")
    print(f"  - 融合后点数: {fusion_result['fused_points']}")
    print(f"  - 压缩比: {fusion_result['compression_ratio']:.2%}")
    print(f"  - 减少百分比: {fusion_result['reduction_percentage']:.1f}%")


def main():
    """运行所有示例"""
    examples = [
        ("基本使用", example_basic_usage),
        ("预处理功能", example_preprocessing),
        ("配准方法比较", example_registration_methods),
        ("融合方法比较", example_fusion_methods),
        ("质量评估", example_evaluation)
    ]
    
    print("点云配准融合系统 - 使用示例")
    print("=" * 50)
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n示例 {i}: {name}")
        print("-" * 40)
        try:
            func()
            print(f"✓ 示例 {i} 完成")
        except Exception as e:
            print(f"✗ 示例 {i} 出错: {e}")
        
        print("\n" + "=" * 50)
    
    print("\n所有示例演示完成！")


if __name__ == '__main__':
    main()