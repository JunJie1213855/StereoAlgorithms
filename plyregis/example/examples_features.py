#!/usr/bin/env python3
"""
PFH/FPFH特征配准专项示例
演示如何使用PFH和FPFH特征进行点云配准
"""
import os
import sys
import numpy as np
import open3d as o3d

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.features import PointCloudFeatures, feature_based_registration
from utils.feature_analysis import FeatureAnalyzer, comprehensive_feature_analysis
from utils.feature_visualization import FeatureVisualizer


def create_test_point_clouds():
    """创建测试点云"""
    print("创建测试点云...")
    
    # 创建一个复杂的几何体
    mesh1 = o3d.geometry.TriangleMesh.create_torus(radius=1.0, tube_radius=0.3)
    pcd1 = mesh1.sample_points_poisson_disk(number_of_points=1000)
    
    # 创建第二个几何体，变换后
    mesh2 = o3d.geometry.TriangleMesh.create_torus(radius=1.0, tube_radius=0.3)
    mesh2.translate([0.2, 0.3, 0.1])
    R = mesh2.get_rotation_matrix_from_xyz((0.15, 0.1, 0.05))
    mesh2.rotate(R)
    mesh2.scale(1.1, center=[0, 0, 0])
    pcd2 = mesh2.sample_points_poisson_disk(number_of_points=1000)
    
    # 添加颜色
    pcd1.paint_uniform_color([1, 0, 0])  # 红色
    pcd2.paint_uniform_color([0, 1, 0])  # 绿色
    
    print(f"创建点云1: {len(pcd1.points)} 点")
    print(f"创建点云2: {len(pcd2.points)} 点")
    
    return pcd1, pcd2


def example_basic_feature_registration():
    """示例1：基本特征配准"""
    print("=== 示例1：基本FPFH特征配准 ===")
    
    # 1. 创建测试数据
    source, target = create_test_point_clouds()
    
    # 2. 预处理
    print("\n预处理点云...")
    from core.preprocessing import PointCloudPreprocessor
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    
    source_processed = preprocessor.preprocess_pipeline(source)
    target_processed = preprocessor.preprocess_pipeline(target)
    
    # 3. 执行FPFH-RANSAC配准
    print("\n执行FPFH-RANSAC配准...")
    result = feature_based_registration(source_processed, target_processed, 
                                       method='fpfh_ransac')
    
    print(f"配准结果:")
    print(f"  - Fitness: {result['fitness']:.4f}")
    print(f"  - RMSE: {result['inlier_rmse']:.4f}")
    print(f"  - 对应点数量: {len(result['correspondence_set'])}")
    
    # 4. 应用变换并可视化
    source_aligned = source_processed.transform(result['transformation'])
    
    print("\n可视化结果...")
    visualizer = FeatureVisualizer()
    visualizer.visualize_feature_matching(
        source_processed, target_processed, result['correspondence_set'],
        result['transformation'], max_lines=30,
        window_name="FPFH-RANSAC配准结果"
    )
    
    # 5. 融合
    print("\n融合点云...")
    from core.fusion import fuse_point_clouds
    fused = fuse_point_clouds([source_aligned, target_processed], method='voxel')
    print(f"融合完成，最终点数: {len(fused.points)}")
    
    return source_processed, target_processed, result


def example_compare_fpfh_pfh():
    """示例2：比较FPFH和PFH"""
    print("\n=== 示例2：比较FPFH vs PFH ===")
    
    # 1. 创建测试数据
    source, target = create_test_point_clouds()
    
    # 2. 预处理
    print("预处理点云...")
    from core.preprocessing import PointCloudPreprocessor
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    
    source_processed = preprocessor.preprocess_pipeline(source)
    target_processed = preprocessor.preprocess_pipeline(target)
    
    # 3. 计算特征
    print("\n计算FPFH特征...")
    feature_calculator = PointCloudFeatures(voxel_size=0.05)
    source_fpfh = feature_calculator.compute_fpfh_features(source_processed)
    target_fpfh = feature_calculator.compute_fpfh_features(target_processed)
    
    print("\n计算PFH特征...")
    source_pfh = feature_calculator.compute_pfh_features(source_processed)
    target_pfh = feature_calculator.compute_pfh_features(target_processed)
    
    # 4. 可视化特征
    print("\n可视化特征...")
    visualizer = FeatureVisualizer()
    
    print("FPFH特征分布:")
    visualizer.visualize_feature_histograms(source_fpfh, "FPFH Feature Distribution")
    
    print("PFH特征分布:")
    visualizer.visualize_feature_histograms(source_pfh, "PFH Feature Distribution")
    
    print("FPFH vs PFH比较:")
    visualizer.compare_feature_distributions(source_fpfh, source_pfh, 
                                          "FPFH", "PFH", 
                                          "FPFH vs PFH Comparison")
    
    # 5. 特征分析
    print("\n分析特征质量...")
    analyzer = FeatureAnalyzer()
    
    fpfh_analysis = analyzer.analyze_feature_quality(source_fpfh, source_processed)
    pfh_analysis = analyzer.analyze_feature_quality(source_pfh, source_processed)
    
    print(f"\nFPFH特征:")
    print(f"  - 维度: {fpfh_analysis['num_features']}")
    print(f"  - 稀疏度: {fpfh_analysis['feature_sparsity']:.4f}")
    print(f"  - 熵: {fpfh_analysis['feature_entropy']:.4f}")
    
    print(f"\nPFH特征:")
    print(f"  - 维度: {pfh_analysis['num_features']}")
    print(f"  - 稀疏度: {pfh_analysis['feature_sparsity']:.4f}")
    print(f"  - 熵: {pfh_analysis['feature_entropy']:.4f}")


def example_hybrid_registration():
    """示例3：混合特征配准"""
    print("\n=== 示例3：混合特征配准 ===")
    
    # 1. 创建测试数据
    source, target = create_test_point_clouds()
    
    # 2. 预处理
    print("预处理点云...")
    from core.preprocessing import PointCloudPreprocessor
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    
    source_processed = preprocessor.preprocess_pipeline(source)
    target_processed = preprocessor.preprocess_pipeline(target)
    
    # 3. 比较不同方法
    methods = ['fpfh_ransac', 'pfh_ransac', 'hybrid_fpfh', 'hybrid_pfh']
    results = {}
    
    for method in methods:
        print(f"\n测试方法: {method}")
        try:
            result = feature_based_registration(source_processed, target_processed, 
                                              method=method)
            results[method] = result
            
            print(f"  ✓ 成功")
            print(f"    Fitness: {result['fitness']:.4f}")
            print(f"    RMSE: {result['inlier_rmse']:.4f}")
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results[method] = {'error': str(e)}
    
    # 4. 可视化比较结果
    print("\n可视化方法比较...")
    visualizer = FeatureVisualizer()
    visualizer.visualize_registration_comparison(results)
    
    return results


def example_feature_analysis_dashboard():
    """示例4：特征分析仪表板"""
    print("\n=== 示例4：综合特征分析 ===")
    
    # 1. 创建测试数据
    source, target = create_test_point_clouds()
    
    # 2. 预处理
    print("预处理点云...")
    from core.preprocessing import PointCloudPreprocessor
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    
    source_processed = preprocessor.preprocess_pipeline(source)
    target_processed = preprocessor.preprocess_pipeline(target)
    
    # 3. 计算特征
    print("计算特征...")
    feature_calculator = PointCloudFeatures(voxel_size=0.05)
    source_fpfh = feature_calculator.compute_fpfh_features(source_processed)
    target_fpfh = feature_calculator.compute_fpfh_features(target_processed)
    
    # 4. 创建特征分析仪表板
    print("创建特征分析仪表板...")
    visualizer = FeatureVisualizer()
    visualizer.create_feature_dashboard(
        source_fpfh, target_fpfh, source_processed, target_processed,
        save_path="feature_dashboard.png"
    )
    
    # 5. 执行基准测试
    print("执行基准测试...")
    analyzer = FeatureAnalyzer()
    benchmark_results = analyzer.benchmark_feature_methods(
        source_processed, target_processed,
        methods=['fpfh_ransac', 'pfh_ransac', 'hybrid_fpfh']
    )
    
    return benchmark_results


def example_feature_descriptiveness():
    """示例5：特征描述性分析"""
    print("\n=== 示例5：特征描述性分析 ===")
    
    # 1. 创建测试数据
    source, target = create_test_point_clouds()
    
    # 2. 预处理
    print("预处理点云...")
    from core.preprocessing import PointCloudPreprocessor
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    
    source_processed = preprocessor.preprocess_pipeline(source)
    
    # 3. 计算特征
    print("计算特征...")
    feature_calculator = PointCloudFeatures(voxel_size=0.05)
    source_fpfh = feature_calculator.compute_fpfh_features(source_processed)
    source_pfh = feature_calculator.compute_pfh_features(source_processed)
    
    # 4. 分析特征描述性
    print("分析特征描述性...")
    analyzer = FeatureAnalyzer()
    
    fpfh_descript = analyzer.analyze_feature_descriptiveness(source_fpfh, source_processed)
    pfh_descript = analyzer.analyze_feature_descriptiveness(source_pfh, source_processed)
    
    print(f"\nFPFH描述性分析:")
    print(f"  - 平均最小距离: {fpfh_descript['mean_min_distance']:.4f}")
    print(f"  - 描述性得分: {fpfh_descript['descriptiveness_score']:.4f}")
    print(f"  - 特征可分性: {fpfh_descript['feature_separability']:.4f}")
    
    print(f"\nPFH描述性分析:")
    print(f"  - 平均最小距离: {pfh_descript['mean_min_distance']:.4f}")
    print(f"  - 描述性得分: {pfh_descript['descriptiveness_score']:.4f}")
    print(f"  - 特征可分性: {pfh_descript['feature_separability']:.4f}")
    
    # 5. 可视化特征统计
    print("\n可视化特征统计...")
    visualizer = FeatureVisualizer()
    visualizer.visualize_feature_statistics(source_fpfh, "FPFH Statistics")
    visualizer.visualize_feature_statistics(source_pfh, "PFH Statistics")


def main():
    """主函数"""
    print("PFH/FPFH特征配准专项示例")
    print("=" * 50)
    
    examples = [
        ("基本FPFH特征配准", example_basic_feature_registration),
        ("比较FPFH vs PFH", example_compare_fpfh_pfh),
        ("混合特征配准", example_hybrid_registration),
        ("综合特征分析", example_feature_analysis_dashboard),
        ("特征描述性分析", example_feature_descriptiveness)
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n示例 {i}: {name}")
        print("-" * 40)
        try:
            func()
            print(f"✓ 示例 {i} 完成")
        except Exception as e:
            print(f"✗ 示例 {i} 出错: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)
    
    print("\n所有PFH/FPFH示例演示完成！")


if __name__ == '__main__':
    main()