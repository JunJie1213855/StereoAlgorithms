#!/usr/bin/env python3
"""
FPFH/PFH功能测试脚本
"""
import sys
import os
import numpy as np
import open3d as o3d

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_feature_import():
    """测试特征模块导入"""
    print("=== 测试特征模块导入 ===")
    
    try:
        from core.features import PointCloudFeatures, feature_based_registration
        from utils.feature_analysis import FeatureAnalyzer
        from utils.feature_visualization import FeatureVisualizer
        print("✓ 所有特征模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def test_basic_features():
    """测试基本特征功能"""
    print("\n=== 测试基本特征功能 ===")
    
    try:
        from core.features import PointCloudFeatures
        
        # 创建简单测试点云
        print("创建测试点云...")
        cube1 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        pcd1 = cube1.sample_points_poisson_disk(number_of_points=200)
        pcd1.paint_uniform_color([1, 0, 0])
        
        cube2 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube2.translate([0.05, 0.05, 0.05])
        R = cube2.get_rotation_matrix_from_xyz((0.1, 0, 0))
        cube2.rotate(R)
        pcd2 = cube2.sample_points_poisson_disk(number_of_points=200)
        pcd2.paint_uniform_color([0, 1, 0])
        
        # 测试特征计算
        print("测试FPFH特征计算...")
        feature_calc = PointCloudFeatures(voxel_size=0.1)
        
        fpfh1 = feature_calc.compute_fpfh_features(pcd1)
        print(f"  ✓ FPFH特征1: {fpfh1.dimension}维, {len(pcd1.points)}点")
        
        fpfh2 = feature_calc.compute_fpfh_features(pcd2)
        print(f"  ✓ FPFH特征2: {fpfh2.dimension}维, {len(pcd2.points)}点")
        
        # 测试PFH特征
        print("测试PFH特征计算...")
        pfh1 = feature_calc.compute_pfh_features(pcd1)
        print(f"  ✓ PFH特征1: {pfh1.dimension}维")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本特征功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_registration():
    """测试特征配准功能"""
    print("\n=== 测试特征配准功能 ===")
    
    try:
        from core.features import feature_based_registration
        from core.preprocessing import PointCloudPreprocessor
        
        # 创建测试点云
        print("创建测试点云...")
        cube1 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        pcd1 = cube1.sample_points_poisson_disk(number_of_points=300)
        pcd1.paint_uniform_color([1, 0, 0])
        
        cube2 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube2.translate([0.1, 0.1, 0.1])
        R = cube2.get_rotation_matrix_from_xyz((0.1, 0, 0))
        cube2.rotate(R)
        pcd2 = cube2.sample_points_poisson_disk(number_of_points=300)
        pcd2.paint_uniform_color([0, 1, 0])
        
        # 预处理
        print("预处理点云...")
        preprocessor = PointCloudPreprocessor(voxel_size=0.1)
        pcd1_proc = preprocessor.preprocess_pipeline(pcd1)
        pcd2_proc = preprocessor.preprocess_pipeline(pcd2)
        
        # 测试FPFH-RANSAC配准
        print("测试FPFH-RANSAC配准...")
        result_fpfh = feature_based_registration(pcd1_proc, pcd2_proc, method='fpfh_ransac')
        print(f"  ✓ FPFH-RANSAC配准完成")
        print(f"    Fitness: {result_fpfh['fitness']:.4f}")
        print(f"    RMSE: {result_fpfh['inlier_rmse']:.4f}")
        print(f"    对应点: {len(result_fpfh['correspondence_set'])}")
        
        # 测试混合配准
        print("测试混合FPFH配准...")
        result_hybrid = feature_based_registration(pcd1_proc, pcd2_proc, method='hybrid_fpfh')
        print(f"  ✓ 混合FPFH配准完成")
        print(f"    Fitness: {result_hybrid['fitness']:.4f}")
        print(f"    RMSE: {result_hybrid['inlier_rmse']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 特征配准功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_analysis():
    """测试特征分析功能"""
    print("\n=== 测试特征分析功能 ===")
    
    try:
        from core.features import PointCloudFeatures
        from utils.feature_analysis import FeatureAnalyzer
        
        # 创建测试点云
        print("创建测试点云...")
        cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        pcd = cube.sample_points_poisson_disk(number_of_points=200)
        
        # 计算特征
        print("计算特征...")
        feature_calc = PointCloudFeatures(voxel_size=0.1)
        fpfh = feature_calc.compute_fpfh_features(pcd)
        
        # 特征质量分析
        print("分析特征质量...")
        analyzer = FeatureAnalyzer()
        quality = analyzer.analyze_feature_quality(fpfh, pcd)
        print(f"  ✓ 特征质量分析完成")
        print(f"    特征维度: {quality['num_features']}")
        print(f"    特征稀疏度: {quality['feature_sparsity']:.4f}")
        print(f"    特征熵: {quality['feature_entropy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 特征分析功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("FPFH/PFH功能测试")
    print("=" * 50)
    
    test_results = []
    
    # 测试模块导入
    test_results.append(("模块导入", test_feature_import()))
    
    # 测试基本功能
    if test_results[0][1]:  # 如果导入成功才继续测试
        test_results.append(("基本特征功能", test_basic_features()))
        test_results.append(("特征配准功能", test_feature_registration()))
        test_results.append(("特征分析功能", test_feature_analysis()))
    
    # 总结结果
    print("\n" + "=" * 50)
    print("测试结果总结:")
    
    passed = 0
    failed = 0
    
    for name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有功能测试通过！")
        return 0
    else:
        print("⚠️  部分功能测试失败，请检查错误信息")
        return 1


if __name__ == '__main__':
    sys.exit(main())