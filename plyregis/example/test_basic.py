#!/usr/bin/env python3
"""
简单测试：验证系统基本功能
"""
import sys
import os
import numpy as np
import open3d as o3d

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.preprocessing import PointCloudPreprocessor
from core.registration import PointCloudRegistration
from core.fusion import PointCloudFusion

def test_basic_functionality():
    """测试基本功能"""
    print("=== 基本功能测试 ===")
    
    # 创建简单测试点云
    print("1. 创建测试点云...")
    cube1 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    pcd1 = cube1.sample_points_poisson_disk(number_of_points=500)
    pcd1.paint_uniform_color([1, 0, 0])
    
    cube2 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    cube2.translate([0.05, 0.05, 0.05])
    R = cube2.get_rotation_matrix_from_xyz((0.05, 0, 0))
    cube2.rotate(R)
    pcd2 = cube2.sample_points_poisson_disk(number_of_points=500)
    pcd2.paint_uniform_color([0, 1, 0])
    
    print(f"   点云1: {len(pcd1.points)} 点")
    print(f"   点云2: {len(pcd2.points)} 点")
    
    # 测试预处理
    print("\n2. 测试预处理...")
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    
    processed1 = preprocessor.preprocess_pipeline(pcd1)
    processed2 = preprocessor.preprocess_pipeline(pcd2)
    
    print(f"   预处理完成")
    print(f"   处理后点云1: {len(processed1.points)} 点")
    print(f"   处理后点云2: {len(processed2.points)} 点")
    
    # 测试ICP配准
    print("\n3. 测试ICP配准...")
    registration = PointCloudRegistration(voxel_size=0.05)
    
    result = registration.execute_icp(processed1, processed2)
    print(f"   ICP配准完成")
    print(f"   Fitness: {result['fitness']:.4f}")
    print(f"   RMSE: {result['inlier_rmse']:.4f}")
    
    # 应用变换
    aligned = processed2.transform(result['transformation'])
    
    # 测试融合
    print("\n4. 测试点云融合...")
    fusion = PointCloudFusion(voxel_size=0.03)
    
    fused = fusion.simple_merge([processed1, aligned])
    print(f"   简单合并: {len(fused.points)} 点")
    
    fused_voxel = fusion.voxel_based_fusion([processed1, aligned])
    print(f"   体素融合: {len(fused_voxel.points)} 点")
    
    # 保存结果
    print("\n5. 保存测试结果...")
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    o3d.io.write_point_cloud(f"{output_dir}/test_fused.ply", fused_voxel)
    print(f"   结果已保存到: {output_dir}/test_fused.ply")
    
    print("\n=== 基本功能测试完成 ===")
    print("✓ 所有基本功能正常工作")

if __name__ == '__main__':
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)