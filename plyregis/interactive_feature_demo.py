#!/usr/bin/env python3
"""
交互式特征配准演示
使用Open3D实时展示特征配准过程
"""
import os
import sys
import numpy as np
import open3d as o3d
import copy

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.features import PointCloudFeatures
from core.preprocessing import PointCloudPreprocessor
from utils.feature_visualization import FeatureVisualizer


def interactive_feature_demo():
    """交互式特征配准演示"""
    print("🎯 交互式特征配准演示")
    print("=" * 50)
    
    # 1. 创建演示点云
    print("\n📦 创建演示点云...")
    
    # 创建一个复杂的几何体
    def create_complex_shape():
        # 创建一个茶壶形状
        mesh = o3d.geometry.TriangleMesh.create_torus_knot(radius=1, tube_radius=0.3, 
                                                       tubular_segments=20, radial_segments=20)
        return mesh
    
    mesh1 = create_complex_shape()
    pcd1 = mesh1.sample_points_poisson_disk(number_of_points=500)
    pcd1.paint_uniform_color([1, 0, 0])  # 红色
    
    mesh2 = create_complex_shape()
    mesh2.translate([0.3, 0.2, 0.1])
    R = mesh2.get_rotation_matrix_from_xyz((0.2, 0.15, 0.1))
    mesh2.rotate(R)
    mesh2.scale(1.1, center=[0, 0, 0])
    pcd2 = mesh2.sample_points_poisson_disk(number_of_points=500)
    pcd2.paint_uniform_color([0, 1, 0])  # 绿色
    
    print(f"   源点云: {len(pcd1.points)} 点")
    print(f"   目标点云: {len(pcd2.points)} 点")
    
    # 2. 显示原始点云
    print("\n👀 步骤1：查看原始点云（有位姿差异）")
    print("   红色：源点云 | 绿色：目标点云")
    
    temp_pcd1 = copy.deepcopy(pcd1)
    temp_pcd2 = copy.deepcopy(pcd2)
    
    # 如果没有颜色，设置颜色
    if not temp_pcd1.has_colors():
        temp_pcd1.paint_uniform_color([1, 0, 0])
    if not temp_pcd2.has_colors():
        temp_pcd2.paint_uniform_color([0, 1, 0])
    
    o3d.visualization.draw_geometries([temp_pcd1, temp_pcd2],
                                     window_name="原始点云 - 可见位姿差异")
    
    # 3. 预处理
    print("\n⚙️ 步骤2：预处理点云...")
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    
    pcd1_proc = preprocessor.preprocess_pipeline(pcd1, downsample=True, 
                                                remove_outliers=True, estimate_normals=True)
    pcd2_proc = preprocessor.preprocess_pipeline(pcd2, downsample=True, 
                                                remove_outliers=True, estimate_normals=True)
    
    print(f"   预处理后点云1: {len(pcd1_proc.points)} 点")
    print(f"   预处理后点云2: {len(pcd2_proc.points)} 点")
    
    # 4. 计算FPFH特征
    print("\n🧮 步骤3：计算FPFH特征...")
    feature_calc = PointCloudFeatures(voxel_size=0.05)
    
    pcd1_features = feature_calc.compute_fpfh_features(pcd1_proc)
    pcd2_features = feature_calc.compute_fpfh_features(pcd2_proc)
    
    print(f"   FPFH特征维度: {pcd1_features.dimension()}")
    print(f"   特征计算完成")
    
    # 5. 显示特征匹配过程
    print("\n🎯 步骤4：执行FPFH-RANSAC配准...")
    
    # 先显示特征匹配预览
    print("   计算特征对应关系...")
    result = feature_calc.execute_fpfh_ransac_registration(pcd1_proc, pcd2_proc)
    
    print(f"   配准结果:")
    print(f"   - Fitness: {result['fitness']:.4f}")
    print(f"   - RMSE: {result['inlier_rmse']:.4f}")
    print(f"   - 对应点数量: {len(result['correspondence_set'])}")
    
    # 6. 显示配准结果
    print("\n🔧 步骤5：查看配准结果")
    
    # 应用变换
    pcd1_aligned = pcd1_proc.transform(result['transformation'])
    
    # 设置颜色
    pcd1_aligned_vis = copy.deepcopy(pcd1_aligned)
    pcd2_vis = copy.deepcopy(pcd2_proc)
    
    if not pcd1_aligned_vis.has_colors():
        pcd1_aligned_vis.paint_uniform_color([1, 0, 0])  # 红色
    if not pcd2_vis.has_colors():
        pcd2_vis.paint_uniform_color([0, 1, 0])  # 绿色
    
    print("   红色：配准后的源点云 | 绿色：目标点云")
    o3d.visualization.draw_geometries([pcd1_aligned_vis, pcd2_vis],
                                     window_name="FPFH-RANSAC配准结果")
    
    # 7. 显示对应关系
    if len(result['correspondence_set']) > 0:
        print(f"\n🔗 步骤6：查看特征对应关系（前20个）")
        
        visualizer = FeatureVisualizer()
        visualizer.visualize_feature_matching(
            pcd1_proc, pcd2_proc, result['correspondence_set'],
            result['transformation'], max_lines=20,
            window_name="特征对应关系 - 黄线显示匹配"
        )
    
    # 8. 融合结果
    print("\n🎨 步骤7：融合点云...")
    
    from core.fusion import fuse_point_clouds
    fused = fuse_point_clouds([pcd1_aligned, pcd2_proc], method='color_aware')
    
    print(f"   融合完成，最终点数: {len(fused.points)}")
    
    # 确保融合点云有颜色
    fused_vis = copy.deepcopy(fused)
    if not fused_vis.has_colors():
        # 根据位置生成颜色
        points = np.asarray(fused_vis.points)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min() + 1e-6)
        colors[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min() + 1e-6)
        colors[:, 2] = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-6)
        fused_vis.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([fused_vis], window_name="融合后的点云")
    
    # 9. 比较不同方法
    print("\n⚖️ 步骤8：比较不同特征配准方法...")
    
    methods = ['fpfh_ransac', 'hybrid_fpfh']
    method_results = {}
    
    for method in methods:
        print(f"\n   测试 {method}...")
        try:
            from core.features import feature_based_registration
            method_result = feature_based_registration(pcd1_proc, pcd2_proc, method=method)
            method_results[method] = method_result
            
            print(f"   ✓ {method} 完成")
            print(f"     Fitness: {method_result['fitness']:.4f}")
            print(f"     RMSE: {method_result['inlier_rmse']:.4f}")
            
        except Exception as e:
            print(f"   ✗ {method} 失败: {e}")
    
    # 10. 最终总结
    print("\n" + "=" * 50)
    print("📊 特征配准演示总结")
    print("=" * 50)
    
    print(f"\n🎯 主要成果:")
    print(f"  ✓ 成功配准 {len(pcd1_proc.points)} + {len(pcd2_proc.points)} = {len(fused.points)} 点")
    print(f"  ✓ FPFH特征维度: {pcd1_features.dimension()}")
    print(f"  ✓ 配准精度: Fitness={result['fitness']:.4f}, RMSE={result['inlier_rmse']:.4f}")
    
    if method_results:
        best_method = max(method_results.keys(), 
                         key=lambda k: method_results[k]['fitness'])
        print(f"  ✓ 最佳方法: {best_method}")
        print(f"    - Fitness: {method_results[best_method]['fitness']:.4f}")
        print(f"    - RMSE: {method_results[best_method]['inlier_rmse']:.4f}")
    
    print(f"\n💡 特征配准的优势:")
    print(f"  • 能够处理大位姿差异")
    print(f"  • 基于几何特征，鲁棒性强")
    print(f"  • FPFH速度快，PFH精度高")
    print(f"  • 混合方法结合两者优势")
    
    print(f"\n🚀 应用场景:")
    print(f"  • 多视角3D重建")
    print(f"  • SLAM后端优化")
    print(f"  • 物体识别与配准")
    print(f"  • 点云地图融合")
    
    # 11. 保存结果
    print(f"\n💾 保存结果...")
    
    output_dir = "./feature_registration_output"
    os.makedirs(output_dir, exist_ok=True)
    
    o3d.io.write_point_cloud(f"{output_dir}/source_aligned.ply", pcd1_aligned)
    o3d.io.write_point_cloud(f"{output_dir}/target.ply", pcd2_proc)
    o3d.io.write_point_cloud(f"{output_dir}/fused.ply", fused)
    
    print(f"   ✓ 对齐源点云: {output_dir}/source_aligned.ply")
    print(f"   ✓ 目标点云: {output_dir}/target.ply")
    print(f"   ✓ 融合点云: {output_dir}/fused.ply")
    
    print(f"\n🎉 特征配准演示完成！")
    print(f"   所有处理后的点云已保存到: {output_dir}/")


def quick_feature_demo():
    """快速特征配准演示"""
    print("🚀 快速特征配准演示")
    print("=" * 30)
    
    # 简单创建点云
    print("创建测试点云...")
    cube1 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    pcd1 = cube1.sample_points_poisson_disk(number_of_points=300)
    pcd1.paint_uniform_color([1, 0, 0])
    
    cube2 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    cube2.translate([0.2, 0.2, 0.1])
    R = cube2.get_rotation_matrix_from_xyz((0.3, 0.2, 0.1))
    cube2.rotate(R)
    pcd2 = cube2.sample_points_poisson_disk(number_of_points=300)
    pcd2.paint_uniform_color([0, 1, 0])
    
    # 预处理
    print("预处理...")
    preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    pcd1_proc = preprocessor.preprocess_pipeline(pcd1)
    pcd2_proc = preprocessor.preprocess_pipeline(pcd2)
    
    # 特征配准
    print("执行FPFH-RANSAC配准...")
    feature_calc = PointCloudFeatures(voxel_size=0.05)
    result = feature_calc.execute_fpfh_ransac_registration(pcd1_proc, pcd2_proc)
    
    print(f"配准结果: Fitness={result['fitness']:.4f}, RMSE={result['inlier_rmse']:.4f}")
    
    # 应用变换
    pcd1_aligned = pcd1_proc.transform(result['transformation'])
    
    # 显示结果
    print("显示配准结果...")
    
    pcd1_vis = copy.deepcopy(pcd1_aligned)
    pcd2_vis = copy.deepcopy(pcd2_proc)
    
    if not pcd1_vis.has_colors():
        pcd1_vis.paint_uniform_color([1, 0, 0])
    if not pcd2_vis.has_colors():
        pcd2_vis.paint_uniform_color([0, 1, 0])
    
    # 添加对应线
    if len(result['correspondence_set']) > 0:
        source_points = np.asarray(pcd1_vis.points)
        target_points = np.asarray(pcd2_vis.points)
        
        line_points = []
        line_indices = []
        
        max_lines = min(15, len(result['correspondence_set']))
        for i in range(max_lines):
            src_idx, tgt_idx = result['correspondence_set'][i]
            if src_idx < len(source_points) and tgt_idx < len(target_points):
                line_points.append(source_points[src_idx])
                line_points.append(target_points[tgt_idx])
                line_indices.append([2*i, 2*i+1])
        
        if len(line_points) > 0:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
            line_set.lines = o3d.utility.Vector2iVector(np.array(line_indices))
            
            line_colors = np.tile([1, 1, 0], (len(line_indices), 1))  # 黄色
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            
            o3d.visualization.draw_geometries(
                [pcd1_vis, pcd2_vis, line_set],
                window_name="快速FPFH配准演示 - 黄线显示特征匹配"
            )
    else:
        o3d.visualization.draw_geometries([pcd1_vis, pcd2_vis])


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征配准可视化演示')
    parser.add_argument('--mode', choices=['interactive', 'quick'], 
                       default='quick', help='演示模式')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示3D可视化窗口')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_feature_demo()
    else:
        quick_feature_demo()


if __name__ == '__main__':
    main()