#!/usr/bin/env python3
"""
特征配准可视化演示
完整展示FPFH/PFH特征配准的可视化过程
"""
import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.features import PointCloudFeatures, feature_based_registration
from core.preprocessing import PointCloudPreprocessor
from utils.feature_analysis import FeatureAnalyzer
from utils.feature_visualization import FeatureVisualizer


class FeatureRegistrationVisualizer:
    """特征配准可视化演示类"""
    
    def __init__(self):
        """初始化可视化演示"""
        self.feature_calculator = PointCloudFeatures(voxel_size=0.05)
        self.analyzer = FeatureAnalyzer()
        self.visualizer = FeatureVisualizer()
        self.preprocessor = PointCloudPreprocessor(voxel_size=0.05)
    
    def create_demo_point_clouds(self):
        """创建演示用的点云对"""
        print("创建演示点云...")
        
        # 创建一个复杂的几何体 - 兔子形状（使用多个球体组合）
        def create_rabbit():
            parts = []
            
            # 身体
            body = o3d.geometry.TriangleMesh.create_sphere(radius=0.8)
            body.translate([0, 0, 0])
            parts.append(body)
            
            # 头
            head = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            head.translate([0, 0, 1.0])
            parts.append(head)
            
            # 耳朵
            ear1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.2, resolution=20)
            ear1.translate([0.3, 0, 1.5])
            ear1.scale(3.0, center=[0.3, 0, 1.5])
            parts.append(ear1)
            
            ear2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.2, resolution=20)
            ear2.translate([-0.3, 0, 1.5])
            ear2.scale(3.0, center=[-0.3, 0, 1.5])
            parts.append(ear2)
            
            # 合并所有部分
            rabbit = parts[0]
            for part in parts[1:]:
                rabbit += part
            
            return rabbit
        
        # 创建原始点云
        rabbit1 = create_rabbit()
        pcd1 = rabbit1.sample_points_poisson_disk(number_of_points=800)
        
        # 创建变换后的点云
        rabbit2 = create_rabbit()
        rabbit2.translate([0.3, 0.2, 0.1])
        R = rabbit2.get_rotation_matrix_from_xyz((0.2, 0.15, 0.1))
        rabbit2.rotate(R)
        rabbit2.scale(1.1, center=[0, 0, 0])
        pcd2 = rabbit2.sample_points_poisson_disk(number_of_points=800)
        
        # 添加颜色
        pcd1.paint_uniform_color([1, 0.3, 0.3])  # 红色
        pcd2.paint_uniform_color([0.3, 1, 0.3])  # 绿色
        
        print(f"创建点云1: {len(pcd1.points)} 点 (红色)")
        print(f"创建点云2: {len(pcd2.points)} 点 (绿色)")
        
        return pcd1, pcd2
    
    def visualize_step1_original_clouds(self, source, target):
        """步骤1：可视化原始点云"""
        print("\n=== 步骤1：原始点云可视化 ===")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Step 1: Original Point Clouds', fontsize=16, fontweight='bold')
        
        # 原始源点云
        points1 = np.asarray(source.points)
        colors1 = np.asarray(source.colors)
        axes[0].scatter(points1[:, 0], points1[:, 1], c=colors1, s=1, alpha=0.6)
        axes[0].set_title('Source Point Cloud (Red)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        
        # 原始目标点云
        points2 = np.asarray(target.points)
        colors2 = np.asarray(target.colors)
        axes[1].scatter(points2[:, 0], points2[:, 1], c=colors2, s=1, alpha=0.6)
        axes[1].set_title('Target Point Cloud (Green)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        
        # 叠加显示
        axes[2].scatter(points1[:, 0], points1[:, 1], c=colors1, s=1, alpha=0.5, label='Source')
        axes[2].scatter(points2[:, 0], points2[:, 1], c=colors2, s=1, alpha=0.5, label='Target')
        axes[2].set_title('Overlapped View (Misalignment Visible)')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        axes[2].set_aspect('equal')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('step1_original_clouds.png', dpi=300, bbox_inches='tight')
        print("✓ 保存: step1_original_clouds.png")
        plt.show()
    
    def visualize_step2_preprocessing(self, source, target):
        """步骤2：预处理可视化"""
        print("\n=== 步骤2：预处理过程 ===")
        
        # 预处理
        source_proc = self.preprocessor.preprocess_pipeline(source)
        target_proc = self.preprocessor.preprocess_pipeline(target)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Step 2: Preprocessing Process', fontsize=16, fontweight='bold')
        
        # 原始点云
        for i, (pcd, title, color) in enumerate([(source, "Original Source", "red"), 
                                                   (target, "Original Target", "green")]):
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # 3D投影
            axes[0, i*2].scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.6)
            axes[0, i*2].set_title(f'{title} - XY View')
            axes[0, i*2].set_xlabel('X')
            axes[0, i*2].set_ylabel('Y')
            axes[0, i*2].set_aspect('equal')
            axes[0, i*2].grid(True, alpha=0.3)
            
            axes[0, i*2+1].scatter(points[:, 0], points[:, 2], c=colors, s=1, alpha=0.6)
            axes[0, i*2+1].set_title(f'{title} - XZ View')
            axes[0, i*2+1].set_xlabel('X')
            axes[0, i*2+1].set_ylabel('Z')
            axes[0, i*2+1].set_aspect('equal')
            axes[0, i*2+1].grid(True, alpha=0.3)
        
        # 预处理后的点云
        for i, (pcd, title, color) in enumerate([(source_proc, "Preprocessed Source", "red"), 
                                                   (target_proc, "Preprocessed Target", "green")]):
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # 3D投影
            axes[1, i*2].scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.6)
            axes[1, i*2].set_title(f'{title} - XY View')
            axes[1, i*2].set_xlabel('X')
            axes[1, i*2].set_ylabel('Y')
            axes[1, i*2].set_aspect('equal')
            axes[1, i*2].grid(True, alpha=0.3)
            
            axes[1, i*2+1].scatter(points[:, 0], points[:, 2], c=colors, s=1, alpha=0.6)
            axes[1, i*2+1].set_title(f'{title} - XZ View')
            axes[1, i*2+1].set_xlabel('X')
            axes[1, i*2+1].set_ylabel('Z')
            axes[1, i*2+1].set_aspect('equal')
            axes[1, i*2+1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('step2_preprocessing.png', dpi=300, bbox_inches='tight')
        print("✓ 保存: step2_preprocessing.png")
        plt.show()
        
        # 统计信息
        print("预处理统计:")
        print(f"  源点云: {len(source.points)} → {len(source_proc.points)} 点")
        print(f"  目标点云: {len(target.points)} → {len(target_proc.points)} 点")
        
        return source_proc, target_proc
    
    def visualize_step3_feature_computation(self, source, target):
        """步骤3：特征计算可视化"""
        print("\n=== 步骤3：FPFH特征计算 ===")
        
        # 计算FPFH特征
        print("计算FPFH特征...")
        source_fpfh = self.feature_calculator.compute_fpfh_features(source)
        target_fpfh = self.feature_calculator.compute_fpfh_features(target)
        
        # 计算PFH特征用于对比
        source_pfh = self.feature_calculator.compute_pfh_features(source)
        
        # 创建特征分析图
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('Step 3: FPFH Feature Computation & Analysis', fontsize=16, fontweight='bold')
        
        # 1. 特征维度信息
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        feature_info = f"""
        FPFH Feature Information:
        
        Source:
        - Dimension: {source_fpfh.dimension()}
        - Num Points: {len(source.points)}
        - Features Shape: {np.asarray(source_fpfh.data).shape}
        
        Target:
        - Dimension: {target_fpfh.dimension()}
        - Num Points: {len(target.points)}
        - Features Shape: {np.asarray(target_fpfh.data).shape}
        
        FPFH vs PFH:
        - FPFH Dimension: 33
        - PFH Dimension: {source_pfh.dimension()}
        - Speed: FPFH ~5x faster
        - Accuracy: PFH slightly higher
        """
        ax1.text(0.1, 0.5, feature_info, fontsize=10, family='monospace',
                verticalalignment='center')
        
        # 2. FPFH特征分布（前8个维度）
        source_data = np.asarray(source_fpfh.data)
        for i in range(min(8, source_fpfh.dimension())):
            ax = fig.add_subplot(gs[0, i+1])
            ax.hist(source_data[:, i], bins=30, color='red', alpha=0.7, edgecolor='black')
            ax.set_title(f'FPFH Dim {i+1}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # 3. FPFH特征热图
        ax_heat = fig.add_subplot(gs[1, :2])
        n_points = min(100, len(source_data))
        n_features = min(16, source_fpfh.dimension())
        heat_data = source_data[:n_points, :n_features]
        heat_data = (heat_data - heat_data.min(axis=0)) / (heat_data.max(axis=0) - heat_data.min(axis=0) + 1e-8)
        
        im = ax_heat.imshow(heat_data.T, cmap='viridis', aspect='auto', interpolation='nearest')
        ax_heat.set_title('FPFH Feature Heatmap (Source)')
        ax_heat.set_xlabel('Point Index')
        ax_heat.set_ylabel('Feature Dimension')
        plt.colorbar(im, ax=ax_heat, label='Normalized Value')
        
        # 4. 特征统计对比
        ax_stats = fig.add_subplot(gs[1, 2:])
        feature_stats = self.analyzer.analyze_feature_quality(source_fpfh, source)
        
        stats_text = f"""
        Feature Quality Metrics:
        
        Dimension: {feature_stats['num_features']}
        Num Points: {feature_stats['num_points']}
        Sparsity: {feature_stats['feature_sparsity']:.4f}
        Entropy: {feature_stats['feature_entropy']:.4f}
        
        Mean Range: {np.mean(feature_stats['feature_range']):.4f}
        Max Range: {np.max(feature_stats['feature_range']):.4f}
        Min Range: {np.min(feature_stats['feature_range']):.4f}
        """
        ax_stats.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                     verticalalignment='center')
        ax_stats.axis('off')
        
        # 5. FPFH vs PFH对比（前6个维度）
        target_data = np.asarray(target_fpfh.data)
        source_pfh_data = np.asarray(source_pfh.data)
        
        for i in range(min(6, source_fpfh.dimension())):
            ax = fig.add_subplot(gs[2, i])
            
            # FPFH
            ax.hist(source_data[:, i], bins=20, alpha=0.5, color='red', label='FPFH')
            # PFH
            ax.hist(source_pfh_data[:, i], bins=20, alpha=0.5, color='blue', label='PFH')
            
            ax.set_title(f'Dim {i+1}: FPFH vs PFH')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('step3_feature_computation.png', dpi=300, bbox_inches='tight')
        print("✓ 保存: step3_feature_computation.png")
        plt.show()
        
        return source_fpfh, target_fpfh
    
    def visualize_step4_matching_process(self, source, target, source_fpfh, target_fpfh):
        """步骤4：特征匹配过程可视化"""
        print("\n=== 步骤4：特征匹配过程 ===")
        
        # 执行配准
        print("执行FPFH-RANSAC配准...")
        result = self.feature_calculator.execute_fpfh_ransac_registration(source, target)
        
        # 获取对应点
        correspondences = result['correspondence_set']
        transformation = result['transformation']
        
        print(f"匹配结果:")
        print(f"  - 对应点数量: {len(correspondences)}")
        print(f"  - Fitness: {result['fitness']:.4f}")
        print(f"  - RMSE: {result['inlier_rmse']:.4f}")
        
        # 创建匹配可视化
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('Step 4: Feature Matching Process', fontsize=16, fontweight='bold')
        
        # 1. 配准前叠加
        ax1 = fig.add_subplot(gs[0, 0])
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        source_colors = np.asarray(source.colors)
        target_colors = np.asarray(target.colors)
        
        ax1.scatter(source_points[:, 0], source_points[:, 1], c=source_colors, s=2, alpha=0.6, label='Source')
        ax1.scatter(target_points[:, 0], target_points[:, 1], c=target_colors, s=2, alpha=0.6, label='Target')
        ax1.set_title('Before Registration')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. 配准后叠加
        ax2 = fig.add_subplot(gs[0, 1])
        source_aligned = source.transform(transformation)
        aligned_points = np.asarray(source_aligned.points)
        
        ax2.scatter(aligned_points[:, 0], aligned_points[:, 1], c=source_colors, s=2, alpha=0.6, label='Aligned Source')
        ax2.scatter(target_points[:, 0], target_points[:, 1], c=target_colors, s=2, alpha=0.6, label='Target')
        ax2.set_title('After Registration')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # 3. 对应线显示（选择部分）
        ax3 = fig.add_subplot(gs[0, 2])
        
        # 显示前30个对应关系
        max_lines = min(30, len(correspondences))
        for i in range(max_lines):
            src_idx, tgt_idx = correspondences[i]
            if src_idx < len(aligned_points) and tgt_idx < len(target_points):
                # 绘制对应线
                ax3.plot([aligned_points[src_idx, 0], target_points[tgt_idx, 0]],
                        [aligned_points[src_idx, 1], target_points[tgt_idx, 1]],
                        'yellow', alpha=0.3, linewidth=0.5)
        
        # 绘制点云
        ax3.scatter(aligned_points[:, 0], aligned_points[:, 1], c=source_colors, s=5, alpha=0.8)
        ax3.scatter(target_points[:, 0], target_points[:, 1], c=target_colors, s=5, alpha=0.8)
        ax3.set_title(f'Correspondence Lines (Top {max_lines})')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # 4. 配准质量指标
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        
        quality_text = f"""
        Registration Quality Metrics:
        
        Method: FPFH-RANSAC
        Fitness: {result['fitness']:.4f}
        Inlier RMSE: {result['inlier_rmse']:.4f}
        Correspondences: {len(correspondences)}
        
        Transformation Matrix:
        [{transformation[0,0]:.3f} {transformation[0,1]:.3f} {transformation[0,2]:.3f} {transformation[0,3]:.3f}]
        [{transformation[1,0]:.3f} {transformation[1,1]:.3f} {transformation[1,2]:.3f} {transformation[1,3]:.3f}]
        [{transformation[2,0]:.3f} {transformation[2,1]:.3f} {transformation[2,2]:.3f} {transformation[2,3]:.3f}]
        [{transformation[3,0]:.3f} {transformation[3,1]:.3f} {transformation[3,2]:.3f} {transformation[3,3]:.3f}]
        """
        ax4.text(0.1, 0.5, quality_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        # 5. 对应点距离分布
        ax5 = fig.add_subplot(gs[1, 1])
        
        # 计算对应点距离
        distances = []
        for src_idx, tgt_idx in correspondences[:100]:  # 限制在100个点
            if src_idx < len(aligned_points) and tgt_idx < len(target_points):
                dist = np.linalg.norm(aligned_points[src_idx] - target_points[tgt_idx])
                distances.append(dist)
        
        if distances:
            ax5.hist(distances, bins=30, color='green', alpha=0.7, edgecolor='black')
            ax5.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.4f}')
            ax5.set_title('Correspondence Distance Distribution')
            ax5.set_xlabel('Distance (m)')
            ax5.set_ylabel('Frequency')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. 匹配成功率
        ax6 = fig.add_subplot(gs[1, 2])
        
        match_stats = self.analyzer.analyze_feature_matching_quality(
            source_fpfh, target_fpfh, correspondences)
        
        if 'num_correspondences' in match_stats:
            categories = ['Total\nMatches', 'Mean\nDistance', 'Median\nDistance', 'Good\nMatch\nRatio']
            values = [
                match_stats.get('num_correspondences', 0),
                match_stats.get('mean_feature_distance', 0),
                match_stats.get('median_feature_distance', 0),
                match_stats.get('good_match_ratio', 0)
            ]
            
            colors_bar = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
            bars = ax6.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax6.set_title('Matching Success Statistics')
            ax6.set_ylabel('Value')
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('step4_matching_process.png', dpi=300, bbox_inches='tight')
        print("✓ 保存: step4_matching_process.png")
        plt.show()
        
        return result, source_aligned
    
    def visualize_step5_comparison(self, source, target, source_aligned, result):
        """步骤5：不同方法对比可视化"""
        print("\n=== 步骤5：配准方法对比 ===")
        
        # 测试不同的配准方法
        methods = ['fpfh_ransac', 'pfh_ransac', 'hybrid_fpfh']
        results_comparison = {}
        
        for method in methods:
            print(f"测试方法: {method}")
            try:
                method_result = feature_based_registration(source, target, method=method)
                results_comparison[method] = method_result
                print(f"  ✓ Fitness: {method_result['fitness']:.4f}, RMSE: {method_result['inlier_rmse']:.4f}")
            except Exception as e:
                print(f"  ✗ 失败: {e}")
                results_comparison[method] = {'error': str(e)}
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Step 5: Registration Method Comparison', fontsize=16, fontweight='bold')
        
        # 提取成功的方法数据
        successful_methods = {k: v for k, v in results_comparison.items() 
                            if 'error' not in v}
        
        if successful_methods:
            methods_list = list(successful_methods.keys())
            fitness_scores = [successful_methods[m]['fitness'] for m in methods_list]
            rmse_scores = [successful_methods[m]['inlier_rmse'] for m in methods_list]
            num_matches = [len(successful_methods[m].get('correspondence_set', [])) for m in methods_list]
            
            # Fitness对比
            axes[0, 0].bar(methods_list, fitness_scores, color='blue', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Fitness Comparison')
            axes[0, 0].set_ylabel('Fitness Score')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].grid(True, alpha=0.3)
            plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=15, ha='right')
            
            # RMSE对比
            axes[0, 1].bar(methods_list, rmse_scores, color='green', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('RMSE Comparison')
            axes[0, 1].set_ylabel('RMSE (m)')
            axes[0, 1].grid(True, alpha=0.3)
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=15, ha='right')
            
            # 对应点数量对比
            axes[0, 2].bar(methods_list, num_matches, color='orange', alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('Number of Correspondences')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].grid(True, alpha=0.3)
            plt.setp(axes[0, 2].xaxis.get_majorticklabels(), rotation=15, ha='right')
            
            # 可视化配准结果对比
            target_points = np.asarray(target.points)
            target_colors = np.asarray(target.colors)
            
            colors = ['red', 'green', 'blue']
            titles = ['FPFH-RANSAC', 'PFH-RANSAC', 'Hybrid-FPFH']
            
            for i, method in enumerate(methods_list[:3]):
                if method in successful_methods:
                    aligned = source.transform(successful_methods[method]['transformation'])
                    aligned_points = np.asarray(aligned.points)
                    source_colors = np.asarray(source.colors)
                    
                    axes[1, i].scatter(aligned_points[:, 0], aligned_points[:, 1], 
                                      c=source_colors, s=3, alpha=0.6, label='Aligned')
                    axes[1, i].scatter(target_points[:, 0], target_points[:, 1], 
                                      c=target_colors, s=3, alpha=0.6, label='Target')
                    axes[1, i].set_title(f'{titles[i]} Result\nFitness: {successful_methods[method]["fitness"]:.3f}')
                    axes[1, i].set_xlabel('X')
                    axes[1, i].set_ylabel('Y')
                    axes[1, i].legend(fontsize=8)
                    axes[1, i].set_aspect('equal')
                    axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('step5_method_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ 保存: step5_method_comparison.png")
        plt.show()
        
        return results_comparison
    
    def visualize_step6_final_result(self, source_aligned, target):
        """步骤6：最终结果可视化"""
        print("\n=== 步骤6：最终融合结果 ===")
        
        # 融合点云
        from core.fusion import fuse_point_clouds
        
        fused = fuse_point_clouds([source_aligned, target], method='color_aware')
        
        # 创建最终结果展示
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('Step 6: Final Fusion Result', fontsize=16, fontweight='bold')
        
        aligned_points = np.asarray(source_aligned.points)
        target_points = np.asarray(target.points)
        fused_points = np.asarray(fused.points)
        aligned_colors = np.asarray(source_aligned.colors)
        target_colors = np.asarray(target.colors)
        fused_colors = np.asarray(fused.colors)
        
        # 1. 配准后的源点云
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(aligned_points[:, 0], aligned_points[:, 1], c=aligned_colors, s=2, alpha=0.6)
        ax1.set_title('Aligned Source Point Cloud')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. 目标点云
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(target_points[:, 0], target_points[:, 1], c=target_colors, s=2, alpha=0.6)
        ax2.set_title('Target Point Cloud')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # 3. 融合结果
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(fused_points[:, 0], fused_points[:, 1], c=fused_colors, s=2, alpha=0.6)
        ax3.set_title('Fused Point Cloud')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # 4. 点云统计信息
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        
        stats_text = f"""
        Point Cloud Statistics:
        
        Aligned Source:
        - Points: {len(aligned_points)}
        - X Range: [{aligned_points[:, 0].min():.2f}, {aligned_points[:, 0].max():.2f}]
        - Y Range: [{aligned_points[:, 1].min():.2f}, {aligned_points[:, 1].max():.2f}]
        - Z Range: [{aligned_points[:, 2].min():.2f}, {aligned_points[:, 2].max():.2f}]
        
        Target:
        - Points: {len(target_points)}
        - X Range: [{target_points[:, 0].min():.2f}, {target_points[:, 0].max():.2f}]
        - Y Range: [{target_points[:, 1].min():.2f}, {target_points[:, 1].max():.2f}]
        - Z Range: [{target_points[:, 2].min():.2f}, {target_points[:, 2].max():.2f}]
        
        Fused Result:
        - Points: {len(fused_points)}
        - Reduction: {len(aligned_points) + len(target_points) - len(fused_points)} points
        - Compression: {len(fused_points) / (len(aligned_points) + len(target_points)):.2%}
        """
        ax4.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        # 5. 颜色分布
        ax5 = fig.add_subplot(gs[1, 1])
        
        # 显示融合点云的颜色分布
        color_hist = np.zeros((3, 256))
        for i in range(3):  # R, G, B
            color_hist[i, :], _ = np.histogram(fused_colors[:, i], bins=256, range=(0, 1))
        
        ax5.plot(color_hist[0, :], color='red', alpha=0.7, label='Red Channel')
        ax5.plot(color_hist[1, :], color='green', alpha=0.7, label='Green Channel')
        ax5.plot(color_hist[2, :], color='blue', alpha=0.7, label='Blue Channel')
        ax5.set_title('Color Distribution in Fused Point Cloud')
        ax5.set_xlabel('Color Value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 3D效果（使用不同视角）
        ax6 = fig.add_subplot(gs[1, 2], projection='3d')
        
        # 选择部分点进行3D显示（避免过多点）
        sample_idx = np.random.choice(len(fused_points), min(1000, len(fused_points)), replace=False)
        sample_points = fused_points[sample_idx]
        sample_colors = fused_colors[sample_idx]
        
        ax6.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
                   c=sample_colors, s=1, alpha=0.6)
        ax6.set_title('3D View of Fused Point Cloud')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        
        # 设置视角
        ax6.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig('step6_final_result.png', dpi=300, bbox_inches='tight')
        print("✓ 保存: step6_final_result.png")
        plt.show()
        
        return fused
    
    def create_comprehensive_dashboard(self, source, target):
        """创建综合仪表板"""
        print("\n=== 创建综合特征配准仪表板 ===")
        
        # 执行完整流程
        source_proc, target_proc = self.visualize_step2_preprocessing(source, target)
        source_fpfh, target_fpfh = self.visualize_step3_feature_computation(source_proc, target_proc)
        result, source_aligned = self.visualize_step4_matching_process(source_proc, target_proc, source_fpfh, target_fpfh)
        comparison = self.visualize_step5_comparison(source_proc, target_proc, source_aligned, result)
        fused = self.visualize_step6_final_result(source_aligned, target_proc)
        
        print("\n=== 可视化完成 ===")
        print("生成的文件:")
        print("  - step1_original_clouds.png")
        print("  - step2_preprocessing.png")
        print("  - step3_feature_computation.png")
        print("  - step4_matching_process.png")
        print("  - step5_method_comparison.png")
        print("  - step6_final_result.png")
        
        return {
            'preprocessed': (source_proc, target_proc),
            'features': (source_fpfh, target_fpfh),
            'registration_result': result,
            'aligned_cloud': source_aligned,
            'method_comparison': comparison,
            'fused_cloud': fused
        }


def main():
    """主函数"""
    print("🎯 特征配准可视化演示")
    print("=" * 50)
    
    # 创建可视化演示器
    demo = FeatureRegistrationVisualizer()
    
    # 创建演示数据
    source, target = demo.create_demo_point_clouds()
    
    # 步骤1：原始点云可视化
    demo.visualize_step1_original_clouds(source, target)
    
    # 执行完整流程
    results = demo.create_comprehensive_dashboard(source, target)
    
    print("\n🎉 特征配准可视化演示完成！")
    print("\n💡 提示：您可以查看生成的PNG文件来了解整个特征配准过程")


if __name__ == '__main__':
    main()