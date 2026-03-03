"""
特征可视化工具模块
提供PFH和FPFH特征的可视化功能
"""
import numpy as np
import open3d as o3d
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import copy


class FeatureVisualizer:
    """特征可视化类"""
    
    def __init__(self):
        """初始化特征可视化器"""
        self.figures = []
    
    def visualize_feature_histograms(self, features: o3d.pipelines.registration.Feature,
                                    title: str = "Feature Distribution",
                                    save_path: Optional[str] = None) -> None:
        """
        可视化特征分布直方图
        
        Args:
            features: 特征数据
            title: 图表标题
            save_path: 保存路径
        """
        feature_data = np.asarray(features.data)
        
        # 创建子图
        n_features = min(features.dimension(), 16)  # 最多显示16个特征
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        fig.suptitle(title, fontsize=16)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_features):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # 绘制直方图
            ax.hist(feature_data[:, i], bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f'Feature {i+1}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征直方图已保存: {save_path}")
        
        plt.show()
        self.figures.append(fig)
    
    def compare_feature_distributions(self, features1: o3d.pipelines.registration.Feature,
                                    features2: o3d.pipelines.registration.Feature,
                                    label1: str = "Features 1",
                                    label2: str = "Features 2",
                                    title: str = "Feature Distribution Comparison",
                                    save_path: Optional[str] = None) -> None:
        """
        比较两组特征的分布
        
        Args:
            features1: 第一组特征
            features2: 第二组特征
            label1: 第一组标签
            label2: 第二组标签
            title: 图表标题
            save_path: 保存路径
        """
        feature_data1 = np.asarray(features1.data)
        feature_data2 = np.asarray(features2.data)
        
        # 比较相同维度的特征
        min_dim = min(features1.dimension(), features2.dimension())
        n_features = min(min_dim, 12)  # 最多显示12个特征
        
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        fig.suptitle(title, fontsize=16)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_features):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # 绘制叠加直方图
            ax.hist(feature_data1[:, i], bins=50, alpha=0.5, 
                   label=label1, color='blue')
            ax.hist(feature_data2[:, i], bins=50, alpha=0.5, 
                   label=label2, color='red')
            ax.set_title(f'Feature {i+1}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征比较图已保存: {save_path}")
        
        plt.show()
        self.figures.append(fig)
    
    def visualize_feature_matching(self, source: o3d.geometry.PointCloud,
                                  target: o3d.geometry.PointCloud,
                                  correspondences: List[Tuple[int, int]],
                                  transformation: np.ndarray,
                                  max_lines: int = 50,
                                  window_name: str = "Feature Matching") -> None:
        """
        可视化特征匹配结果
        
        Args:
            source: 源点云
            target: 目标点云
            correspondences: 对应点对
            transformation: 变换矩阵
            max_lines: 最大显示线条数
            window_name: 窗口名称
        """
        print(f"可视化特征匹配 (显示前 {min(max_lines, len(correspondences))} 条)")
        
        # 创建点云副本
        source_vis = copy.deepcopy(source)
        target_vis = copy.deepcopy(target)
        
        # 设置颜色
        if not source_vis.has_colors():
            source_vis.paint_uniform_color([1, 0, 0])  # 红色
        if not target_vis.has_colors():
            target_vis.paint_uniform_color([0, 1, 0])  # 绿色
        
        # 应用变换
        source_transformed = source_vis.transform(transformation)
        
        # 创建对应线
        source_points = np.asarray(source_transformed.points)
        target_points = np.asarray(target_vis.points)
        
        line_points = []
        line_indices = []
        
        for i, (src_idx, tgt_idx) in enumerate(correspondences[:max_lines]):
            if src_idx < len(source_points) and tgt_idx < len(target_points):
                line_points.append(source_points[src_idx])
                line_points.append(target_points[tgt_idx])
                line_indices.append([2*i, 2*i+1])
        
        if len(line_points) > 0:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
            line_set.lines = o3d.utility.Vector2iVector(np.array(line_indices))
            
            # 设置线条颜色（黄色）
            line_colors = np.tile([1, 1, 0], (len(line_indices), 1))
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            
            o3d.visualization.draw_geometries(
                [source_transformed, target_vis, line_set],
                window_name=window_name
            )
        else:
            print("没有有效的对应点可以显示")
            o3d.visualization.draw_geometries(
                [source_transformed, target_vis],
                window_name=window_name
            )
    
    def visualize_feature_statistics(self, features: o3d.pipelines.registration.Feature,
                                   feature_name: str = "Features",
                                   save_path: Optional[str] = None) -> None:
        """
        可视化特征统计信息
        
        Args:
            features: 特征数据
            feature_name: 特征名称
            save_path: 保存路径
        """
        feature_data = np.asarray(features.data)
        
        # 计算统计信息
        mean = np.mean(feature_data, axis=0)
        std = np.std(feature_data, axis=0)
        min_val = np.min(feature_data, axis=0)
        max_val = np.max(feature_data, axis=0)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{feature_name} Statistics', fontsize=16)
        
        # 均值图
        axes[0, 0].bar(range(len(mean)), mean, color='blue', alpha=0.7)
        axes[0, 0].set_title('Feature Mean')
        axes[0, 0].set_xlabel('Feature Index')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 标准差图
        axes[0, 1].bar(range(len(std)), std, color='green', alpha=0.7)
        axes[0, 1].set_title('Feature Standard Deviation')
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Std Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 范围图
        axes[1, 0].bar(range(len(min_val)), max_val - min_val, color='orange', alpha=0.7)
        axes[1, 0].set_title('Feature Range (Max - Min)')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Range')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 箱线图
        axes[1, 1].boxplot(feature_data[:, :min(20, features.dimension())])
        axes[1, 1].set_title('Feature Distribution (First 20)')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征统计图已保存: {save_path}")
        
        plt.show()
        self.figures.append(fig)
    
    def visualize_feature_heatmap(self, features: o3d.pipelines.registration.Feature,
                                 title: str = "Feature Heatmap",
                                 save_path: Optional[str] = None) -> None:
        """
        可视化特征热图
        
        Args:
            features: 特征数据
            title: 图表标题
            save_path: 保存路径
        """
        feature_data = np.asarray(features.data)
        
        # 只显示前100个点和前20个特征
        n_points = min(100, len(feature_data))
        n_features = min(20, features.dimension())
        
        display_data = feature_data[:n_points, :n_features]
        
        # 归一化到[0,1]范围
        display_data = (display_data - display_data.min(axis=0)) / \
                      (display_data.max(axis=0) - display_data.min(axis=0) + 1e-8)
        
        # 创建热图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(display_data.T, cmap='viridis', aspect='auto', interpolation='nearest')
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Point Index', fontsize=12)
        ax.set_ylabel('Feature Index', fontsize=12)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Feature Value', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征热图已保存: {save_path}")
        
        plt.show()
        self.figures.append(fig)
    
    def visualize_registration_comparison(self, 
                                        results: Dict[str, Dict],
                                        save_path: Optional[str] = None) -> None:
        """
        可视化不同配准方法的比较结果
        
        Args:
            results: 配准结果字典
            save_path: 保存路径
        """
        # 提取成功的方法
        methods = []
        fitness_scores = []
        rmse_scores = []
        execution_times = []
        
        for method, result in results.items():
            if isinstance(result, dict) and result.get('success', False):
                methods.append(method)
                fitness_scores.append(result.get('fitness', 0))
                rmse_scores.append(result.get('inlier_rmse', 0))
                execution_times.append(result.get('execution_time', 0))
        
        if not methods:
            print("没有成功的配准方法可以比较")
            return
        
        # 创建比较图表
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Registration Method Comparison', fontsize=16)
        
        # Fitness比较
        axes[0].bar(methods, fitness_scores, color='blue', alpha=0.7)
        axes[0].set_title('Fitness Comparison')
        axes[0].set_ylabel('Fitness Score')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # RMSE比较
        axes[1].bar(methods, rmse_scores, color='green', alpha=0.7)
        axes[1].set_title('RMSE Comparison')
        axes[1].set_ylabel('RMSE (m)')
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 执行时间比较
        axes[2].bar(methods, execution_times, color='orange', alpha=0.7)
        axes[2].set_title('Execution Time Comparison')
        axes[2].set_ylabel('Time (s)')
        axes[2].grid(True, alpha=0.3)
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"配准方法比较图已保存: {save_path}")
        
        plt.show()
        self.figures.append(fig)
    
    def create_feature_dashboard(self, 
                                source_features: o3d.pipelines.registration.Feature,
                                target_features: o3d.pipelines.registration.Feature,
                                source_pcd: o3d.geometry.PointCloud,
                                target_pcd: o3d.geometry.PointCloud,
                                save_path: Optional[str] = None) -> None:
        """
        创建特征分析仪表板
        
        Args:
            source_features: 源特征
            target_features: 目标特征
            source_pcd: 源点云
            target_pcd: 目标点云
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Feature Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. 源特征分布
        ax1 = fig.add_subplot(gs[0, 0])
        source_data = np.asarray(source_features.data)
        for i in range(min(5, source_features.dimension())):
            ax1.hist(source_data[:, i], bins=30, alpha=0.5, label=f'F{i+1}')
        ax1.set_title('Source Feature Distributions')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 目标特征分布
        ax2 = fig.add_subplot(gs[0, 1])
        target_data = np.asarray(target_features.data)
        for i in range(min(5, target_features.dimension())):
            ax2.hist(target_data[:, i], bins=30, alpha=0.5, label=f'F{i+1}')
        ax2.set_title('Target Feature Distributions')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 特征统计对比
        ax3 = fig.add_subplot(gs[0, 2])
        source_mean = np.mean(source_data, axis=0)
        target_mean = np.mean(target_data, axis=0)
        min_dim = min(len(source_mean), len(target_mean))
        
        x = np.arange(min_dim)
        width = 0.35
        ax3.bar(x - width/2, source_mean[:min_dim], width, label='Source', alpha=0.7)
        ax3.bar(x + width/2, target_mean[:min_dim], width, label='Target', alpha=0.7)
        ax3.set_title('Feature Mean Comparison')
        ax3.set_xlabel('Feature Index')
        ax3.set_ylabel('Mean Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 源特征热图
        ax4 = fig.add_subplot(gs[1, 0])
        n_points = min(50, len(source_data))
        n_features = min(15, source_features.dimension())
        source_heat = source_data[:n_points, :n_features]
        source_heat = (source_heat - source_heat.min(axis=0)) / \
                      (source_heat.max(axis=0) - source_heat.min(axis=0) + 1e-8)
        im1 = ax4.imshow(source_heat.T, cmap='viridis', aspect='auto')
        ax4.set_title('Source Feature Heatmap')
        ax4.set_xlabel('Point Index')
        ax4.set_ylabel('Feature Index')
        plt.colorbar(im1, ax=ax4)
        
        # 5. 目标特征热图
        ax5 = fig.add_subplot(gs[1, 1])
        n_points = min(50, len(target_data))
        n_features = min(15, target_features.dimension())
        target_heat = target_data[:n_points, :n_features]
        target_heat = (target_heat - target_heat.min(axis=0)) / \
                      (target_heat.max(axis=0) - target_heat.min(axis=0) + 1e-8)
        im2 = ax5.imshow(target_heat.T, cmap='viridis', aspect='auto')
        ax5.set_title('Target Feature Heatmap')
        ax5.set_xlabel('Point Index')
        ax5.set_ylabel('Feature Index')
        plt.colorbar(im2, ax=ax5)
        
        # 6. 特征范围对比
        ax6 = fig.add_subplot(gs[1, 2])
        source_range = np.max(source_data, axis=0) - np.min(source_data, axis=0)
        target_range = np.max(target_data, axis=0) - np.min(target_data, axis=0)
        min_dim = min(len(source_range), len(target_range))
        
        x = np.arange(min_dim)
        width = 0.35
        ax6.bar(x - width/2, source_range[:min_dim], width, label='Source', alpha=0.7)
        ax6.bar(x + width/2, target_range[:min_dim], width, label='Target', alpha=0.7)
        ax6.set_title('Feature Range Comparison')
        ax6.set_xlabel('Feature Index')
        ax6.set_ylabel('Range')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. 点云信息
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.axis('off')
        info_text = f"""
        Point Cloud Information:
        
        Source:
        - Points: {len(source_pcd.points)}
        - Has Colors: {source_pcd.has_colors()}
        - Has Normals: {source_pcd.has_normals()}
        - Feature Dim: {source_features.dimension()}
        
        Target:
        - Points: {len(target_pcd.points)}
        - Has Colors: {target_pcd.has_colors()}
        - Has Normals: {target_pcd.has_normals()}
        - Feature Dim: {target_features.dimension()}
        """
        ax7.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
                verticalalignment='center')
        
        # 8. 特征质量指标
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        
        # 计算质量指标
        source_entropy = -np.sum(np.histogram(source_data.flatten(), bins=50)[0] * 
                               np.log(np.histogram(source_data.flatten(), bins=50)[0] + 1e-8))
        target_entropy = -np.sum(np.histogram(target_data.flatten(), bins=50)[0] * 
                               np.log(np.histogram(target_data.flatten(), bins=50)[0] + 1e-8))
        
        quality_text = f"""
        Feature Quality Metrics:
        
        Source Features:
        - Mean: {np.mean(source_data):.4f}
        - Std: {np.std(source_data):.4f}
        - Min: {np.min(source_data):.4f}
        - Max: {np.max(source_data):.4f}
        - Entropy: {source_entropy:.2f}
        
        Target Features:
        - Mean: {np.mean(target_data):.4f}
        - Std: {np.std(target_data):.4f}
        - Min: {np.min(target_data):.4f}
        - Max: {np.max(target_data):.4f}
        - Entropy: {target_entropy:.2f}
        """
        ax8.text(0.1, 0.5, quality_text, fontsize=12, family='monospace',
                verticalalignment='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征分析仪表板已保存: {save_path}")
        
        plt.show()
        self.figures.append(fig)
    
    def close_all_figures(self):
        """关闭所有图表"""
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()


def quick_feature_visualization(features: o3d.pipelines.registration.Feature,
                               title: str = "Feature Visualization") -> None:
    """
    快速特征可视化
    
    Args:
        features: 特征数据
        title: 标题
    """
    visualizer = FeatureVisualizer()
    visualizer.visualize_feature_statistics(features, title)
    visualizer.visualize_feature_heatmap(features, title)