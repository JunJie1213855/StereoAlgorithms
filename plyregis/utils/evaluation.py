"""
质量评估工具模块
提供点云配准和融合质量评估功能
"""
import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Optional
from scipy.spatial import cKDTree


class RegistrationEvaluator:
    """配准质量评估类"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate_registration_result(self, source: o3d.geometry.PointCloud,
                                    target: o3d.geometry.PointCloud,
                                    transformation: np.ndarray,
                                    threshold: float = 0.02) -> Dict:
        """
        评估配准结果
        
        Args:
            source: 源点云
            target: 目标点云
            transformation: 变换矩阵
            threshold: 距离阈值
            
        Returns:
            评估结果字典
        """
        # 应用变换
        source_transformed = source.transform(transformation)
        
        # 计算最近邻距离
        source_points = np.asarray(source_transformed.points)
        target_points = np.asarray(target.points)
        
        if len(source_points) == 0 or len(target_points) == 0:
            return {
                'error': 'Source or target point cloud is empty',
                'fitness': 0.0,
                'inlier_rmse': float('inf'),
                'overlap_ratio': 0.0
            }
        
        # 使用KD树进行最近邻搜索
        target_tree = cKDTree(target_points)
        distances, indices = target_tree.query(source_points, k=1)
        
        # 计算内点
        inliers = distances < threshold
        num_inliers = np.sum(inliers)
        
        # 计算评估指标
        fitness = num_inliers / len(source_points) if len(source_points) > 0 else 0.0
        inlier_rmse = np.sqrt(np.mean(distances[inliers]**2)) if num_inliers > 0 else float('inf')
        
        # 计算重叠比率
        source_tree = cKDTree(source_points)
        reverse_distances, _ = source_tree.query(target_points, k=1)
        overlap_inliers = reverse_distances < threshold
        overlap_ratio = (np.sum(overlap_inliers) + num_inliers) / (len(source_points) + len(target_points))
        
        return {
            'fitness': fitness,
            'inlier_rmse': inlier_rmse,
            'overlap_ratio': overlap_ratio,
            'num_inliers': num_inliers,
            'total_source_points': len(source_points),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'median_distance': np.median(distances),
            'max_distance': np.max(distances)
        }
    
    def compute_hausdorff_distance(self, pcd1: o3d.geometry.PointCloud,
                                  pcd2: o3d.geometry.PointCloud) -> Dict:
        """
        计算豪斯多夫距离
        
        Args:
            pcd1: 第一个点云
            pcd2: 第二个点云
            
        Returns:
            距离统计信息
        """
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        
        if len(points1) == 0 or len(points2) == 0:
            return {
                'hausdorff': float('inf'),
                'mean_hausdorff': float('inf'),
                'median_hausdorff': float('inf')
            }
        
        # 计算从pcd1到pcd2的距离
        tree1_to_2 = cKDTree(points2)
        distances_1_to_2, _ = tree1_to_2.query(points1, k=1)
        
        # 计算从pcd2到pcd1的距离
        tree2_to_1 = cKDTree(points1)
        distances_2_to_1, _ = tree2_to_1.query(points2, k=1)
        
        # 豪斯多夫距离
        hausdorff = max(np.max(distances_1_to_2), np.max(distances_2_to_1))
        mean_hausdorff = (np.mean(distances_1_to_2) + np.mean(distances_2_to_1)) / 2
        median_hausdorff = (np.median(distances_1_to_2) + np.median(distances_2_to_1)) / 2
        
        return {
            'hausdorff': hausdorff,
            'mean_hausdorff': mean_hausdorff,
            'median_hausdorff': median_hausdorff,
            'max_distance_1_to_2': np.max(distances_1_to_2),
            'max_distance_2_to_1': np.max(distances_2_to_1),
            'mean_distance_1_to_2': np.mean(distances_1_to_2),
            'mean_distance_2_to_1': np.mean(distances_2_to_1)
        }
    
    def compare_point_clouds(self, original: o3d.geometry.PointCloud,
                           processed: o3d.geometry.PointCloud) -> Dict:
        """
        比较原始点云和处理后的点云
        
        Args:
            original: 原始点云
            processed: 处理后的点云
            
        Returns:
            比较结果
        """
        original_points = np.asarray(original.points)
        processed_points = np.asarray(processed.points)
        
        result = {
            'original_points': len(original_points),
            'processed_points': len(processed_points),
            'reduction_ratio': (len(original_points) - len(processed_points)) / len(original_points) if len(original_points) > 0 else 0.0,
        }
        
        # 如果有颜色，比较颜色信息
        if original.has_colors() and processed.has_colors():
            original_colors = np.asarray(original.colors)
            processed_colors = np.asarray(processed.colors)
            
            color_diff = np.abs(original_colors[:len(processed_colors)] - processed_colors)
            result['color_change'] = {
                'mean': np.mean(color_diff),
                'std': np.std(color_diff),
                'max': np.max(color_diff)
            }
        
        return result


class FusionEvaluator:
    """融合质量评估类"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate_fusion_result(self, original_clouds: List[o3d.geometry.PointCloud],
                             fused_cloud: o3d.geometry.PointCloud) -> Dict:
        """
        评估融合结果
        
        Args:
            original_clouds: 原始点云列表
            fused_cloud: 融合后的点云
            
        Returns:
            评估结果
        """
        print("评估融合结果...")
        
        # 计算原始点云总点数
        total_original_points = sum(len(pcd.points) for pcd in original_clouds)
        fused_points = len(fused_cloud.points)
        
        # 计算压缩比
        compression_ratio = fused_points / total_original_points if total_original_points > 0 else 0.0
        
        # 计算融合后点云的统计信息
        fused_points_array = np.asarray(fused_cloud.points)
        point_density = len(fused_points_array) / self._compute_bounding_volume(fused_cloud)
        
        result = {
            'total_original_points': total_original_points,
            'fused_points': fused_points,
            'compression_ratio': compression_ratio,
            'reduction_percentage': (1 - compression_ratio) * 100,
            'point_density': point_density,
            'num_input_clouds': len(original_clouds)
        }
        
        # 评估颜色信息保留情况
        if all(pcd.has_colors() for pcd in original_clouds) and fused_cloud.has_colors():
            color_stats = self._evaluate_color_preservation(original_clouds, fused_cloud)
            result['color_preservation'] = color_stats
        
        return result
    
    def _compute_bounding_volume(self, pcd: o3d.geometry.PointCloud) -> float:
        """计算点云的边界体积"""
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = np.asarray(bbox.min_bound)
        max_bound = np.asarray(bbox.max_bound)
        return np.prod(max_bound - min_bound)
    
    def _evaluate_color_preservation(self, original_clouds: List[o3d.geometry.PointCloud],
                                   fused_cloud: o3d.geometry.PointCloud) -> Dict:
        """评估颜色信息保留情况"""
        # 合并原始点云的颜色
        all_original_colors = []
        for pcd in original_clouds:
            if pcd.has_colors():
                all_original_colors.extend(np.asarray(pcd.colors))
        
        original_colors = np.array(all_original_colors)
        fused_colors = np.asarray(fused_cloud.colors)
        
        if len(original_colors) == 0 or len(fused_colors) == 0:
            return {'error': 'No color information available'}
        
        # 计算颜色统计信息
        return {
            'original_color_mean': np.mean(original_colors, axis=0).tolist(),
            'fused_color_mean': np.mean(fused_colors, axis=0).tolist(),
            'original_color_std': np.std(original_colors, axis=0).tolist(),
            'fused_color_std': np.std(fused_colors, axis=0).tolist()
        }
    
    def detect_registration_artifacts(self, point_clouds: List[o3d.geometry.PointCloud],
                                    threshold: float = 0.01) -> Dict:
        """
        检测配准伪影
        
        Args:
            point_clouds: 点云列表
            threshold: 检测阈值
            
        Returns:
            伪影检测结果
        """
        print("检测配准伪影...")
        
        artifacts = {
            'has_artifacts': False,
            'detected_issues': []
        }
        
        # 检查每个点云的质量
        for i, pcd in enumerate(point_clouds):
            points = np.asarray(pcd.points)
            
            # 检查异常点
            if len(points) > 0:
                # 计算点到原点的距离
                distances = np.linalg.norm(points, axis=1)
                
                # 检查是否有异常远的点
                median_dist = np.median(distances)
                outliers = distances > (median_dist * 10)
                
                if np.sum(outliers) > len(points) * 0.01:  # 超过1%的点异常
                    artifacts['has_artifacts'] = True
                    artifacts['detected_issues'].append(
                        f'PointCloud_{i}: 包含 {np.sum(outliers)} 个异常远点'
                    )
        
        return artifacts
    
    def compute_overlap_matrices(self, point_clouds: List[o3d.geometry.PointCloud],
                               threshold: float = 0.02) -> np.ndarray:
        """
        计算点云间的重叠矩阵
        
        Args:
            point_clouds: 点云列表
            threshold: 重叠判定阈值
            
        Returns:
            重叠矩阵
        """
        num_clouds = len(point_clouds)
        overlap_matrix = np.zeros((num_clouds, num_clouds))
        
        for i in range(num_clouds):
            for j in range(i, num_clouds):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    overlap = self._compute_pairwise_overlap(
                        point_clouds[i], point_clouds[j], threshold)
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap
        
        return overlap_matrix
    
    def _compute_pairwise_overlap(self, pcd1: o3d.geometry.PointCloud,
                                pcd2: o3d.geometry.PointCloud,
                                threshold: float) -> float:
        """计算两个点云间的重叠率"""
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        # 计算从pcd1到pcd2的重叠
        tree1_to_2 = cKDTree(points2)
        distances_1_to_2, _ = tree1_to_2.query(points1, k=1)
        overlap_1_to_2 = np.sum(distances_1_to_2 < threshold) / len(points1)
        
        # 计算从pcd2到pcd1的重叠
        tree2_to_1 = cKDTree(points1)
        distances_2_to_1, _ = tree2_to_1.query(points2, k=1)
        overlap_2_to_1 = np.sum(distances_2_to_1 < threshold) / len(points2)
        
        # 平均重叠率
        return (overlap_1_to_2 + overlap_2_to_1) / 2


def generate_evaluation_report(registration_results: List[Dict],
                              fusion_results: Dict,
                              save_path: Optional[str] = None) -> str:
    """
    生成评估报告
    
    Args:
        registration_results: 配准结果列表
        fusion_results: 融合结果
        save_path: 保存路径
        
    Returns:
        报告文本
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("点云配准与融合评估报告")
    report_lines.append("=" * 60)
    
    # 配准结果摘要
    report_lines.append("\n## 配准结果摘要")
    for i, result in enumerate(registration_results):
        report_lines.append(f"\n### 配准对 {i+1}")
        report_lines.append(f"  - Fitness: {result.get('fitness', 0):.4f}")
        report_lines.append(f"  - RMSE: {result.get('inlier_rmse', 0):.4f}")
        report_lines.append(f"  - 重叠率: {result.get('overlap_ratio', 0):.4f}")
    
    # 融合结果摘要
    report_lines.append("\n## 融合结果摘要")
    report_lines.append(f"  - 原始点云总数: {fusion_results.get('total_original_points', 0)}")
    report_lines.append(f"  - 融合后点数: {fusion_results.get('fused_points', 0)}")
    report_lines.append(f"  - 压缩比: {fusion_results.get('compression_ratio', 0):.4f}")
    report_lines.append(f"  - 减少百分比: {fusion_results.get('reduction_percentage', 0):.2f}%")
    
    # 颜色保留信息
    if 'color_preservation' in fusion_results:
        report_lines.append("\n## 颜色信息保留")
        color_info = fusion_results['color_preservation']
        report_lines.append(f"  - 原始颜色均值: {color_info.get('original_color_mean', 'N/A')}")
        report_lines.append(f"  - 融合颜色均值: {color_info.get('fused_color_mean', 'N/A')}")
    
    report_lines.append("\n" + "=" * 60)
    
    report_text = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"报告已保存到: {save_path}")
    
    return report_text