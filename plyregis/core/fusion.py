"""
点云融合模块
提供多点点云融合和重建功能
"""
import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional
import copy


class PointCloudFusion:
    """点云融合类"""
    
    def __init__(self, voxel_size: float = 0.01):
        """
        初始化融合器
        
        Args:
            voxel_size: 融合时使用的体素大小
        """
        self.voxel_size = voxel_size
    
    def simple_merge(self, point_clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """
        简单合并多个点云
        
        Args:
            point_clouds: 点云列表
            
        Returns:
            合并后的点云
        """
        if not point_clouds:
            raise ValueError("点云列表不能为空")
        
        print(f"合并 {len(point_clouds)} 个点云...")
        
        merged = o3d.geometry.PointCloud()
        
        for i, pcd in enumerate(point_clouds):
            print(f"添加点云 {i+1}/{len(point_clouds)}: {len(pcd.points)} 点")
            merged += pcd
        
        print(f"合并完成，总点数: {len(merged.points)}")
        return merged
    
    def voxel_based_fusion(self, point_clouds: List[o3d.geometry.PointCloud],
                          voxel_size: Optional[float] = None) -> o3d.geometry.PointCloud:
        """
        基于体素的点云融合
        
        Args:
            point_clouds: 点云列表
            voxel_size: 体素大小
            
        Returns:
            融合后的点云
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        print(f"执行基于体素的融合，体素大小: {voxel_size}")
        
        # 首先合并所有点云
        merged = self.simple_merge(point_clouds)
        
        # 使用体素下采样进行融合
        # 每个体素内的点会被合并为单个点，颜色取平均
        fused = merged.voxel_down_sample(voxel_size)
        
        print(f"融合完成，原始点数: {len(merged.points)}, 融合后点数: {len(fused.points)}")
        return fused
    
    def statistical_fusion(self, point_clouds: List[o3d.geometry.PointCloud],
                          voxel_size: Optional[float] = None) -> o3d.geometry.PointCloud:
        """
        统计融合方法
        
        Args:
            point_clouds: 点云列表
            voxel_size: 体素大小
            
        Returns:
            融合后的点云
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        print(f"执行统计融合，体素大小: {voxel_size}")
        
        # 合并点云
        merged = self.simple_merge(point_clouds)
        
        # 计算点云的统计信息
        points = np.asarray(merged.points)
        colors = np.asarray(merged.colors) if merged.has_colors() else None
        
        if colors is not None:
            # 创建新的点云，每个体素保留颜色方差最小的点
            voxel_indices = np.floor(points / voxel_size).astype(np.int32)
            
            # 计算每个体素的唯一键
            voxel_keys = voxel_indices[:, 0] * 73856093 + \
                        voxel_indices[:, 1] * 19349663 + \
                        voxel_indices[:, 2] * 83492791
            
            unique_keys, inverse_indices = np.unique(voxel_keys, return_inverse=True)
            
            fused_points = []
            fused_colors = []
            
            for i in range(len(unique_keys)):
                mask = inverse_indices == i
                voxel_points = points[mask]
                voxel_colors = colors[mask]
                
                if len(voxel_points) == 1:
                    # 只有一个点，直接使用
                    fused_points.append(voxel_points[0])
                    fused_colors.append(voxel_colors[0])
                else:
                    # 多个点，选择颜色方差最小的点
                    color_variances = np.var(voxel_colors, axis=1)
                    best_idx = np.argmin(color_variances)
                    fused_points.append(voxel_points[best_idx])
                    fused_colors.append(voxel_colors[best_idx])
            
            # 创建融合后的点云
            fused = o3d.geometry.PointCloud()
            fused.points = o3d.utility.Vector3dVector(np.array(fused_points))
            fused.colors = o3d.utility.Vector3dVector(np.array(fused_colors))
            
            print(f"统计融合完成，原始点数: {len(points)}, 融合后点数: {len(fused.points)}")
            return fused
        else:
            # 没有颜色信息，使用简单的体素融合
            return self.voxel_based_fusion(point_clouds, voxel_size)
    
    def moving_least_squares_fusion(self, point_clouds: List[o3d.geometry.PointCloud],
                                   search_radius: float = 0.1,
                                   polynomial_degree: int = 2) -> o3d.geometry.PointCloud:
        """
        移动最小二乘法融合
        
        Args:
            point_clouds: 点云列表
            search_radius: 搜索半径
            polynomial_degree: 多项式次数
            
        Returns:
            融合后的点云
        """
        print(f"执行移动最小二乘法融合，搜索半径: {search_radius}")
        
        # 首先进行简单合并
        merged = self.simple_merge(point_clouds)
        
        # 计算法向量（如果不存在）
        if not merged.has_normals():
            merged.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius * 2, max_nn=30)
            )
        
        # 执行移动最小二乘法平滑
        fused = merged.filter_smooth_mls(
            search_radius=search_radius,
            polynomial_degree=polynomial_degree
        )
        
        print(f"MLS融合完成，原始点数: {len(merged.points)}, 融合后点数: {len(fused.points)}")
        return fused
    
    def color_aware_fusion(self, point_clouds: List[o3d.geometry.PointCloud],
                          voxel_size: Optional[float] = None,
                          color_weight: float = 0.5) -> o3d.geometry.PointCloud:
        """
        颜色感知融合
        
        Args:
            point_clouds: 点云列表
            voxel_size: 体素大小
            color_weight: 颜色权重 (0-1)
            
        Returns:
            融合后的点云
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        print(f"执行颜色感知融合，体素大小: {voxel_size}, 颜色权重: {color_weight}")
        
        # 检查所有点云是否都有颜色
        has_colors = all(pcd.has_colors() for pcd in point_clouds)
        
        if not has_colors:
            print("警告：部分点云缺少颜色信息，使用普通体素融合")
            return self.voxel_based_fusion(point_clouds, voxel_size)
        
        # 合并点云
        merged = self.simple_merge(point_clouds)
        
        points = np.asarray(merged.points)
        colors = np.asarray(merged.colors)
        
        # 创建体素网格
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # 为每个体素计算代表性的颜色和位置
        voxel_dict = {}
        
        for i in range(len(points)):
            voxel_key = tuple(voxel_indices[i])
            
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = {
                    'points': [],
                    'colors': []
                }
            
            voxel_dict[voxel_key]['points'].append(points[i])
            voxel_dict[voxel_key]['colors'].append(colors[i])
        
        # 计算每个体素的代表点
        fused_points = []
        fused_colors = []
        
        for voxel_data in voxel_dict.values():
            voxel_points = np.array(voxel_data['points'])
            voxel_colors = np.array(voxel_data['colors'])
            
            # 计算平均位置
            avg_point = np.mean(voxel_points, axis=0)
            
            # 计算加权平均颜色
            avg_color = np.mean(voxel_colors, axis=0)
            
            fused_points.append(avg_point)
            fused_colors.append(avg_color)
        
        # 创建融合后的点云
        fused = o3d.geometry.PointCloud()
        fused.points = o3d.utility.Vector3dVector(np.array(fused_points))
        fused.colors = o3d.utility.Vector3dVector(np.array(fused_colors))
        
        print(f"颜色感知融合完成，原始点数: {len(points)}, 融合后点数: {len(fused.points)}")
        return fused
    
    def poisson_surface_reconstruction(self, point_clouds: List[o3d.geometry.PointCloud],
                                      depth: int = 8) -> o3d.geometry.TriangleMesh:
        """
        泊松表面重建
        
        Args:
            point_clouds: 点云列表
            depth: 重建深度
            
        Returns:
            重建的网格
        """
        print("执行泊松表面重建...")
        
        # 融合点云
        fused = self.voxel_based_fusion(point_clouds)
        
        # 确保法向量存在
        if not fused.has_normals():
            fused.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
            )
        
        # 执行泊松重建
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            fused, depth=depth)
        
        print(f"泊松重建完成，网格顶点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")
        return mesh
    
    def ball_pivoting_reconstruction(self, point_clouds: List[o3d.geometry.PointCloud],
                                    radii: Optional[List[float]] = None) -> o3d.geometry.TriangleMesh:
        """
        滚球重建
        
        Args:
            point_clouds: 点云列表
            radii: 滚球半径列表
            
        Returns:
            重建的网格
        """
        print("执行滚球重建...")
        
        # 融合点云
        fused = self.voxel_based_fusion(point_clouds)
        
        # 确保法向量存在
        if not fused.has_normals():
            fused.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
            )
        
        # 设置默认滚球半径
        if radii is None:
            radii = [self.voxel_size, self.voxel_size * 2, self.voxel_size * 4]
        
        # 执行滚球重建
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            fused, o3d.utility.DoubleVector(radii))
        
        print(f"滚球重建完成，网格顶点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")
        return mesh
    
    def remove_duplicate_points(self, point_cloud: o3d.geometry.PointCloud,
                               voxel_size: Optional[float] = None) -> o3d.geometry.PointCloud:
        """
        移除重复点
        
        Args:
            point_cloud: 输入点云
            voxel_size: 体素大小
            
        Returns:
            移除重复点后的点云
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        print(f"移除重复点，体素大小: {voxel_size}")
        
        # 使用体素下采样来移除重复点
        deduplicated = point_cloud.voxel_down_sample(voxel_size)
        
        print(f"原始点数: {len(point_cloud.points)}, 去重后点数: {len(deduplicated.points)}")
        return deduplicated
    
    def smart_fusion(self, point_clouds: List[o3d.geometry.PointCloud],
                    method: str = 'color_aware',
                    voxel_size: Optional[float] = None) -> o3d.geometry.PointCloud:
        """
        智能融合方法选择
        
        Args:
            point_clouds: 点云列表
            method: 融合方法 ('simple', 'voxel', 'statistical', 'color_aware', 'mls')
            voxel_size: 体素大小
            
        Returns:
            融合后的点云
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        print(f"=== 开始智能融合，方法: {method} ===")
        
        if method == 'simple':
            return self.simple_merge(point_clouds)
        elif method == 'voxel':
            return self.voxel_based_fusion(point_clouds, voxel_size)
        elif method == 'statistical':
            return self.statistical_fusion(point_clouds, voxel_size)
        elif method == 'color_aware':
            return self.color_aware_fusion(point_clouds, voxel_size)
        elif method == 'mls':
            return self.moving_least_squares_fusion(point_clouds, voxel_size)
        else:
            raise ValueError(f"未知的融合方法: {method}")


def fuse_point_clouds(point_clouds: List[o3d.geometry.PointCloud],
                     method: str = 'color_aware',
                     voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    融合点云的便捷函数
    
    Args:
        point_clouds: 点云列表
        method: 融合方法
        voxel_size: 体素大小
        
    Returns:
        融合后的点云
    """
    fusion = PointCloudFusion(voxel_size=voxel_size)
    return fusion.smart_fusion(point_clouds, method=method)