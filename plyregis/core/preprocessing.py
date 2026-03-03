"""
点云预处理模块
提供点云加载、滤波、下采样等预处理功能
"""
import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional
import warnings


class PointCloudPreprocessor:
    """点云预处理类"""
    
    def __init__(self, voxel_size: float = 0.02):
        """
        初始化预处理器
        
        Args:
            voxel_size: 体素大小，用于下采样
        """
        self.voxel_size = voxel_size
    
    def load_point_clouds(self, file_paths: List[str]) -> List[o3d.geometry.PointCloud]:
        """
        加载多个点云文件
        
        Args:
            file_paths: 点云文件路径列表
            
        Returns:
            加载的点云对象列表
        """
        point_clouds = []
        for path in file_paths:
            try:
                pcd = o3d.io.read_point_cloud(path)
                if len(pcd.points) == 0:
                    warnings.warn(f"点云文件 {path} 为空或无法读取")
                    continue
                point_clouds.append(pcd)
                print(f"成功加载点云: {path}, 点数: {len(pcd.points)}")
            except Exception as e:
                print(f"加载点云文件 {path} 时出错: {e}")
        
        return point_clouds
    
    def voxel_downsample(self, pcd: o3d.geometry.PointCloud, 
                        voxel_size: Optional[float] = None) -> o3d.geometry.PointCloud:
        """
        体素下采样
        
        Args:
            pcd: 输入点云
            voxel_size: 体素大小，如果为None则使用默认值
            
        Returns:
            下采样后的点云
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        downsampled = pcd.voxel_down_sample(voxel_size)
        print(f"体素下采样: {len(pcd.points)} -> {len(downsampled.points)} 点")
        return downsampled
    
    def remove_statistical_outliers(self, pcd: o3d.geometry.PointCloud,
                                   nb_neighbors: int = 20,
                                   std_ratio: float = 2.0) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        统计离群点滤波
        
        Args:
            pcd: 输入点云
            nb_neighbors: 邻居点数量
            std_ratio: 标准差比率
            
        Returns:
            滤波后的点云和内点索引
        """
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, 
                                               std_ratio=std_ratio)
        print(f"统计滤波: {len(pcd.points)} -> {len(cl.points)} 点 (移除 {len(pcd.points) - len(cl.points)} 个离群点)")
        return cl, ind
    
    def remove_radius_outliers(self, pcd: o3d.geometry.PointCloud,
                              nb_points: int = 16,
                              radius: float = 0.05) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        半径离群点滤波
        
        Args:
            pcd: 输入点云
            nb_points: 最小邻居点数
            radius: 搜索半径
            
        Returns:
            滤波后的点云和内点索引
        """
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        print(f"半径滤波: {len(pcd.points)} -> {len(cl.points)} 点 (移除 {len(pcd.points) - len(cl.points)} 个离群点)")
        return cl, ind
    
    def estimate_normals(self, pcd: o3d.geometry.PointCloud,
                        search_param: Optional[o3d.geometry.KDTreeSearchParam] = None) -> o3d.geometry.PointCloud:
        """
        估计点云法向量
        
        Args:
            pcd: 输入点云
            search_param: 搜索参数
            
        Returns:
            带法向量的点云
        """
        if search_param is None:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30)
        
        pcd.estimate_normals(search_param=search_param)
        
        # 使法向量方向一致
        pcd.orient_normals_consistent_tangent_plane(100)
        
        print("法向量估计完成")
        return pcd
    
    def preprocess_pipeline(self, pcd: o3d.geometry.PointCloud,
                           downsample: bool = True,
                           remove_outliers: bool = True,
                           estimate_normals: bool = True) -> o3d.geometry.PointCloud:
        """
        完整的预处理流程
        
        Args:
            pcd: 输入点云
            downsample: 是否进行下采样
            remove_outliers: 是否移除离群点
            estimate_normals: 是否估计法向量
            
        Returns:
            预处理后的点云
        """
        print(f"开始预处理，原始点数: {len(pcd.points)}")
        
        # 下采样
        if downsample:
            pcd = self.voxel_downsample(pcd)
        
        # 移除离群点
        if remove_outliers:
            pcd, _ = self.remove_statistical_outliers(pcd)
        
        # 估计法向量
        if estimate_normals:
            pcd = self.estimate_normals(pcd)
        
        print(f"预处理完成，最终点数: {len(pcd.points)}")
        return pcd
    
    def crop_point_cloud(self, pcd: o3d.geometry.PointCloud,
                        min_bound: np.ndarray,
                        max_bound: np.ndarray) -> o3d.geometry.PointCloud:
        """
        裁剪点云到指定边界框
        
        Args:
            pcd: 输入点云
            min_bound: 最小边界 [x_min, y_min, z_min]
            max_bound: 最大边界 [x_max, y_max, z_max]
            
        Returns:
            裁剪后的点云
        """
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, 
                                                   max_bound=max_bound)
        cropped = pcd.crop(bbox)
        print(f"点云裁剪: {len(pcd.points)} -> {len(cropped.points)} 点")
        return cropped
    
    def compute_point_cloud_statistics(self, pcd: o3d.geometry.PointCloud) -> dict:
        """
        计算点云统计信息
        
        Args:
            pcd: 输入点云
            
        Returns:
            统计信息字典
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        stats = {
            'num_points': len(points),
            'bounds': {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist()
            },
            'centroid': points.mean(axis=0).tolist(),
            'std': points.std(axis=0).tolist(),
            'has_colors': pcd.has_colors(),
            'has_normals': pcd.has_normals()
        }
        
        if colors is not None:
            stats['color_stats'] = {
                'mean': colors.mean(axis=0).tolist(),
                'std': colors.std(axis=0).tolist()
            }
        
        return stats


def batch_preprocess_point_clouds(file_paths: List[str],
                                  voxel_size: float = 0.02,
                                  apply_outlier_removal: bool = True) -> List[o3d.geometry.PointCloud]:
    """
    批量预处理多个点云文件
    
    Args:
        file_paths: 点云文件路径列表
        voxel_size: 体素大小
        apply_outlier_removal: 是否应用离群点移除
        
    Returns:
        预处理后的点云列表
    """
    preprocessor = PointCloudPreprocessor(voxel_size=voxel_size)
    point_clouds = preprocessor.load_point_clouds(file_paths)
    
    processed_clouds = []
    for i, pcd in enumerate(point_clouds):
        print(f"\n预处理点云 {i+1}/{len(point_clouds)}")
        processed = preprocessor.preprocess_pipeline(
            pcd,
            downsample=True,
            remove_outliers=apply_outlier_removal,
            estimate_normals=True
        )
        processed_clouds.append(processed)
    
    return processed_clouds