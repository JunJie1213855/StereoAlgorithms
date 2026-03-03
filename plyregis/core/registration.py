"""
点云配准算法核心模块
提供多种点云配准算法实现
"""
import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional, Dict
import copy


class PointCloudRegistration:
    """点云配准类"""
    
    def __init__(self, voxel_size: float = 0.02):
        """
        初始化配准器
        
        Args:
            voxel_size: 配准时使用的体素大小
        """
        self.voxel_size = voxel_size
        self.transformation_history = []
    
    def prepare_source_target(self, source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             voxel_size: Optional[float] = None) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        准备源点云和目标点云
        
        Args:
            source: 源点云
            target: 目标点云
            voxel_size: 下采样体素大小
            
        Returns:
            准备好的源点云和目标点云
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # 确保法向量存在
        if not source_down.has_normals():
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        if not target_down.has_normals():
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        return source_down, target_down
    
    def execute_icp(self, source: o3d.geometry.PointCloud,
                   target: o3d.geometry.PointCloud,
                   threshold: float = 0.02,
                   trans_init: Optional[np.ndarray] = None,
                   max_iteration: int = 50,
                   point_to_plane: bool = False) -> Dict:
        """
        执行ICP配准算法
        
        Args:
            source: 源点云
            target: 目标点云
            threshold: 距离阈值
            trans_init: 初始变换矩阵
            max_iteration: 最大迭代次数
            point_to_plane: 是否使用点对平面ICP
            
        Returns:
            配准结果字典
        """
        if trans_init is None:
            trans_init = np.eye(4)
        
        source_down, target_down = self.prepare_source_target(source, target)
        
        if point_to_plane:
            # 点对平面ICP
            print("执行点对平面ICP配准...")
            result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
            )
        else:
            # 点对点ICP
            print("执行点对点ICP配准...")
            result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
            )
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set
        }
    
    def execute_colored_icp(self, source: o3d.geometry.PointCloud,
                           target: o3d.geometry.PointCloud,
                           voxel_size: Optional[float] = None,
                           trans_init: Optional[np.ndarray] = None,
                           max_iteration: int = 50) -> Dict:
        """
        执行颜色ICP配准算法
        
        Args:
            source: 源点云
            target: 目标点云
            voxel_size: 体素大小
            trans_init: 初始变换矩阵
            max_iteration: 最大迭代次数
            
        Returns:
            配准结果字典
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        if trans_init is None:
            trans_init = np.eye(4)
        
        # 检查颜色信息
        if not source.has_colors() or not target.has_colors():
            print("警告：点云缺少颜色信息，将使用普通ICP")
            return self.execute_icp(source, target, voxel_size, trans_init, max_iteration)
        
        print("执行颜色ICP配准...")
        source_down, target_down = self.prepare_source_target(source, target, voxel_size)
        
        result = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, voxel_size, trans_init,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=max_iteration
            )
        )
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set
        }
    
    def execute_ndt(self, source: o3d.geometry.PointCloud,
                   target: o3d.geometry.PointCloud,
                   threshold: float = 0.02,
                   trans_init: Optional[np.ndarray] = None,
                   max_iteration: int = 50) -> Dict:
        """
        执行NDT配准算法
        
        注意：如果Open3D版本不支持NDT，将回退到FGR
        
        Args:
            source: 源点云
            target: 目标点云
            threshold: 距离阈值
            trans_init: 初始变换矩阵
            max_iteration: 最大迭代次数
            
        Returns:
            配准结果字典
        """
        if trans_init is None:
            trans_init = np.eye(4)
        
        print("执行NDT配准...")
        source_down, target_down = self.prepare_source_target(source, target)
        
        try:
            # 尝试使用NDT
            result = o3d.pipelines.registration.registration_ndt(
                source_down, target_down, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationForNDT(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
            )
        except AttributeError:
            # 如果NDT不可用，回退到FGR
            print("警告：NDT不可用，使用FGR作为替代")
            return self.execute_fgr(source, target)
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set
        }
    
    def execute_fgr(self, source: o3d.geometry.PointCloud,
                   target: o3d.geometry.PointCloud,
                   voxel_size: Optional[float] = None) -> Dict:
        """
        执行快速全局配准(Fast Global Registration)
        
        Args:
            source: 源点云
            target: 目标点云
            voxel_size: 体素大小
            
        Returns:
            配准结果字典
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        print("执行快速全局配准...")
        source_down, target_down = self.prepare_source_target(source, target, voxel_size)
        
        # 计算FPFH特征
        print("计算FPFH特征...")
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        
        # 执行快速全局配准
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=voxel_size * 0.5
            )
        )
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set
        }
    
    def coarse_to_fine_registration(self, source: o3d.geometry.PointCloud,
                                   target: o3d.geometry.PointCloud,
                                   use_ndt_coarse: bool = True,
                                   use_colored_icp: bool = True) -> Dict:
        """
        粗到细配准流程
        
        Args:
            source: 源点云
            target: 目标点云
            use_ndt_coarse: 是否使用NDT进行粗配准
            use_colored_icp: 是否使用颜色ICP进行精配准
            
        Returns:
            配准结果字典
        """
        print("=== 开始粗到细配准流程 ===")
        
        # 粗配准
        if use_ndt_coarse:
            print("1. 粗配准阶段 (NDT)")
            coarse_result = self.execute_ndt(source, target, 
                                           threshold=self.voxel_size * 5,
                                           trans_init=np.eye(4),
                                           max_iteration=50)
            trans_init = coarse_result['transformation']
            print(f"   粗配准 - Fitness: {coarse_result['fitness']:.4f}, "
                  f"RMSE: {coarse_result['inlier_rmse']:.4f}")
        else:
            # 使用快速全局配准作为粗配准
            print("1. 粗配准阶段 (FGR)")
            coarse_result = self.execute_fgr(source, target)
            trans_init = coarse_result['transformation']
            print(f"   粗配准 - Fitness: {coarse_result['fitness']:.4f}, "
                  f"RMSE: {coarse_result['inlier_rmse']:.4f}")
        
        # 精配准
        if use_colored_icp and source.has_colors() and target.has_colors():
            print("2. 精配准阶段 (颜色ICP)")
            fine_result = self.execute_colored_icp(source, target,
                                                   voxel_size=self.voxel_size,
                                                   trans_init=trans_init,
                                                   max_iteration=50)
        else:
            print("2. 精配准阶段 (点对平面ICP)")
            fine_result = self.execute_icp(source, target,
                                          threshold=self.voxel_size,
                                          trans_init=trans_init,
                                          max_iteration=50,
                                          point_to_plane=True)
        
        print(f"   精配准 - Fitness: {fine_result['fitness']:.4f}, "
              f"RMSE: {fine_result['inlier_rmse']:.4f}")
        print("=== 配准流程完成 ===")
        
        return fine_result
    
    def multiway_registration(self, point_clouds: List[o3d.geometry.PointCloud],
                             reference_idx: int = 0) -> Tuple[List[o3d.geometry.PointCloud], List[np.ndarray]]:
        """
        多视角配准
        
        Args:
            point_clouds: 点云列表
            reference_idx: 参考点云索引
            
        Returns:
            配准后的点云列表和变换矩阵列表
        """
        print(f"=== 开始多视角配准，参考点云索引: {reference_idx} ===")
        
        if reference_idx >= len(point_clouds):
            raise ValueError(f"参考索引 {reference_idx} 超出范围")
        
        # 复制点云以避免修改原始数据
        aligned_clouds = [copy.deepcopy(pc) for pc in point_clouds]
        transformations = [np.eye(4) for _ in range(len(point_clouds))]
        
        reference = aligned_clouds[reference_idx]
        
        # 将每个点云配准到参考点云
        for i, pcd in enumerate(aligned_clouds):
            if i == reference_idx:
                continue
            
            print(f"\n配准点云 {i} 到参考点云 {reference_idx}")
            result = self.coarse_to_fine_registration(pcd, reference)
            
            # 应用变换
            transformations[i] = result['transformation']
            aligned_clouds[i] = pcd.transform(result['transformation'])
            
            print(f"配准完成 - Fitness: {result['fitness']:.4f}, "
                  f"RMSE: {result['inlier_rmse']:.4f}")
        
        print("\n=== 多视角配准完成 ===")
        return aligned_clouds, transformations


def pairwise_registration(source: o3d.geometry.PointCloud,
                         target: o3d.geometry.PointCloud,
                         method: str = 'coarse_to_fine',
                         voxel_size: float = 0.02) -> Dict:
    """
    两两配准的便捷函数
    
    Args:
        source: 源点云
        target: 目标点云
        method: 配准方法 ('icp', 'colored_icp', 'ndt', 'fgr', 'coarse_to_fine')
        voxel_size: 体素大小
        
    Returns:
        配准结果字典
    """
    registration = PointCloudRegistration(voxel_size=voxel_size)
    
    if method == 'icp':
        return registration.execute_icp(source, target)
    elif method == 'colored_icp':
        return registration.execute_colored_icp(source, target)
    elif method == 'ndt':
        return registration.execute_ndt(source, target)
    elif method == 'fgr':
        return registration.execute_fgr(source, target)
    elif method == 'coarse_to_fine':
        return registration.coarse_to_fine_registration(source, target)
    else:
        raise ValueError(f"未知的配准方法: {method}")