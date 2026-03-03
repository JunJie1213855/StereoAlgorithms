"""
点云特征计算模块
提供PFH和FPFH特征计算和基于特征的配准算法
"""
import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional, Dict
import copy


class PointCloudFeatures:
    """点云特征计算类"""
    
    def __init__(self, voxel_size: float = 0.02):
        """
        初始化特征计算器
        
        Args:
            voxel_size: 默认体素大小
        """
        self.voxel_size = voxel_size
    
    def compute_fpfh_features(self, pcd: o3d.geometry.PointCloud,
                             voxel_size: Optional[float] = None,
                             search_radius: Optional[float] = None) -> o3d.pipelines.registration.Feature:
        """
        计算FPFH特征
        
        Args:
            pcd: 输入点云
            voxel_size: 体素大小，如果为None则使用默认值
            search_radius: 搜索半径，如果为None则自动计算
            
        Returns:
            FPFH特征
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
            
        if search_radius is None:
            search_radius = voxel_size * 5
        
        print("计算FPFH特征...")
        
        # 确保点云有法向量
        if not pcd.has_normals():
            print("  点云无法向量，正在估计...")
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30)
            )
        
        # 下采样以提高计算效率
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # 重新估计下采样后点云的法向量
        if not pcd_down.has_normals():
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30)
            )
        
        # 计算FPFH特征
        fpfh_features = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=100)
        )
        
        print(f"  FPFH特征计算完成: {fpfh_features.dimension()}维特征, {len(pcd_down.points)}个点")
        return fpfh_features
    
    def compute_pfh_features(self, pcd: o3d.geometry.PointCloud,
                            voxel_size: Optional[float] = None,
                            search_radius: Optional[float] = None) -> np.ndarray:
        """
        计算PFH特征（手动实现）
        
        注意：Open3D不直接支持PFH，这里提供一个简化实现
        
        Args:
            pcd: 输入点云
            voxel_size: 体素大小
            search_radius: 搜索半径
            
        Returns:
            PFH特征数组
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
            
        if search_radius is None:
            search_radius = voxel_size * 5
        
        print("计算PFH特征（简化实现）...")
        
        # 确保点云有法向量
        if not pcd.has_normals():
            print("  点云无法向量，正在估计...")
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30)
            )
        
        # 下采样
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # 重新估计法向量
        if not pcd_down.has_normals():
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30)
            )
        
        # 计算简化的PFH特征
        # 这里使用Open3D的FPFH作为PFH的近似
        # 真正的PFH计算复杂度更高，需要O(nk²)的时间
        pfh_features = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=50)
        )
        
        print(f"  PFH特征计算完成: {pfh_features.dimension()}维特征, {len(pcd_down.points)}个点")
        print("  注意：此实现使用FPFH作为PFH的近似，真正的PFH计算复杂度更高")
        
        return pfh_features
    
    def execute_fpfh_ransac_registration(self, source: o3d.geometry.PointCloud,
                                       target: o3d.geometry.PointCloud,
                                       voxel_size: Optional[float] = None,
                                       distance_threshold: float = 0.02,
                                       max_iterations: int = 100000) -> Dict:
        """
        基于FPFH特征的RANSAC配准
        
        Args:
            source: 源点云
            target: 目标点云
            voxel_size: 体素大小
            distance_threshold: 距离阈值
            max_iterations: 最大迭代次数
            
        Returns:
            配准结果字典
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        print("=== 执行FPFH-RANSAC配准 ===")
        
        # 计算FPFH特征
        print("1. 计算FPFH特征")
        source_fpfh = self.compute_fpfh_features(source, voxel_size)
        target_fpfh = self.compute_fpfh_features(target, voxel_size)
        
        # 下采样点云
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # 执行RANSAC配准
        print("2. 执行RANSAC配准")
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations)
        )
        
        print("3. 配准完成")
        print(f"   Fitness: {result.fitness:.4f}")
        print(f"   RMSE: {result.inlier_rmse:.4f}")
        print(f"   对应点数量: {len(result.correspondence_set)}")
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set,
            'method': 'fpfh_ransac'
        }
    
    def execute_pfh_ransac_registration(self, source: o3d.geometry.PointCloud,
                                      target: o3d.geometry.PointCloud,
                                      voxel_size: Optional[float] = None,
                                      distance_threshold: float = 0.02,
                                      max_iterations: int = 50000) -> Dict:
        """
        基于PFH特征的RANSAC配准
        
        Args:
            source: 源点云
            target: 目标点云
            voxel_size: 体素大小
            distance_threshold: 距离阈值
            max_iterations: 最大迭代次数
            
        Returns:
            配准结果字典
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        print("=== 执行PFH-RANSAC配准 ===")
        
        # 计算PFH特征（使用FPFH近似）
        print("1. 计算PFH特征")
        source_pfh = self.compute_pfh_features(source, voxel_size)
        target_pfh = self.compute_pfh_features(target, voxel_size)
        
        # 下采样点云
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # 执行RANSAC配准
        print("2. 执行RANSAC配准")
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_pfh, target_pfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations)
        )
        
        print("3. 配准完成")
        print(f"   Fitness: {result.fitness:.4f}")
        print(f"   RMSE: {result.inlier_rmse:.4f}")
        print(f"   对应点数量: {len(result.correspondence_set)}")
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set,
            'method': 'pfh_ransac'
        }
    
    def execute_hybrid_feature_registration(self, source: o3d.geometry.PointCloud,
                                          target: o3d.geometry.PointCloud,
                                          voxel_size: Optional[float] = None,
                                          use_pfh_coarse: bool = True,
                                          use_icp_fine: bool = True) -> Dict:
        """
        混合特征配准方法
        
        先使用特征方法进行粗配准，再使用ICP进行精配准
        
        Args:
            source: 源点云
            target: 目标点云
            voxel_size: 体素大小
            use_pfh_coarse: 是否使用PFH进行粗配准（False则使用FPFH）
            use_icp_fine: 是否使用ICP进行精配准
            
        Returns:
            配准结果字典
        """
        print("=== 执行混合特征配准 ===")
        
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        # 粗配准阶段
        if use_pfh_coarse:
            print("1. 粗配准阶段 (PFH-RANSAC)")
            coarse_result = self.execute_pfh_ransac_registration(source, target, voxel_size)
            feature_method = 'PFH'
        else:
            print("1. 粗配准阶段 (FPFH-RANSAC)")
            coarse_result = self.execute_fpfh_ransac_registration(source, target, voxel_size)
            feature_method = 'FPFH'
        
        trans_init = coarse_result['transformation']
        print(f"   {feature_method}粗配准 - Fitness: {coarse_result['fitness']:.4f}, "
              f"RMSE: {coarse_result['inlier_rmse']:.4f}")
        
        # 精配准阶段
        final_result = coarse_result
        
        if use_icp_fine:
            print("2. 精配准阶段 (ICP)")
            
            # 应用粗配准变换
            source_transformed = source.transform(trans_init)
            
            # 使用ICP进行精配准
            from core.registration import PointCloudRegistration
            reg = PointCloudRegistration(voxel_size=voxel_size)
            
            icp_result = reg.execute_icp(source_transformed, target, 
                                        trans_init=np.eye(4),
                                        point_to_plane=True)
            
            # 合并变换矩阵
            final_transformation = icp_result['transformation'] @ trans_init
            
            final_result = {
                'transformation': final_transformation,
                'fitness': icp_result['fitness'],
                'inlier_rmse': icp_result['inlier_rmse'],
                'correspondence_set': icp_result['correspondence_set'],
                'method': f'hybrid_{feature_method}_icp',
                'coarse_result': coarse_result,
                'fine_result': icp_result
            }
            
            print(f"   ICP精配准 - Fitness: {icp_result['fitness']:.4f}, "
                  f"RMSE: {icp_result['inlier_rmse']:.4f}")
        
        print("=== 混合特征配准完成 ===")
        return final_result
    
    def compare_feature_methods(self, source: o3d.geometry.PointCloud,
                              target: o3d.geometry.PointCloud,
                              voxel_size: Optional[float] = None) -> Dict:
        """
        比较不同特征配准方法的效果
        
        Args:
            source: 源点云
            target: 目标点云
            voxel_size: 体素大小
            
        Returns:
            比较结果字典
        """
        print("=== 比较不同特征配准方法 ===")
        
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        results = {}
        
        # FPFH-RANSAC
        print("\n1. FPFH-RANSAC配准")
        try:
            fpfh_result = self.execute_fpfh_ransac_registration(source, target, voxel_size)
            results['fpfh_ransac'] = fpfh_result
        except Exception as e:
            print(f"FPFH-RANSAC失败: {e}")
            results['fpfh_ransac'] = {'error': str(e)}
        
        # PFH-RANSAC
        print("\n2. PFH-RANSAC配准")
        try:
            pfh_result = self.execute_pfh_ransac_registration(source, target, voxel_size)
            results['pfh_ransac'] = pfh_result
        except Exception as e:
            print(f"PFH-RANSAC失败: {e}")
            results['pfh_ransac'] = {'error': str(e)}
        
        # 混合方法
        print("\n3. 混合特征配准")
        try:
            hybrid_result = self.execute_hybrid_feature_registration(source, target, voxel_size)
            results['hybrid'] = hybrid_result
        except Exception as e:
            print(f"混合方法失败: {e}")
            results['hybrid'] = {'error': str(e)}
        
        # 比较结果
        print("\n=== 方法比较结果 ===")
        for method, result in results.items():
            if 'error' not in result:
                print(f"{method}:")
                print(f"  Fitness: {result['fitness']:.4f}")
                print(f"  RMSE: {result['inlier_rmse']:.4f}")
            else:
                print(f"{method}: 失败 - {result['error']}")
        
        return results


def feature_based_registration(source: o3d.geometry.PointCloud,
                              target: o3d.geometry.PointCloud,
                              method: str = 'fpfh_ransac',
                              voxel_size: float = 0.02) -> Dict:
    """
    基于特征的配准便捷函数
    
    Args:
        source: 源点云
        target: 目标点云
        method: 配准方法 ('fpfh_ransac', 'pfh_ransac', 'hybrid_fpfh', 'hybrid_pfh')
        voxel_size: 体素大小
        
    Returns:
        配准结果字典
    """
    features = PointCloudFeatures(voxel_size=voxel_size)
    
    if method == 'fpfh_ransac':
        return features.execute_fpfh_ransac_registration(source, target)
    elif method == 'pfh_ransac':
        return features.execute_pfh_ransac_registration(source, target)
    elif method == 'hybrid_fpfh':
        return features.execute_hybrid_feature_registration(source, target, use_pfh_coarse=False)
    elif method == 'hybrid_pfh':
        return features.execute_hybrid_feature_registration(source, target, use_pfh_coarse=True)
    else:
        raise ValueError(f"未知的特征配准方法: {method}")