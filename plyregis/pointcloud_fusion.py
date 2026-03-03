#!/usr/bin/env python3
"""
点云配准与融合主程序
支持多视角点云的自动配准和融合
"""
import os
import sys
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List
import warnings

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.preprocessing import PointCloudPreprocessor, batch_preprocess_point_clouds
from core.registration import PointCloudRegistration
from core.fusion import PointCloudFusion
from utils.visualization import PointCloudVisualizer
from utils.evaluation import RegistrationEvaluator, FusionEvaluator, generate_evaluation_report


class PointCloudFusionPipeline:
    """点云配准融合流程类"""
    
    def __init__(self, voxel_size: float = 0.02):
        """
        初始化流程
        
        Args:
            voxel_size: 处理时使用的体素大小
        """
        self.voxel_size = voxel_size
        self.preprocessor = PointCloudPreprocessor(voxel_size)
        self.registration = PointCloudRegistration(voxel_size)
        self.fusion = PointCloudFusion(voxel_size)
        self.visualizer = PointCloudVisualizer()
        self.reg_evaluator = RegistrationEvaluator()
        self.fusion_evaluator = FusionEvaluator()
        
        # 存储处理结果
        self.original_clouds = []
        self.preprocessed_clouds = []
        self.aligned_clouds = []
        self.fused_cloud = None
        self.transformations = []
        self.registration_results = []
    
    def load_point_clouds(self, file_paths: List[str]) -> bool:
        """
        加载点云文件
        
        Args:
            file_paths: 点云文件路径列表
            
        Returns:
            是否成功加载
        """
        print("=== 开始加载点云文件 ===")
        
        self.original_clouds = self.preprocessor.load_point_clouds(file_paths)
        
        if len(self.original_clouds) == 0:
            print("错误：未能加载任何点云文件")
            return False
        
        print(f"成功加载 {len(self.original_clouds)} 个点云文件")
        return True
    
    def preprocess_point_clouds(self, downsample: bool = True,
                               remove_outliers: bool = True,
                               estimate_normals: bool = True) -> None:
        """
        预处理点云
        
        Args:
            downsample: 是否下采样
            remove_outliers: 是否移除离群点
            estimate_normals: 是否估计法向量
        """
        print("=== 开始预处理点云 ===")
        
        self.preprocessed_clouds = []
        for i, pcd in enumerate(self.original_clouds):
            print(f"\n预处理点云 {i+1}/{len(self.original_clouds)}")
            processed = self.preprocessor.preprocess_pipeline(
                pcd,
                downsample=downsample,
                remove_outliers=remove_outliers,
                estimate_normals=estimate_normals
            )
            self.preprocessed_clouds.append(processed)
        
        print(f"\n预处理完成，处理了 {len(self.preprocessed_clouds)} 个点云")
    
    def register_point_clouds(self, reference_idx: int = 0,
                            method: str = 'coarse_to_fine') -> None:
        """
        配准点云
        
        Args:
            reference_idx: 参考点云索引
            method: 配准方法
        """
        print("=== 开始点云配准 ===")
        print(f"参考点云索引: {reference_idx}")
        print(f"配准方法: {method}")
        
        if reference_idx >= len(self.preprocessed_clouds):
            raise ValueError(f"参考索引 {reference_idx} 超出范围")
        
        # 使用多视角配准
        if method == 'multiway':
            self.aligned_clouds, self.transformations = \
                self.registration.multiway_registration(self.preprocessed_clouds, reference_idx)
        else:
            # 依次配准到参考点云
            reference = self.preprocessed_clouds[reference_idx]
            self.aligned_clouds = [copy.deepcopy(pcd) for pcd in self.preprocessed_clouds]
            self.transformations = [np.eye(4) for _ in range(len(self.preprocessed_clouds))]
            
            for i, pcd in enumerate(self.preprocessed_clouds):
                if i == reference_idx:
                    continue
                
                print(f"\n配准点云 {i} 到参考点云 {reference_idx}")
                
                # 选择配准方法
                if method == 'coarse_to_fine':
                    result = self.registration.coarse_to_fine_registration(pcd, reference)
                elif method == 'icp':
                    result = self.registration.execute_icp(pcd, reference)
                elif method == 'colored_icp':
                    result = self.registration.execute_colored_icp(pcd, reference)
                elif method == 'ndt':
                    result = self.registration.execute_ndt(pcd, reference)
                elif method in ['fpfh_ransac', 'pfh_ransac', 'hybrid_fpfh', 'hybrid_pfh']:
                    # 使用特征配准方法
                    from core.features import feature_based_registration
                    result = feature_based_registration(pcd, reference, method=method)
                else:
                    raise ValueError(f"未知的配准方法: {method}")
                
                # 应用变换
                self.transformations[i] = result['transformation']
                self.aligned_clouds[i] = pcd.transform(result['transformation'])
                self.registration_results.append(result)
                
                print(f"配准完成 - Fitness: {result['fitness']:.4f}, "
                      f"RMSE: {result['inlier_rmse']:.4f}")
        
        print("\n=== 点云配准完成 ===")
    
    def fuse_point_clouds(self, method: str = 'color_aware') -> o3d.geometry.PointCloud:
        """
        融合点云
        
        Args:
            method: 融合方法
            
        Returns:
            融合后的点云
        """
        print("=== 开始点云融合 ===")
        print(f"融合方法: {method}")
        
        self.fused_cloud = self.fusion.smart_fusion(self.aligned_clouds, method=method)
        
        print(f"融合完成，最终点数: {len(self.fused_cloud.points)}")
        return self.fused_cloud
    
    def evaluate_results(self) -> dict:
        """
        评估处理结果
        
        Returns:
            评估结果字典
        """
        print("=== 开始评估结果 ===")
        
        results = {}
        
        # 评估配准质量
        if self.registration_results:
            print("\n评估配准质量...")
            results['registration'] = self.registration_results
        
        # 评估融合质量
        if self.fused_cloud is not None:
            print("\n评估融合质量...")
            fusion_results = self.fusion_evaluator.evaluate_fusion_result(
                self.original_clouds, self.fused_cloud)
            results['fusion'] = fusion_results
            
            # 检测配准伪影
            artifacts = self.fusion_evaluator.detect_registration_artifacts(self.aligned_clouds)
            results['artifacts'] = artifacts
        
        # 计算重叠矩阵
        if len(self.aligned_clouds) > 1:
            print("\n计算重叠矩阵...")
            overlap_matrix = self.fusion_evaluator.compute_overlap_matrices(self.aligned_clouds)
            results['overlap_matrix'] = overlap_matrix
        
        return results
    
    def visualize_results(self, show_step_by_step: bool = False) -> None:
        """
        可视化结果
        
        Args:
            show_step_by_step: 是否逐步显示处理过程
        """
        print("=== 开始可视化 ===")
        
        if show_step_by_step:
            # 逐步显示
            print("1. 原始点云")
            self.visualizer.visualize_multiple_point_clouds(self.original_clouds)
            
            print("2. 预处理后点云")
            self.visualizer.visualize_multiple_point_clouds(self.preprocessed_clouds)
            
            print("3. 配准后点云")
            self.visualizer.visualize_multiple_point_clouds(self.aligned_clouds)
            
            print("4. 融合后点云")
            self.visualizer.visualize_single_point_cloud(self.fused_cloud)
        else:
            # 显示融合结果对比
            print("显示融合过程")
            self.visualizer.visualize_fusion_process(
                self.original_clouds, self.aligned_clouds, self.fused_cloud)
    
    def save_results(self, output_dir: str, save_intermediate: bool = False) -> None:
        """
        保存结果
        
        Args:
            output_dir: 输出目录
            save_intermediate: 是否保存中间结果
        """
        print("=== 保存结果 ===")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 保存融合后的点云
        if self.fused_cloud is not None:
            fused_path = output_path / "fused_pointcloud.ply"
            o3d.io.write_point_cloud(str(fused_path), self.fused_cloud)
            print(f"融合点云已保存: {fused_path}")
        
        # 保存配准后的点云
        if save_intermediate:
            aligned_dir = output_path / "aligned"
            aligned_dir.mkdir(exist_ok=True)
            
            for i, pcd in enumerate(self.aligned_clouds):
                aligned_path = aligned_dir / f"aligned_{i}.ply"
                o3d.io.write_point_cloud(str(aligned_path), pcd)
            
            print(f"配准点云已保存到: {aligned_dir}")
        
        # 保存评估报告
        evaluation_results = self.evaluate_results()
        report_path = output_path / "evaluation_report.txt"
        report = generate_evaluation_report(
            self.registration_results, 
            evaluation_results.get('fusion', {}),
            str(report_path)
        )
        print(f"评估报告已保存: {report_path}")
        
        print(f"所有结果已保存到: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='点云配准与融合系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 处理多个点云文件
  python pointcloud_fusion.py -i cloud1.ply cloud2.ply cloud3.ply -o results/
  
  # 使用特定配准方法
  python pointcloud_fusion.py -i *.ply -m colored_icp -o results/
  
  # 使用FPFH特征配准
  python pointcloud_fusion.py -i *.ply -m fpfh_ransac -o results/
  
  # 使用混合特征配准（推荐）
  python pointcloud_fusion.py -i *.ply -m hybrid_fpfh -o results/
  
  # 指定参考点云索引
  python pointcloud_fusion.py -i *.ply -r 0 -o results/
  
  # 保存中间结果
  python pointcloud_fusion.py -i *.ply -s -o results/
        '''
    )
    
    # 输入参数
    parser.add_argument('-i', '--input', nargs='+', required=True,
                       help='输入点云文件路径（支持通配符）')
    parser.add_argument('-r', '--reference', type=int, default=0,
                       help='参考点云索引（默认：0）')
    
    # 处理参数
    parser.add_argument('-v', '--voxel_size', type=float, default=0.02,
                       help='体素大小（默认：0.02）')
    parser.add_argument('-m', '--registration_method', 
                       choices=['coarse_to_fine', 'icp', 'colored_icp', 'ndt', 'multiway', 
                              'fpfh_ransac', 'pfh_ransac', 'hybrid_fpfh', 'hybrid_pfh'],
                       default='coarse_to_fine',
                       help='配准方法（默认：coarse_to_fine）')
    parser.add_argument('-f', '--fusion_method',
                       choices=['simple', 'voxel', 'statistical', 'color_aware', 'mls'],
                       default='color_aware',
                       help='融合方法（默认：color_aware）')
    
    # 输出参数
    parser.add_argument('-o', '--output', default='./fusion_output',
                       help='输出目录（默认：./fusion_output）')
    parser.add_argument('-s', '--save_intermediate', action='store_true',
                       help='保存中间结果')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                       help='启用可视化')
    parser.add_argument('--step_by_step', action='store_true',
                       help='逐步显示处理过程')
    
    # 预处理参数
    parser.add_argument('--no_downsample', action='store_true',
                       help='不进行下采样')
    parser.add_argument('--no_outlier_removal', action='store_true',
                       help='不进行离群点移除')
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_files = []
    for file_path in args.input:
        if os.path.exists(file_path):
            input_files.append(file_path)
        else:
            warnings.warn(f"文件不存在: {file_path}")
    
    if len(input_files) < 2:
        print("错误：至少需要两个点云文件进行配准")
        return 1
    
    print(f"找到 {len(input_files)} 个点云文件")
    
    try:
        # 创建处理流程
        pipeline = PointCloudFusionPipeline(voxel_size=args.voxel_size)
        
        # 1. 加载点云
        if not pipeline.load_point_clouds(input_files):
            return 1
        
        # 2. 预处理
        pipeline.preprocess_point_clouds(
            downsample=not args.no_downsample,
            remove_outliers=not args.no_outlier_removal
        )
        
        # 3. 配准
        pipeline.register_point_clouds(
            reference_idx=args.reference,
            method=args.registration_method
        )
        
        # 4. 融合
        pipeline.fuse_point_clouds(method=args.fusion_method)
        
        # 5. 评估
        evaluation_results = pipeline.evaluate_results()
        print("\n=== 评估结果摘要 ===")
        if 'fusion' in evaluation_results:
            fusion_res = evaluation_results['fusion']
            print(f"原始点云总数: {fusion_res['total_original_points']}")
            print(f"融合后点数: {fusion_res['fused_points']}")
            print(f"压缩比: {fusion_res['compression_ratio']:.2%}")
            print(f"减少百分比: {fusion_res['reduction_percentage']:.1f}%")
        
        # 6. 可视化
        if args.visualize:
            pipeline.visualize_results(show_step_by_step=args.step_by_step)
        
        # 7. 保存结果
        pipeline.save_results(args.output, save_intermediate=args.save_intermediate)
        
        print("\n=== 处理完成 ===")
        return 0
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import copy  # 添加copy导入
    sys.exit(main())