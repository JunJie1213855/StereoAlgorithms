"""
特征分析工具模块
提供PFH和FPFH特征的质量分析和对比功能
"""
import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Optional
import time
from scipy.spatial import cKDTree


class FeatureAnalyzer:
    """特征分析类"""
    
    def __init__(self):
        """初始化特征分析器"""
        self.analysis_results = {}
    
    def analyze_feature_quality(self, features: o3d.pipelines.registration.Feature,
                              point_cloud: o3d.geometry.PointCloud) -> Dict:
        """
        分析特征质量
        
        Args:
            features: 特征数据
            point_cloud: 对应的点云
            
        Returns:
            特征质量分析结果
        """
        print("分析特征质量...")
        
        feature_data = np.asarray(features.data)
        
        analysis = {
            'num_features': features.dimension(),
            'num_points': len(point_cloud.points),
            'feature_mean': np.mean(feature_data, axis=0).tolist(),
            'feature_std': np.std(feature_data, axis=0).tolist(),
            'feature_min': np.min(feature_data, axis=0).tolist(),
            'feature_max': np.max(feature_data, axis=0).tolist(),
            'feature_range': (np.max(feature_data, axis=0) - np.min(feature_data, axis=0)).tolist(),
            'feature_sparsity': np.mean(feature_data == 0),
            'feature_entropy': self._compute_entropy(feature_data)
        }
        
        print(f"  特征维度: {analysis['num_features']}")
        print(f"  点数量: {analysis['num_points']}")
        print(f"  特征稀疏度: {analysis['feature_sparsity']:.4f}")
        print(f"  特征熵: {analysis['feature_entropy']:.4f}")
        
        return analysis
    
    def _compute_entropy(self, feature_data: np.ndarray) -> float:
        """计算特征的熵"""
        # 将特征值归一化到[0,1]范围
        feature_normalized = (feature_data - feature_data.min(axis=0)) / \
                           (feature_data.max(axis=0) - feature_data.min(axis=0) + 1e-8)
        
        # 计算直方图
        hist, _ = np.histogram(feature_normalized.flatten(), bins=50, range=(0, 1))
        
        # 计算熵
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        
        return entropy
    
    def compare_features(self, features1: o3d.pipelines.registration.Feature,
                        pcd1: o3d.geometry.PointCloud,
                        features2: o3d.pipelines.registration.Feature,
                        pcd2: o3d.geometry.PointCloud,
                        label1: str = "Features1",
                        label2: str = "Features2") -> Dict:
        """
        比较两组特征
        
        Args:
            features1: 第一组特征
            pcd1: 第一组点云
            features2: 第二组特征
            pcd2: 第二组点云
            label1: 第一组标签
            label2: 第二组标签
            
        Returns:
            比较结果
        """
        print(f"比较特征组: {label1} vs {label2}")
        
        # 分析每组特征
        analysis1 = self.analyze_feature_quality(features1, pcd1)
        analysis2 = self.analyze_feature_quality(features2, pcd2)
        
        # 计算特征相似性
        feature_data1 = np.asarray(features1.data)
        feature_data2 = np.asarray(features2.data)
        
        # 如果特征维度不同，无法直接比较
        if features1.dimension() != features2.dimension():
            print(f"  警告：特征维度不同 ({features1.dimension()} vs {features2.dimension()})")
            similarity = None
        else:
            # 计算特征分布的相似性
            similarity = self._compute_feature_similarity(feature_data1, feature_data2)
        
        comparison = {
            label1: analysis1,
            label2: analysis2,
            'similarity': similarity,
            'dimension_match': features1.dimension() == features2.dimension(),
            'point_ratio': len(pcd1.points) / len(pcd2.points) if len(pcd2.points) > 0 else float('inf')
        }
        
        if similarity is not None:
            print(f"  特征相似度: {similarity:.4f}")
        
        return comparison
    
    def _compute_feature_similarity(self, features1: np.ndarray, 
                                  features2: np.ndarray) -> float:
        """计算两组特征的相似度"""
        # 计算特征分布的统计相似性
        mean_diff = np.abs(np.mean(features1, axis=0) - np.mean(features2, axis=0))
        std_diff = np.abs(np.std(features1, axis=0) - np.std(features2, axis=0))
        
        # 归一化差异
        max_mean = np.maximum(np.mean(features1, axis=0), np.mean(features2, axis=0))
        max_std = np.maximum(np.std(features1, axis=0), np.std(features2, axis=0))
        
        normalized_diff = (mean_diff / (max_mean + 1e-8) + std_diff / (max_std + 1e-8)) / 2
        
        # 相似度 = 1 - 平均归一化差异
        similarity = 1 - np.mean(normalized_diff)
        
        return similarity
    
    def analyze_feature_matching_quality(self, 
                                       source_features: o3d.pipelines.registration.Feature,
                                       target_features: o3d.pipelines.registration.Feature,
                                       correspondences: List[Tuple[int, int]]) -> Dict:
        """
        分析特征匹配质量
        
        Args:
            source_features: 源特征
            target_features: 目标特征
            correspondences: 对应点对列表
            
        Returns:
            匹配质量分析结果
        """
        print("分析特征匹配质量...")
        
        if len(correspondences) == 0:
            return {
                'num_correspondences': 0,
                'match_quality': 'No matches found'
            }
        
        source_data = np.asarray(source_features.data)
        target_data = np.asarray(target_features.data)
        
        # 计算对应特征之间的距离
        feature_distances = []
        for src_idx, tgt_idx in correspondences:
            if src_idx < len(source_data) and tgt_idx < len(target_data):
                distance = np.linalg.norm(source_data[src_idx] - target_data[tgt_idx])
                feature_distances.append(distance)
        
        if len(feature_distances) == 0:
            return {
                'num_correspondences': len(correspondences),
                'match_quality': 'Invalid correspondences'
            }
        
        feature_distances = np.array(feature_distances)
        
        analysis = {
            'num_correspondences': len(correspondences),
            'mean_feature_distance': np.mean(feature_distances),
            'std_feature_distance': np.std(feature_distances),
            'min_feature_distance': np.min(feature_distances),
            'max_feature_distance': np.max(feature_distances),
            'median_feature_distance': np.median(feature_distances),
            'good_match_ratio': np.sum(feature_distances < np.median(feature_distances)) / len(feature_distances)
        }
        
        print(f"  对应点数量: {analysis['num_correspondences']}")
        print(f"  平均特征距离: {analysis['mean_feature_distance']:.4f}")
        print(f"  中位数特征距离: {analysis['median_feature_distance']:.4f}")
        print(f"  好匹配比例: {analysis['good_match_ratio']:.2%}")
        
        return analysis
    
    def benchmark_feature_methods(self, 
                                 source: o3d.geometry.PointCloud,
                                 target: o3d.geometry.PointCloud,
                                 methods: List[str] = None) -> Dict:
        """
        基准测试不同的特征方法
        
        Args:
            source: 源点云
            target: 目标点云
            methods: 要测试的方法列表，如果为None则测试所有方法
            
        Returns:
            基准测试结果
        """
        if methods is None:
            methods = ['fpfh_ransac', 'pfh_ransac', 'hybrid_fpfh', 'hybrid_pfh']
        
        print("=== 特征方法基准测试 ===")
        
        from core.features import PointCloudFeatures
        
        feature_calculator = PointCloudFeatures()
        benchmark_results = {}
        
        for method in methods:
            print(f"\n测试方法: {method}")
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 执行配准
                from core.features import feature_based_registration
                result = feature_based_registration(source, target, method=method)
                
                # 记录结束时间
                end_time = time.time()
                
                # 存储结果
                benchmark_results[method] = {
                    'success': True,
                    'execution_time': end_time - start_time,
                    'fitness': result.get('fitness', 0),
                    'inlier_rmse': result.get('inlier_rmse', float('inf')),
                    'transformation': result.get('transformation'),
                    'num_correspondences': len(result.get('correspondence_set', []))
                }
                
                print(f"  ✓ 成功")
                print(f"    时间: {benchmark_results[method]['execution_time']:.2f}秒")
                print(f"    Fitness: {benchmark_results[method]['fitness']:.4f}")
                print(f"    RMSE: {benchmark_results[method]['inlier_rmse']:.4f}")
                
            except Exception as e:
                benchmark_results[method] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': 0
                }
                print(f"  ✗ 失败: {e}")
        
        # 生成推荐
        print("\n=== 方法推荐 ===")
        recommendation = self._generate_method_recommendation(benchmark_results)
        print(recommendation['summary'])
        
        benchmark_results['recommendation'] = recommendation
        
        return benchmark_results
    
    def _generate_method_recommendation(self, benchmark_results: Dict) -> Dict:
        """根据基准测试结果生成推荐"""
        successful_methods = {k: v for k, v in benchmark_results.items() 
                            if v.get('success', False) and isinstance(v, dict)}
        
        if not successful_methods:
            return {
                'summary': '没有成功的方法',
                'recommended_method': None,
                'reason': '所有方法都失败了'
            }
        
        # 按不同标准评分
        best_fitness = max(successful_methods.items(), 
                          key=lambda x: x[1].get('fitness', 0))
        best_speed = min(successful_methods.items(), 
                        key=lambda x: x[1].get('execution_time', float('inf')))
        best_accuracy = min(successful_methods.items(), 
                          key=lambda x: x[1].get('inlier_rmse', float('inf')))
        
        # 综合评分 (考虑速度、精度和fitness)
        for method, result in successful_methods.items():
            if method != 'recommendation':
                fitness_score = result.get('fitness', 0)
                accuracy_score = 1 / (result.get('inlier_rmse', 1) + 1)
                speed_score = 1 / (result.get('execution_time', 1) + 1)
                
                # 加权综合评分
                result['overall_score'] = (0.4 * fitness_score + 
                                         0.4 * accuracy_score + 
                                         0.2 * speed_score)
        
        best_overall = max(successful_methods.items(), 
                          key=lambda x: x[1].get('overall_score', 0))
        
        summary = f"""
基于基准测试结果的方法推荐:

🏆 综合最佳: {best_overall[0]}
   - Fitness: {best_overall[1]['fitness']:.4f}
   - RMSE: {best_overall[1]['inlier_rmse']:.4f}  
   - 时间: {best_overall[1]['execution_time']:.2f}秒
   - 综合评分: {best_overall[1]['overall_score']:.4f}

⚡ 最快速度: {best_speed[0]} ({best_speed[1]['execution_time']:.2f}秒)
🎯 最高精度: {best_accuracy[0]} (RMSE: {best_accuracy[1]['inlier_rmse']:.4f})
💪 最佳Fitness: {best_fitness[0]} ({best_fitness[1]['fitness']:.4f})
"""
        
        return {
            'summary': summary,
            'recommended_method': best_overall[0],
            'best_fitness_method': best_fitness[0],
            'fastest_method': best_speed[0],
            'most_accurate_method': best_accuracy[0],
            'reason': f'综合考虑速度、精度和fitness，{best_overall[0]}表现最佳'
        }
    
    def analyze_feature_descriptiveness(self, 
                                     features: o3d.pipelines.registration.Feature,
                                     point_cloud: o3d.geometry.PointCloud,
                                     num_test_points: int = 100) -> Dict:
        """
        分析特征描述性（区分能力）
        
        Args:
            features: 特征数据
            point_cloud: 对应点云
            num_test_points: 测试点数量
            
        Returns:
            描述性分析结果
        """
        print("分析特征描述性...")
        
        feature_data = np.asarray(features.data)
        
        # 随机选择测试点
        num_points = min(num_test_points, len(feature_data))
        test_indices = np.random.choice(len(feature_data), num_points, replace=False)
        test_features = feature_data[test_indices]
        
        # 计算特征之间的最小距离（描述性指标）
        tree = cKDTree(feature_data)
        
        # 找到每个测试点的最近邻（除了自己）
        min_distances = []
        for i, idx in enumerate(test_indices):
            distances, indices = tree.query(test_features[i], k=2)  # k=2因为第一个是自己
            if len(distances) > 1:
                min_distances.append(distances[1])  # 第二近的点距离
        
        min_distances = np.array(min_distances)
        
        analysis = {
            'mean_min_distance': np.mean(min_distances),
            'std_min_distance': np.std(min_distances),
            'median_min_distance': np.median(min_distances),
            'descriptiveness_score': np.median(min_distances),  # 中位数距离越大，描述性越强
            'feature_separability': np.std(min_distances) / (np.mean(min_distances) + 1e-8)
        }
        
        print(f"  平均最小距离: {analysis['mean_min_distance']:.4f}")
        print(f"  中位数最小距离: {analysis['median_min_distance']:.4f}")
        print(f"  描述性得分: {analysis['descriptiveness_score']:.4f}")
        print(f"  特征可分性: {analysis['feature_separability']:.4f}")
        
        return analysis


def comprehensive_feature_analysis(source: o3d.geometry.PointCloud,
                                 target: o3d.geometry.PointCloud,
                                 methods: List[str] = None) -> Dict:
    """
    综合特征分析
    
    Args:
        source: 源点云
        target: 目标点云
        methods: 要分析的方法列表
        
    Returns:
        综合分析结果
    """
    print("=== 开始综合特征分析 ===")
    
    analyzer = FeatureAnalyzer()
    from core.features import PointCloudFeatures
    
    feature_calculator = PointCloudFeatures()
    
    # 计算不同类型的特征
    print("\n计算特征...")
    source_fpfh = feature_calculator.compute_fpfh_features(source)
    target_fpfh = feature_calculator.compute_fpfh_features(target)
    
    source_pfh = feature_calculator.compute_pfh_features(source)
    target_pfh = feature_calculator.compute_pfh_features(target)
    
    # 分析特征质量
    print("\n分析特征质量...")
    fpfh_quality = analyzer.analyze_feature_quality(source_fpfh, source)
    pfh_quality = analyzer.analyze_feature_quality(source_pfh, source)
    
    # 比较FPFH vs PFH
    print("\n比较FPFH vs PFH...")
    feature_comparison = analyzer.compare_features(
        source_fpfh, source, source_pfh, source, 
        "FPFH", "PFH"
    )
    
    # 基准测试
    print("\n执行基准测试...")
    benchmark_results = analyzer.benchmark_feature_methods(source, target, methods)
    
    # 分析特征描述性
    print("\n分析特征描述性...")
    fpfh_descriptiveness = analyzer.analyze_feature_descriptiveness(source_fpfh, source)
    pfh_descriptiveness = analyzer.analyze_feature_descriptiveness(source_pfh, source)
    
    comprehensive_results = {
        'feature_quality': {
            'FPFH': fpfh_quality,
            'PFH': pfh_quality
        },
        'feature_comparison': feature_comparison,
        'benchmark': benchmark_results,
        'descriptiveness': {
            'FPFH': fpfh_descriptiveness,
            'PFH': pfh_descriptiveness
        }
    }
    
    # 生成总结
    print("\n=== 分析总结 ===")
    summary = analyzer._generate_analysis_summary(comprehensive_results)
    print(summary)
    
    comprehensive_results['summary'] = summary
    
    return comprehensive_results