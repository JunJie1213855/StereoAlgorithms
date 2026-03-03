"""
可视化工具模块
提供点云配准和融合过程的可视化功能
"""
import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple
import copy


class PointCloudVisualizer:
    """点云可视化类"""
    
    def __init__(self):
        """初始化可视化器"""
        self.vis_windows = []
    
    def visualize_single_point_cloud(self, pcd: o3d.geometry.PointCloud,
                                    window_name: str = "Point Cloud",
                                    point_size: float = 1.0) -> None:
        """
        可视化单个点云
        
        Args:
            pcd: 点云对象
            window_name: 窗口名称
            point_size: 点的大小
        """
        # 设置点大小
        pcd_copy = copy.deepcopy(pcd)
        
        print(f"可视化点云: {window_name}, 点数: {len(pcd.points)}")
        
        # 如果有颜色，显示颜色；否则显示法向量
        if not pcd_copy.has_colors():
            # 根据深度设置颜色
            points = np.asarray(pcd_copy.points)
            depths = np.linalg.norm(points, axis=1)
            min_depth, max_depth = depths.min(), depths.max()
            normalized_depths = (depths - min_depth) / (max_depth - min_depth + 1e-6)
            colors = np.zeros((len(points), 3))
            colors[:, 0] = normalized_depths  # Red channel
            colors[:, 1] = 1 - normalized_depths  # Blue channel
            pcd_copy.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.visualization.draw_geometries([pcd_copy], window_name=window_name)
    
    def visualize_multiple_point_clouds(self, point_clouds: List[o3d.geometry.PointCloud],
                                       names: Optional[List[str]] = None,
                                       window_name: str = "Multiple Point Clouds") -> None:
        """
        可视化多个点云
        
        Args:
            point_clouds: 点云列表
            names: 点云名称列表
            window_name: 窗口名称
        """
        if names is None:
            names = [f"PointCloud_{i}" for i in range(len(point_clouds))]
        
        if len(point_clouds) != len(names):
            raise ValueError("点云数量和名称数量不匹配")
        
        print(f"可视化 {len(point_clouds)} 个点云")
        
        # 为每个点云分配不同的颜色
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
            [1, 0.5, 0],  # Orange
            [0.5, 0, 1],  # Purple
        ]
        
        colored_clouds = []
        for i, pcd in enumerate(point_clouds):
            pcd_copy = copy.deepcopy(pcd)
            
            if not pcd_copy.has_colors():
                # 为没有颜色的点云分配颜色
                color = colors[i % len(colors)]
                uniform_color = np.tile(color, (len(pcd_copy.points), 1))
                pcd_copy.colors = o3d.utility.Vector3dVector(uniform_color)
            
            colored_clouds.append(pcd_copy)
            print(f"  - {names[i]}: {len(pcd.points)} 点")
        
        o3d.visualization.draw_geometries(colored_clouds, window_name=window_name)
    
    def visualize_registration_result(self, source: o3d.geometry.PointCloud,
                                     target: o3d.geometry.PointCloud,
                                     transformed_source: o3d.geometry.PointCloud,
                                     transformation: np.ndarray,
                                     window_name: str = "Registration Result") -> None:
        """
        可视化配准结果
        
        Args:
            source: 原始源点云
            target: 目标点云
            transformed_source: 变换后的源点云
            transformation: 变换矩阵
            window_name: 窗口名称
        """
        print("可视化配准结果")
        
        # 创建临时点云用于可视化
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        transformed_temp = copy.deepcopy(transformed_source)
        
        # 为不同点云设置颜色
        if not source_temp.has_colors():
            source_temp.paint_uniform_color([1, 0, 0])  # Red
        if not target_temp.has_colors():
            target_temp.paint_uniform_color([0, 1, 0])  # Green
        if not transformed_temp.has_colors():
            transformed_temp.paint_uniform_color([0, 0, 1])  # Blue
        
        # 创建坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
        print(f"变换矩阵:\n{transformation}")
        
        o3d.visualization.draw_geometries(
            [source_temp, target_temp, transformed_temp, coordinate_frame],
            window_name=window_name
        )
    
    def visualize_fusion_process(self, original_clouds: List[o3d.geometry.PointCloud],
                               aligned_clouds: List[o3d.geometry.PointCloud],
                               fused_cloud: o3d.geometry.PointCloud,
                               window_name: str = "Fusion Process") -> None:
        """
        可视化融合过程
        
        Args:
            original_clouds: 原始点云列表
            aligned_clouds: 配准后的点云列表
            fused_cloud: 融合后的点云
            window_name: 窗口名称
        """
        print("可视化融合过程")
        
        # 创建三个视图：原始、配准后、融合后
        clouds_with_colors = []
        
        # 原始点云（红色系）
        for i, pcd in enumerate(original_clouds):
            pcd_temp = copy.deepcopy(pcd)
            if not pcd_temp.has_colors():
                pcd_temp.paint_uniform_color([1, 0.2 + i*0.1, 0.2])
            clouds_with_colors.append(pcd_temp)
        
        # 配准后的点云（绿色系）
        for i, pcd in enumerate(aligned_clouds):
            pcd_temp = copy.deepcopy(pcd)
            if not pcd_temp.has_colors():
                pcd_temp.paint_uniform_color([0.2, 1, 0.2 + i*0.1])
            clouds_with_colors.append(pcd_temp)
        
        # 融合后的点云（蓝色）
        fused_temp = copy.deepcopy(fused_cloud)
        if not fused_temp.has_colors():
            fused_temp.paint_uniform_color([0.2, 0.2, 1])
        clouds_with_colors.append(fused_temp)
        
        o3d.visualization.draw_geometries(clouds_with_colors, window_name=window_name)
    
    def visualize_with_correspondences(self, source: o3d.geometry.PointCloud,
                                      target: o3d.geometry.PointCloud,
                                      correspondences: List[Tuple[int, int]],
                                      transformation: np.ndarray,
                                      max_lines: int = 100,
                                      window_name: str = "Correspondences") -> None:
        """
        可视化对应点关系
        
        Args:
            source: 源点云
            target: 目标点云
            correspondences: 对应点索引列表
            transformation: 变换矩阵
            max_lines: 最大显示线条数
            window_name: 窗口名称
        """
        print(f"可视化对应点关系 (显示前 {min(max_lines, len(correspondences))} 条)")
        
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        
        if not source_temp.has_colors():
            source_temp.paint_uniform_color([1, 0, 0])  # Red
        if not target_temp.has_colors():
            target_temp.paint_uniform_color([0, 1, 0])  # Green
        
        # 变换源点云
        source_transformed = source_temp.transform(transformation)
        
        # 创建对应线
        source_points = np.asarray(source_transformed.points)
        target_points = np.asarray(target_temp.points)
        
        line_points = []
        line_indices = []
        
        for i, (src_idx, tgt_idx) in enumerate(correspondences[:max_lines]):
            line_points.append(source_points[src_idx])
            line_points.append(target_points[tgt_idx])
            line_indices.append([2*i, 2*i+1])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(line_indices))
        
        # 设置线条颜色（黄色）
        line_colors = np.tile([1, 1, 0], (len(line_indices), 1))
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        
        o3d.visualization.draw_geometries(
            [source_transformed, target_temp, line_set],
            window_name=window_name
        )
    
    def create_animation_frames(self, point_clouds: List[o3d.geometry.PointCloud],
                              transformations: List[np.ndarray],
                              num_frames: int = 30) -> List[o3d.geometry.PointCloud]:
        """
        创建配准过程的动画帧
        
        Args:
            point_clouds: 点云列表
            transformations: 变换矩阵列表
            num_frames: 每个变换的帧数
            
        Returns:
            动画帧列表
        """
        print("创建动画帧...")
        
        frames = []
        
        for i, (pcd, trans) in enumerate(zip(point_clouds, transformations)):
            # 为每个变换创建插值帧
            identity = np.eye(4)
            
            for frame in range(num_frames):
                # 线性插值变换矩阵
                t = frame / num_frames
                interpolated_trans = (1 - t) * identity + t * trans
                
                # 应用插值变换
                frame_pcd = copy.deepcopy(pcd)
                frame_pcd.transform(interpolated_trans)
                frames.append(frame_pcd)
        
        print(f"创建了 {len(frames)} 个动画帧")
        return frames
    
    def compare_before_after(self, before: o3d.geometry.PointCloud,
                           after: o3d.geometry.PointCloud,
                           title_before: str = "Before",
                           title_after: str = "After") -> None:
        """
        对比显示处理前后的点云
        
        Args:
            before: 处理前的点云
            after: 处理后的点云
            title_before: 处理前标题
            title_after: 处理后标题
        """
        print("对比显示")
        print(f"{title_before}: {len(before.points)} 点")
        print(f"{title_after}: {len(after.points)} 点")
        
        # 分别显示
        self.visualize_single_point_cloud(before, window_name=title_before)
        self.visualize_single_point_cloud(after, window_name=title_after)
    
    def interactive_visualization(self, point_clouds: List[o3d.geometry.PointCloud],
                                 show_normals: bool = False,
                                 show_bounding_box: bool = False) -> None:
        """
        交互式可视化
        
        Args:
            point_clouds: 点云列表
            show_normals: 是否显示法向量
            show_bounding_box: 是否显示边界框
        """
        print("启动交互式可视化")
        
        visualization_objects = []
        
        for pcd in point_clouds:
            pcd_copy = copy.deepcopy(pcd)
            
            if show_normals and not pcd_copy.has_normals():
                pcd_copy.estimate_normals()
            
            if show_bounding_box:
                bbox = pcd_copy.get_axis_aligned_bounding_box()
                bbox.color = [1, 0, 0]
                visualization_objects.append(bbox)
            
            visualization_objects.append(pcd_copy)
        
        o3d.visualization.draw_geometries(
            visualization_objects,
            mesh_show_back_face=True,
            mesh_show_wireframe=True
        )


def quick_visualize(point_clouds: List[o3d.geometry.PointCloud],
                   colors: Optional[List[List[float]]] = None) -> None:
    """
    快速可视化点云列表
    
    Args:
        point_clouds: 点云列表
        colors: 颜色列表
    """
    visualizer = PointCloudVisualizer()
    
    if colors is not None:
        for pcd, color in zip(point_clouds, colors):
            if not pcd.has_colors():
                pcd.paint_uniform_color(color)
    
    visualizer.visualize_multiple_point_clouds(point_clouds)