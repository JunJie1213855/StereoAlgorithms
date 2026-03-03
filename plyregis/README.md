# 点云配准与融合系统

一个完整的颜色点云配准和融合解决方案，支持多视角点云的自动配准、融合和重建。

## 功能特性

### 🔧 核心功能
- **多算法配准**: 支持ICP、颜色ICP、NDT、FGR等多种配准算法
- **智能融合**: 提供体素融合、统计融合、颜色感知融合等多种融合方法
- **自动预处理**: 点云滤波、下采样、法向量估计
- **质量评估**: 配准质量评估、融合效果分析、重叠区域检测
- **可视化**: 实时可视化配准过程和融合结果

### 🎯 支持的算法

#### 配准算法
- **ICP**: 经典迭代最近点算法
- **Colored ICP**: 结合颜色信息的ICP，适用于纹理丰富的场景
- **NDT**: 正态分布变换，适合粗配准
- **FGR**: 快速全局配准
- **粗到细配准**: NDT + Colored ICP的级联策略

#### 融合算法
- **简单合并**: 直接合并多个点云
- **体素融合**: 基于体素网格的融合
- **统计融合**: 基于统计信息的融合
- **颜色感知融合**: 考虑颜色信息的融合
- **MLS融合**: 移动最小二乘法平滑融合

## 安装

### 环境要求
- Python >= 3.8
- Open3D >= 0.17.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
# 处理多个点云文件
python pointcloud_fusion.py -i cloud1.ply cloud2.ply cloud3.ply -o results/

# 使用通配符
python pointcloud_fusion.py -i *.ply -o results/
```

### 高级选项

```bash
# 使用颜色ICP配准和颜色感知融合
python pointcloud_fusion.py -i *.ply -m colored_icp -f color_aware -o results/

# 指定参考点云索引
python pointcloud_fusion.py -i *.ply -r 0 -o results/

# 保存中间结果并启用可视化
python pointcloud_fusion.py -i *.ply -s --visualize -o results/

# 调整体素大小
python pointcloud_fusion.py -i *.ply -v 0.01 -o results/
```

### 参数说明

#### 必需参数
- `-i, --input`: 输入点云文件路径（支持多个文件或通配符）

#### 可选参数
- `-r, --reference`: 参考点云索引（默认：0）
- `-v, --voxel_size`: 体素大小（默认：0.02）
- `-m, --registration_method`: 配准方法
  - `coarse_to_fine`: 粗到细配准（默认）
  - `icp`: ICP配准
  - `colored_icp`: 颜色ICP配准
  - `ndt`: NDT配准
  - `multiway`: 多视角配准
- `-f, --fusion_method`: 融合方法
  - `color_aware`: 颜色感知融合（默认）
  - `voxel`: 体素融合
  - `statistical`: 统计融合
  - `simple`: 简单合并
  - `mls`: MLS融合
- `-o, --output`: 输出目录（默认：./fusion_output）
- `-s, --save_intermediate`: 保存中间结果
- `--visualize`: 启用可视化
- `--step_by_step`: 逐步显示处理过程
- `--no_downsample`: 不进行下采样
- `--no_outlier_removal`: 不进行离群点移除

## 项目结构

```
plyregis/
├── core/
│   ├── preprocessing.py      # 预处理模块
│   ├── registration.py        # 配准算法核心
│   └── fusion.py              # 融合算法
├── utils/
│   ├── visualization.py       # 可视化工具
│   └── evaluation.py          # 质量评估
├── pointcloud_fusion.py       # 主程序入口
├── requirements.txt           # 依赖管理
└── README.md                  # 项目说明
```

## 处理流程

1. **加载点云**: 支持多种点云格式（.ply, .pcd等）
2. **预处理**: 
   - 体素下采样
   - 统计滤波去噪
   - 法向量估计
3. **配准**: 
   - 粗配准（NDT/FGR）
   - 精配准（Colored ICP）
4. **融合**: 
   - 去重处理
   - 颜色融合
   - 平滑优化
5. **评估**: 
   - 配准质量评估
   - 融合效果分析
6. **输出**: 
   - 融合点云
   - 评估报告
   - 可视化结果

## 输出结果

处理完成后，在输出目录中会生成：

- `fused_pointcloud.ply`: 融合后的点云文件
- `evaluation_report.txt`: 详细的评估报告
- `aligned/`: 配准后的各个点云（如果启用`-s`选项）

## 示例

### 示例1：基本处理

```bash
python pointcloud_fusion.py -i scan1.ply scan2.ply scan3.ply -o output/
```

### 示例2：高质量处理

```bash
python pointcloud_fusion.py -i scan*.ply \
  -m colored_icp \
  -f color_aware \
  -v 0.01 \
  -s \
  --visualize \
  -o high_quality_output/
```

### 示例3：快速处理

```bash
python pointcloud_fusion.py -i scan*.ply \
  -m ndt \
  -f voxel \
  -v 0.05 \
  --no_downsample \
  -o quick_output/
```

## 注意事项

1. **输入点云格式**: 建议使用PLY格式，确保包含颜色信息
2. **内存使用**: 处理大量点云时注意内存使用，可以调大`voxel_size`
3. **配准质量**: 如果配准效果不理想，可以尝试不同的配准方法
4. **颜色信息**: 颜色ICP和颜色感知融合需要点云包含颜色信息

## 技术支持

如遇到问题，请检查：
1. 输入点云文件是否有效
2. 点云是否包含足够的重叠区域
3. 体素大小设置是否合适
4. 系统内存是否充足

## 更新日志

### v1.0.0
- 初始版本发布
- 支持多种配准和融合算法
- 完整的可视化和评估功能