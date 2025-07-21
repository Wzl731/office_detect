#!/usr/bin/env python3
"""
降维可视化分析脚本
使用PCA和t-SNE对train.csv和data文件夹数据进行降维可视化
彩色标注不同数据集，直观展示数据分布和聚类情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class DimensionalityReductionAnalyzer:
    def __init__(self):
        """初始化降维分析器"""
        self.output_dir = Path('featuretrain')
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据存储
        self.train_white = None
        self.train_black = None
        self.good250623 = None
        self.bad250623 = None
        self.feature_columns = None
        
        # 降维结果存储
        self.pca_results = {}
        self.tsne_results = {}
        
    def load_data(self):
        """加载所有数据"""
        print("📊 加载数据进行降维分析...")
        
        try:
            # 加载train.csv
            train_data = pd.read_csv('train.csv')
            self.train_white = train_data.iloc[:2939].copy()
            self.train_black = train_data.iloc[2939:].copy()
            
            # 加载data文件夹数据
            self.good250623 = pd.read_csv('data/good250623_features.csv')
            self.bad250623 = pd.read_csv('data/bad250623_features.csv')
            
            # 获取特征列
            self.feature_columns = train_data.columns[1:].tolist()
            
            print(f"✅ 数据加载成功:")
            print(f"   - train白样本: {len(self.train_white)}")
            print(f"   - train黑样本: {len(self.train_black)}")
            print(f"   - good250623: {len(self.good250623)}")
            print(f"   - bad250623: {len(self.bad250623)}")
            print(f"   - 特征维度: {len(self.feature_columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def prepare_combined_data(self):
        """准备合并数据用于降维分析"""
        print(f"\n🔧 准备降维数据 (使用全部数据)...")

        # 使用全部数据，不进行采样
        train_white_sample = self.train_white
        train_black_sample = self.train_black
        good250623_sample = self.good250623
        bad250623_sample = self.bad250623
        
        # 提取特征数据
        train_white_features = train_white_sample[self.feature_columns].values
        train_black_features = train_black_sample[self.feature_columns].values
        good250623_features = good250623_sample[self.feature_columns].values
        bad250623_features = bad250623_sample[self.feature_columns].values
        
        # 合并所有数据
        all_features = np.vstack([
            train_white_features,
            train_black_features,
            good250623_features,
            bad250623_features
        ])
        
        # 创建标签
        labels = (
            ['train_white'] * len(train_white_features) +
            ['train_black'] * len(train_black_features) +
            ['good250623'] * len(good250623_features) +
            ['bad250623'] * len(bad250623_features)
        )
        
        # 创建颜色映射
        color_map = {
            'train_white': '#1f77b4',    # 蓝色
            'train_black': '#ff7f0e',    # 橙色
            'good250623': '#2ca02c',     # 绿色
            'bad250623': '#d62728'       # 红色
        }
        
        colors = [color_map[label] for label in labels]
        
        print(f"✅ 数据准备完成 (全部数据):")
        print(f"   - 总样本数: {len(all_features)}")
        print(f"   - train_white: {len(train_white_features)}")
        print(f"   - train_black: {len(train_black_features)}")
        print(f"   - good250623: {len(good250623_features)}")
        print(f"   - bad250623: {len(bad250623_features)}")
        
        return all_features, labels, colors, color_map
    
    def perform_pca_analysis(self, features, labels, colors, color_map):
        """执行PCA降维分析"""
        print("\n🔍 执行PCA降维分析...")
        
        # 标准化数据
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # PCA降维到2D
        pca_2d = PCA(n_components=2, random_state=42)
        pca_2d_result = pca_2d.fit_transform(features_scaled)
        
        # PCA降维到3D
        pca_3d = PCA(n_components=3, random_state=42)
        pca_3d_result = pca_3d.fit_transform(features_scaled)
        
        # 保存结果
        self.pca_results = {
            '2d': pca_2d_result,
            '3d': pca_3d_result,
            'explained_variance_2d': pca_2d.explained_variance_ratio_,
            'explained_variance_3d': pca_3d.explained_variance_ratio_,
            'labels': labels,
            'colors': colors
        }
        
        print(f"✅ PCA分析完成:")
        print(f"   - PC1解释方差: {pca_2d.explained_variance_ratio_[0]:.3f}")
        print(f"   - PC2解释方差: {pca_2d.explained_variance_ratio_[1]:.3f}")
        print(f"   - 总解释方差: {sum(pca_2d.explained_variance_ratio_):.3f}")
        
        return pca_2d_result, pca_3d_result
    
    def perform_tsne_analysis(self, features, labels, colors):
        """执行t-SNE降维分析"""
        print("\n🔍 执行t-SNE降维分析...")
        
        # 标准化数据
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # t-SNE降维到2D
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_2d_result = tsne_2d.fit_transform(features_scaled)
        
        # 保存结果
        self.tsne_results = {
            '2d': tsne_2d_result,
            'labels': labels,
            'colors': colors
        }
        
        print(f"✅ t-SNE分析完成")
        
        return tsne_2d_result
    
    def create_pca_visualizations(self, color_map):
        """创建PCA可视化图表"""
        print("\n📈 生成PCA可视化图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. PCA 2D散点图
        pca_2d = self.pca_results['2d']
        labels = self.pca_results['labels']
        
        for dataset in color_map.keys():
            mask = np.array(labels) == dataset
            axes[0,0].scatter(pca_2d[mask, 0], pca_2d[mask, 1], 
                            c=color_map[dataset], label=dataset, alpha=0.6, s=20)
        
        axes[0,0].set_xlabel(f'PC1 ({self.pca_results["explained_variance_2d"][0]:.1%} variance)')
        axes[0,0].set_ylabel(f'PC2 ({self.pca_results["explained_variance_2d"][1]:.1%} variance)')
        axes[0,0].set_title('PCA 2D投影 - 数据集分布对比')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. PCA解释方差图
        pca_3d_var = self.pca_results['explained_variance_3d']
        cumsum_var = np.cumsum(pca_3d_var)
        
        axes[0,1].bar(range(1, len(pca_3d_var)+1), pca_3d_var, alpha=0.7, color='skyblue', label='单独解释方差')
        axes[0,1].plot(range(1, len(cumsum_var)+1), cumsum_var, 'ro-', label='累积解释方差')
        axes[0,1].set_xlabel('主成分')
        axes[0,1].set_ylabel('解释方差比例')
        axes[0,1].set_title('PCA解释方差分析')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 白样本对比 (train_white vs good250623)
        white_mask = np.array(labels) == 'train_white'
        good_mask = np.array(labels) == 'good250623'
        
        axes[1,0].scatter(pca_2d[white_mask, 0], pca_2d[white_mask, 1], 
                         c='blue', label='train_white', alpha=0.6, s=20)
        axes[1,0].scatter(pca_2d[good_mask, 0], pca_2d[good_mask, 1], 
                         c='green', label='good250623', alpha=0.6, s=20)
        axes[1,0].set_xlabel(f'PC1 ({self.pca_results["explained_variance_2d"][0]:.1%} variance)')
        axes[1,0].set_ylabel(f'PC2 ({self.pca_results["explained_variance_2d"][1]:.1%} variance)')
        axes[1,0].set_title('PCA - 白样本对比 (特征偏移可视化)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 黑样本对比 (train_black vs bad250623)
        black_mask = np.array(labels) == 'train_black'
        bad_mask = np.array(labels) == 'bad250623'
        
        axes[1,1].scatter(pca_2d[black_mask, 0], pca_2d[black_mask, 1], 
                         c='orange', label='train_black', alpha=0.6, s=20)
        axes[1,1].scatter(pca_2d[bad_mask, 0], pca_2d[bad_mask, 1], 
                         c='red', label='bad250623', alpha=0.6, s=20)
        axes[1,1].set_xlabel(f'PC1 ({self.pca_results["explained_variance_2d"][0]:.1%} variance)')
        axes[1,1].set_ylabel(f'PC2 ({self.pca_results["explained_variance_2d"][1]:.1%} variance)')
        axes[1,1].set_title('PCA - 黑样本对比 (特征偏移可视化)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = self.output_dir / 'pca_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ PCA图已保存: {output_file}")
    
    def create_tsne_visualizations(self, color_map):
        """创建t-SNE可视化图表"""
        print("\n📈 生成t-SNE可视化图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        tsne_2d = self.tsne_results['2d']
        labels = self.tsne_results['labels']
        
        # 1. t-SNE 2D散点图 - 全部数据
        for dataset in color_map.keys():
            mask = np.array(labels) == dataset
            axes[0,0].scatter(tsne_2d[mask, 0], tsne_2d[mask, 1], 
                            c=color_map[dataset], label=dataset, alpha=0.6, s=20)
        
        axes[0,0].set_xlabel('t-SNE 1')
        axes[0,0].set_ylabel('t-SNE 2')
        axes[0,0].set_title('t-SNE 2D投影 - 数据集聚类分析')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. t-SNE密度图
        axes[0,1].hexbin(tsne_2d[:, 0], tsne_2d[:, 1], gridsize=30, cmap='Blues', alpha=0.7)
        axes[0,1].set_xlabel('t-SNE 1')
        axes[0,1].set_ylabel('t-SNE 2')
        axes[0,1].set_title('t-SNE 密度分布')
        
        # 3. 白样本对比
        white_mask = np.array(labels) == 'train_white'
        good_mask = np.array(labels) == 'good250623'
        
        axes[1,0].scatter(tsne_2d[white_mask, 0], tsne_2d[white_mask, 1], 
                         c='blue', label='train_white', alpha=0.6, s=20)
        axes[1,0].scatter(tsne_2d[good_mask, 0], tsne_2d[good_mask, 1], 
                         c='green', label='good250623', alpha=0.6, s=20)
        axes[1,0].set_xlabel('t-SNE 1')
        axes[1,0].set_ylabel('t-SNE 2')
        axes[1,0].set_title('t-SNE - 白样本聚类对比')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 黑样本对比
        black_mask = np.array(labels) == 'train_black'
        bad_mask = np.array(labels) == 'bad250623'
        
        axes[1,1].scatter(tsne_2d[black_mask, 0], tsne_2d[black_mask, 1], 
                         c='orange', label='train_black', alpha=0.6, s=20)
        axes[1,1].scatter(tsne_2d[bad_mask, 0], tsne_2d[bad_mask, 1], 
                         c='red', label='bad250623', alpha=0.6, s=20)
        axes[1,1].set_xlabel('t-SNE 1')
        axes[1,1].set_ylabel('t-SNE 2')
        axes[1,1].set_title('t-SNE - 黑样本聚类对比')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = self.output_dir / 'tsne_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ t-SNE图已保存: {output_file}")

    def create_3d_pca_visualization(self, color_map):
        """创建3D PCA可视化"""
        print("\n📈 生成3D PCA可视化...")

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(15, 5))

        pca_3d = self.pca_results['3d']
        labels = self.pca_results['labels']

        # 1. 全部数据3D视图
        ax1 = fig.add_subplot(131, projection='3d')
        for dataset in color_map.keys():
            mask = np.array(labels) == dataset
            ax1.scatter(pca_3d[mask, 0], pca_3d[mask, 1], pca_3d[mask, 2],
                       c=color_map[dataset], label=dataset, alpha=0.6, s=15)

        ax1.set_xlabel(f'PC1 ({self.pca_results["explained_variance_3d"][0]:.1%})')
        ax1.set_ylabel(f'PC2 ({self.pca_results["explained_variance_3d"][1]:.1%})')
        ax1.set_zlabel(f'PC3 ({self.pca_results["explained_variance_3d"][2]:.1%})')
        ax1.set_title('3D PCA - 全部数据')
        ax1.legend()

        # 2. 白样本3D对比
        ax2 = fig.add_subplot(132, projection='3d')
        white_mask = np.array(labels) == 'train_white'
        good_mask = np.array(labels) == 'good250623'

        ax2.scatter(pca_3d[white_mask, 0], pca_3d[white_mask, 1], pca_3d[white_mask, 2],
                   c='blue', label='train_white', alpha=0.6, s=15)
        ax2.scatter(pca_3d[good_mask, 0], pca_3d[good_mask, 1], pca_3d[good_mask, 2],
                   c='green', label='good250623', alpha=0.6, s=15)

        ax2.set_xlabel(f'PC1 ({self.pca_results["explained_variance_3d"][0]:.1%})')
        ax2.set_ylabel(f'PC2 ({self.pca_results["explained_variance_3d"][1]:.1%})')
        ax2.set_zlabel(f'PC3 ({self.pca_results["explained_variance_3d"][2]:.1%})')
        ax2.set_title('3D PCA - 白样本对比')
        ax2.legend()

        # 3. 黑样本3D对比
        ax3 = fig.add_subplot(133, projection='3d')
        black_mask = np.array(labels) == 'train_black'
        bad_mask = np.array(labels) == 'bad250623'

        ax3.scatter(pca_3d[black_mask, 0], pca_3d[black_mask, 1], pca_3d[black_mask, 2],
                   c='orange', label='train_black', alpha=0.6, s=15)
        ax3.scatter(pca_3d[bad_mask, 0], pca_3d[bad_mask, 1], pca_3d[bad_mask, 2],
                   c='red', label='bad250623', alpha=0.6, s=15)

        ax3.set_xlabel(f'PC1 ({self.pca_results["explained_variance_3d"][0]:.1%})')
        ax3.set_ylabel(f'PC2 ({self.pca_results["explained_variance_3d"][1]:.1%})')
        ax3.set_zlabel(f'PC3 ({self.pca_results["explained_variance_3d"][2]:.1%})')
        ax3.set_title('3D PCA - 黑样本对比')
        ax3.legend()

        plt.tight_layout()

        # 保存图片
        output_file = self.output_dir / 'pca_3d_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 3D PCA图已保存: {output_file}")

    def calculate_separation_metrics(self):
        """计算数据集分离度指标"""
        print("\n📊 计算数据集分离度指标...")

        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import pdist, squareform

        # PCA结果分析
        pca_2d = self.pca_results['2d']
        labels = self.pca_results['labels']

        # 转换标签为数值
        label_map = {'train_white': 0, 'train_black': 1, 'good250623': 2, 'bad250623': 3}
        numeric_labels = [label_map[label] for label in labels]

        # 计算轮廓系数
        pca_silhouette = silhouette_score(pca_2d, numeric_labels)

        # t-SNE结果分析
        tsne_2d = self.tsne_results['2d']
        tsne_silhouette = silhouette_score(tsne_2d, numeric_labels)

        # 计算组内和组间距离
        def calculate_group_distances(data, labels):
            unique_labels = list(set(labels))
            group_stats = {}

            for label in unique_labels:
                mask = np.array(labels) == label
                group_data = data[mask]

                if len(group_data) > 1:
                    # 组内平均距离
                    intra_distances = pdist(group_data)
                    intra_mean = np.mean(intra_distances)

                    group_stats[label] = {
                        'intra_distance': intra_mean,
                        'center': np.mean(group_data, axis=0),
                        'size': len(group_data)
                    }

            # 计算组间距离
            inter_distances = {}
            centers = {label: stats['center'] for label, stats in group_stats.items()}

            for i, label1 in enumerate(unique_labels):
                for label2 in unique_labels[i+1:]:
                    if label1 in centers and label2 in centers:
                        dist = np.linalg.norm(centers[label1] - centers[label2])
                        inter_distances[f'{label1}_vs_{label2}'] = dist

            return group_stats, inter_distances

        pca_group_stats, pca_inter_dist = calculate_group_distances(pca_2d, labels)
        tsne_group_stats, tsne_inter_dist = calculate_group_distances(tsne_2d, labels)

        # 保存分离度分析结果
        separation_file = self.output_dir / 'separation_analysis.txt'
        with open(separation_file, 'w', encoding='utf-8') as f:
            f.write("数据集分离度分析报告\n")
            f.write("=" * 50 + "\n\n")

            f.write("轮廓系数 (Silhouette Score):\n")
            f.write(f"  PCA: {pca_silhouette:.4f}\n")
            f.write(f"  t-SNE: {tsne_silhouette:.4f}\n\n")

            f.write("PCA组间距离:\n")
            for pair, dist in pca_inter_dist.items():
                f.write(f"  {pair}: {dist:.4f}\n")

            f.write("\nt-SNE组间距离:\n")
            for pair, dist in tsne_inter_dist.items():
                f.write(f"  {pair}: {dist:.4f}\n")

            f.write("\n关键对比:\n")
            white_comparison = pca_inter_dist.get('train_white_vs_good250623', 0)
            black_comparison = pca_inter_dist.get('train_black_vs_bad250623', 0)
            f.write(f"  白样本分离度 (PCA): {white_comparison:.4f}\n")
            f.write(f"  黑样本分离度 (PCA): {black_comparison:.4f}\n")

        print(f"✅ 分离度分析已保存: {separation_file}")
        print(f"   - PCA轮廓系数: {pca_silhouette:.4f}")
        print(f"   - t-SNE轮廓系数: {tsne_silhouette:.4f}")

        return pca_silhouette, tsne_silhouette

    def generate_dimensionality_report(self):
        """生成降维分析综合报告"""
        print("\n📄 生成降维分析报告...")

        report_file = self.output_dir / 'dimensionality_reduction_report.md'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 降维可视化分析报告\n\n")

            f.write("## 分析概述\n\n")
            f.write("本报告使用PCA和t-SNE降维技术对train.csv和data文件夹数据进行可视化分析，")
            f.write("通过彩色标注不同数据集，直观展示数据分布、聚类情况和特征偏移。\n\n")

            f.write("## 数据集标注\n\n")
            f.write("- 🔵 **train_white**: train.csv中的良性样本 (蓝色)\n")
            f.write("- 🟠 **train_black**: train.csv中的恶意样本 (橙色)\n")
            f.write("- 🟢 **good250623**: good250623_features.csv良性样本 (绿色)\n")
            f.write("- 🔴 **bad250623**: bad250623_features.csv恶意样本 (红色)\n\n")

            f.write("## PCA分析结果\n\n")
            if self.pca_results:
                var_2d = self.pca_results['explained_variance_2d']
                var_3d = self.pca_results['explained_variance_3d']
                f.write(f"- **PC1解释方差**: {var_2d[0]:.1%}\n")
                f.write(f"- **PC2解释方差**: {var_2d[1]:.1%}\n")
                f.write(f"- **前2个主成分总解释方差**: {sum(var_2d):.1%}\n")
                f.write(f"- **前3个主成分总解释方差**: {sum(var_3d):.1%}\n\n")

            f.write("## 生成的可视化文件\n\n")
            f.write("### PCA可视化\n")
            f.write("- `pca_analysis.png`: 2D PCA分析图 (4个子图)\n")
            f.write("  - 全部数据集2D投影\n")
            f.write("  - PCA解释方差分析\n")
            f.write("  - 白样本对比 (特征偏移可视化)\n")
            f.write("  - 黑样本对比 (特征偏移可视化)\n")
            f.write("- `pca_3d_analysis.png`: 3D PCA分析图\n\n")

            f.write("### t-SNE可视化\n")
            f.write("- `tsne_analysis.png`: t-SNE聚类分析图 (4个子图)\n")
            f.write("  - 全部数据集聚类分析\n")
            f.write("  - t-SNE密度分布\n")
            f.write("  - 白样本聚类对比\n")
            f.write("  - 黑样本聚类对比\n\n")

            f.write("### 分析数据\n")
            f.write("- `separation_analysis.txt`: 数据集分离度分析\n")
            f.write("- `dimensionality_reduction_report.md`: 本报告\n\n")

            f.write("## 如何解读图表\n\n")
            f.write("### PCA图解读\n")
            f.write("- **聚集程度**: 同色点聚集表示数据一致性好\n")
            f.write("- **分离程度**: 不同色点分离表示数据集差异大\n")
            f.write("- **重叠区域**: 重叠表示特征相似，分离表示特征偏移\n\n")

            f.write("### t-SNE图解读\n")
            f.write("- **聚类结构**: t-SNE更好地保持局部结构\n")
            f.write("- **密度分布**: 显示数据的聚集模式\n")
            f.write("- **异常点**: 远离主要聚类的点可能是异常样本\n\n")

            f.write("## 特征偏移诊断\n\n")
            f.write("通过降维可视化可以直观地看到:\n")
            f.write("1. **白样本偏移**: train_white和good250623的分布差异\n")
            f.write("2. **黑样本偏移**: train_black和bad250623的分布差异\n")
            f.write("3. **聚类质量**: 同类样本的聚集程度\n")
            f.write("4. **数据质量**: 异常点和噪声的分布\n")

        print(f"✅ 降维分析报告已保存: {report_file}")

    def run_analysis(self):
        """运行完整的降维分析"""
        print("🎯 降维可视化分析 (使用全部数据)")
        print("=" * 60)

        # 1. 加载数据
        if not self.load_data():
            return False

        # 2. 准备合并数据
        features, labels, colors, color_map = self.prepare_combined_data()

        # 3. PCA分析
        pca_2d, pca_3d = self.perform_pca_analysis(features, labels, colors, color_map)

        # 4. t-SNE分析
        tsne_2d = self.perform_tsne_analysis(features, labels, colors)

        # 5. 创建可视化
        self.create_pca_visualizations(color_map)
        self.create_3d_pca_visualization(color_map)
        self.create_tsne_visualizations(color_map)

        # 6. 计算分离度指标
        self.calculate_separation_metrics()

        # 7. 生成报告
        self.generate_dimensionality_report()

        print(f"\n🎉 降维分析完成！")
        print(f"📁 所有结果已保存到 featuretrain/ 文件夹")
        print(f"📊 生成的可视化文件:")
        print(f"   - pca_analysis.png (2D PCA分析)")
        print(f"   - pca_3d_analysis.png (3D PCA分析)")
        print(f"   - tsne_analysis.png (t-SNE聚类分析)")

        return True

def main():
    """主函数"""
    analyzer = DimensionalityReductionAnalyzer()
    success = analyzer.run_analysis()  # 使用全部数据

    if success:
        print("\n✅ 降维可视化分析完成！")
        print("🔍 请查看生成的PNG文件来观察数据分布和特征偏移情况")
    else:
        print("\n❌ 降维分析失败！")

if __name__ == "__main__":
    main()
