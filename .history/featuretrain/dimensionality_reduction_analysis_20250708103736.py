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
    
    def prepare_combined_data(self, sample_size=2000):
        """准备合并数据用于降维分析"""
        print(f"\n🔧 准备降维数据 (每组采样{sample_size}个样本)...")
        
        # 为了计算效率，对大数据集进行采样
        def safe_sample(df, n):
            return df.sample(n=min(n, len(df)), random_state=42)
        
        # 采样数据
        train_white_sample = safe_sample(self.train_white, sample_size)
        train_black_sample = safe_sample(self.train_black, sample_size)
        good250623_sample = safe_sample(self.good250623, sample_size)
        bad250623_sample = safe_sample(self.bad250623, sample_size)
        
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
        
        print(f"✅ 数据准备完成:")
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
