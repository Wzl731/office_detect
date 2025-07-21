#!/usr/bin/env python3
"""
第一步：数据探索和基础统计分析
分析黑、白、误报样本的基本特征分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataExplorer:
    def __init__(self):
        self.white_data = None
        self.black_data = None
        self.misc_data = None
        self.feature_names = None
        
    def load_data(self):
        """加载所有数据"""
        print("📊 加载数据...")
        
        self.white_data = pd.read_excel('../data/good250623_features.xlsx')
        self.black_data = pd.read_excel('../data/bad250623_features.xlsx')
        self.misc_data = pd.read_excel('../data/good2bad_features.xlsx')
        
        self.feature_names = [col for col in self.white_data.columns if col != 'FILENAME']
        
        print(f"✅ 白样本: {self.white_data.shape}")
        print(f"✅ 黑样本: {self.black_data.shape}")
        print(f"✅ 误报样本: {self.misc_data.shape}")
        print(f"✅ 特征数量: {len(self.feature_names)}")
        
    def basic_statistics(self):
        """基础统计分析"""
        print("\n📈 基础统计分析...")
        
        stats_summary = []
        
        for feature in self.feature_names:
            white_vals = self.white_data[feature].fillna(0)
            black_vals = self.black_data[feature].fillna(0)
            misc_vals = self.misc_data[feature].fillna(0)
            
            stats_summary.append({
                'feature': feature,
                'white_mean': white_vals.mean(),
                'white_std': white_vals.std(),
                'white_median': white_vals.median(),
                'black_mean': black_vals.mean(),
                'black_std': black_vals.std(),
                'black_median': black_vals.median(),
                'misc_mean': misc_vals.mean(),
                'misc_std': misc_vals.std(),
                'misc_median': misc_vals.median(),
                'white_nonzero_ratio': (white_vals > 0).mean(),
                'black_nonzero_ratio': (black_vals > 0).mean(),
                'misc_nonzero_ratio': (misc_vals > 0).mean(),
            })
        
        stats_df = pd.DataFrame(stats_summary)
        stats_df.to_excel('data_statistics_summary.xlsx', index=False)
        print("✅ 基础统计已保存到: data_statistics_summary.xlsx")
        
        return stats_df
    
    def feature_distribution_analysis(self):
        """特征分布分析"""
        print("\n📊 特征分布分析...")
        
        # 选择最有区分度的特征进行可视化
        discriminative_features = []
        
        for feature in self.feature_names:
            white_vals = self.white_data[feature].fillna(0)
            black_vals = self.black_data[feature].fillna(0)
            misc_vals = self.misc_data[feature].fillna(0)
            
            # 计算方差比和均值差异
            if white_vals.std() > 0 and black_vals.std() > 0:
                mean_diff = abs(white_vals.mean() - black_vals.mean())
                pooled_std = np.sqrt((white_vals.var() + black_vals.var()) / 2)
                effect_size = mean_diff / (pooled_std + 1e-8)
                
                discriminative_features.append({
                    'feature': feature,
                    'effect_size': effect_size,
                    'misc_closer_to_black': abs(misc_vals.mean() - black_vals.mean()) < abs(misc_vals.mean() - white_vals.mean())
                })
        
        # 排序并选择前12个最有区分度的特征
        discriminative_df = pd.DataFrame(discriminative_features)
        discriminative_df = discriminative_df.sort_values('effect_size', ascending=False)
        top_features = discriminative_df.head(12)['feature'].tolist()
        
        # 绘制分布图
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            ax = axes[i]
            
            white_vals = self.white_data[feature].fillna(0)
            black_vals = self.black_data[feature].fillna(0)
            misc_vals = self.misc_data[feature].fillna(0)
            
            # 绘制直方图 - 使用频次而不是密度，便于理解
            # 为了便于比较，对样本数量进行归一化显示
            white_weights = np.ones_like(white_vals) / len(white_vals) * 100  # 转换为百分比
            black_weights = np.ones_like(black_vals) / len(black_vals) * 100
            misc_weights = np.ones_like(misc_vals) / len(misc_vals) * 100

            ax.hist(white_vals, bins=30, alpha=0.6, label=f'白样本(n={len(white_vals)})',
                   color='blue', weights=white_weights)
            ax.hist(black_vals, bins=30, alpha=0.6, label=f'黑样本(n={len(black_vals)})',
                   color='red', weights=black_weights)
            ax.hist(misc_vals, bins=20, alpha=0.8, label=f'误报样本(n={len(misc_vals)})',
                   color='orange', weights=misc_weights)

            ax.set_title(f'{feature}')
            ax.set_xlabel('特征值')
            ax.set_ylabel('百分比 (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('关键特征分布对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_distributions_detailed.png', dpi=300, bbox_inches='tight')
        print("✅ 特征分布图已保存到: feature_distributions_detailed.png")
        plt.show()
        
        return discriminative_df
    
    def correlation_analysis(self):
        """特征相关性分析"""
        print("\n🔗 特征相关性分析...")
        
        # 合并所有数据进行相关性分析
        all_data = pd.concat([
            self.white_data[self.feature_names].assign(sample_type='white'),
            self.black_data[self.feature_names].assign(sample_type='black'),
            self.misc_data[self.feature_names].assign(sample_type='misc')
        ], ignore_index=True).fillna(0)
        
        # 计算特征相关性矩阵
        feature_corr = all_data[self.feature_names].corr()
        
        # 找出高相关性的特征对
        high_corr_pairs = []
        for i in range(len(feature_corr.columns)):
            for j in range(i+1, len(feature_corr.columns)):
                corr_val = feature_corr.iloc[i, j]
                if abs(corr_val) > 0.7:  # 高相关性阈值
                    high_corr_pairs.append({
                        'feature1': feature_corr.columns[i],
                        'feature2': feature_corr.columns[j],
                        'correlation': corr_val
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
            high_corr_df.to_excel('high_correlation_features.xlsx', index=False)
            print(f"✅ 发现 {len(high_corr_pairs)} 对高相关性特征，已保存到: high_correlation_features.xlsx")
            
            # 显示前10对
            print("\n🔗 高相关性特征对 (|r| > 0.7):")
            for _, row in high_corr_df.head(10).iterrows():
                print(f"  {row['feature1']} <-> {row['feature2']}: r = {row['correlation']:.3f}")
        else:
            print("✅ 未发现高相关性特征对")
        
        return feature_corr, high_corr_pairs
    
    def missing_value_analysis(self):
        """缺失值分析"""
        print("\n❓ 缺失值分析...")
        
        missing_summary = []
        
        for feature in self.feature_names:
            white_missing = self.white_data[feature].isna().sum()
            black_missing = self.black_data[feature].isna().sum()
            misc_missing = self.misc_data[feature].isna().sum()
            
            if white_missing > 0 or black_missing > 0 or misc_missing > 0:
                missing_summary.append({
                    'feature': feature,
                    'white_missing': white_missing,
                    'white_missing_pct': white_missing / len(self.white_data) * 100,
                    'black_missing': black_missing,
                    'black_missing_pct': black_missing / len(self.black_data) * 100,
                    'misc_missing': misc_missing,
                    'misc_missing_pct': misc_missing / len(self.misc_data) * 100,
                })
        
        if missing_summary:
            missing_df = pd.DataFrame(missing_summary)
            missing_df.to_excel('missing_values_analysis.xlsx', index=False)
            print(f"✅ 发现 {len(missing_summary)} 个特征有缺失值，已保存到: missing_values_analysis.xlsx")
        else:
            print("✅ 所有特征都没有缺失值")
        
        return missing_summary

def main():
    """主函数"""
    print("🎯 数据探索和基础统计分析")
    print("=" * 60)
    
    explorer = DataExplorer()
    
    # 1. 加载数据
    explorer.load_data()
    
    # 2. 基础统计分析
    stats_df = explorer.basic_statistics()
    
    # 3. 特征分布分析
    discriminative_df = explorer.feature_distribution_analysis()
    
    # 4. 相关性分析
    feature_corr, high_corr_pairs = explorer.correlation_analysis()
    
    # 5. 缺失值分析
    missing_summary = explorer.missing_value_analysis()
    
    print("\n🎉 数据探索完成!")
    print("📄 生成的文件:")
    print("  - data_statistics_summary.xlsx: 基础统计摘要")
    print("  - feature_distributions_detailed.png: 特征分布图")
    print("  - high_correlation_features.xlsx: 高相关性特征")
    print("  - missing_values_analysis.xlsx: 缺失值分析")

if __name__ == "__main__":
    main()
