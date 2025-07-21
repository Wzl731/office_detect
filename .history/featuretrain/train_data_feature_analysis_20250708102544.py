#!/usr/bin/env python3
"""
训练数据特征分析脚本
对train.csv和data文件夹中的数据进行全面的特征分析
包括：KS检验、KDE曲线图、中位数/均值差分析

分析组合：
1. train.csv黑样本 vs bad250623_features.csv
2. train.csv白样本 vs good250623_features.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp
import warnings
from pathlib import Path
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class TrainDataFeatureAnalyzer:
    def __init__(self):
        """初始化特征分析器"""
        self.output_dir = Path('featuretrain')
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据存储
        self.train_data = None
        self.train_black = None  # train.csv中的恶意样本
        self.train_white = None  # train.csv中的良性样本
        self.bad250623_data = None
        self.good250623_data = None
        
        # 分析结果存储
        self.black_analysis_results = {}
        self.white_analysis_results = {}
        
    def load_data(self):
        """加载所有数据"""
        print("📊 加载数据...")
        
        try:
            # 加载train.csv
            print("  📖 加载train.csv...")
            self.train_data = pd.read_csv('train.csv')
            print(f"    ✅ train.csv: {len(self.train_data)} 行 × {len(self.train_data.columns)} 列")
            
            # 根据训练脚本的配置分割黑白样本
            # 前2939个是良性样本，后面是恶意样本
            self.train_white = self.train_data.iloc[:2939].copy()
            self.train_black = self.train_data.iloc[2939:].copy()
            
            print(f"    📋 train白样本: {len(self.train_white)} 个")
            print(f"    📋 train黑样本: {len(self.train_black)} 个")
            
            # 加载data文件夹中的数据
            print("  📖 加载data文件夹数据...")
            self.bad250623_data = pd.read_csv('data/bad250623_features.csv')
            self.good250623_data = pd.read_csv('data/good250623_features.csv')
            
            print(f"    ✅ bad250623: {len(self.bad250623_data)} 行 × {len(self.bad250623_data.columns)} 列")
            print(f"    ✅ good250623: {len(self.good250623_data)} 行 × {len(self.good250623_data.columns)} 列")
            
            # 获取特征列（排除文件名列）
            self.feature_columns = self.train_data.columns[1:].tolist()
            print(f"    📋 特征维度: {len(self.feature_columns)}")
            
            return True
            
        except Exception as e:
            print(f"    ❌ 数据加载失败: {e}")
            return False
    
    def perform_ks_analysis(self, data1, data2, data1_name, data2_name, analysis_type):
        """执行KS检验分析"""
        print(f"\n🔍 执行KS检验: {data1_name} vs {data2_name}")
        
        results = []
        
        for feature in self.feature_columns:
            try:
                # 提取特征数据
                values1 = data1[feature].dropna()
                values2 = data2[feature].dropna()
                
                if len(values1) == 0 or len(values2) == 0:
                    continue
                
                # KS检验
                ks_stat, p_value = ks_2samp(values1, values2)
                
                # 计算统计量
                mean1, mean2 = values1.mean(), values2.mean()
                median1, median2 = values1.median(), values2.median()
                std1, std2 = values1.std(), values2.std()
                
                # 计算差异
                mean_diff = abs(mean1 - mean2)
                median_diff = abs(median1 - median2)
                
                # 效应大小 (Cohen's d)
                pooled_std = np.sqrt(((len(values1) - 1) * std1**2 + (len(values2) - 1) * std2**2) / 
                                   (len(values1) + len(values2) - 2))
                cohens_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                
                results.append({
                    'feature': feature,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    f'{data1_name}_mean': mean1,
                    f'{data2_name}_mean': mean2,
                    f'{data1_name}_median': median1,
                    f'{data2_name}_median': median2,
                    f'{data1_name}_std': std1,
                    f'{data2_name}_std': std2,
                    'mean_diff': mean_diff,
                    'median_diff': median_diff,
                    'cohens_d': cohens_d,
                    f'{data1_name}_count': len(values1),
                    f'{data2_name}_count': len(values2)
                })
                
            except Exception as e:
                print(f"    ⚠️  特征 {feature} 分析失败: {e}")
                continue
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # 按KS统计量排序
            results_df = results_df.sort_values('ks_statistic', ascending=False)
            
            # 保存结果
            output_file = self.output_dir / f'ks_analysis_{analysis_type}.csv'
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"    ✅ KS分析结果已保存: {output_file}")
            
            # 统计信息
            significant_count = results_df['significant'].sum()
            print(f"    📊 显著差异特征: {significant_count}/{len(results_df)}")
            print(f"    📊 平均KS统计量: {results_df['ks_statistic'].mean():.4f}")
            
            return results_df
        else:
            print(f"    ❌ 没有有效的分析结果")
            return None
    
    def create_kde_plots(self, data1, data2, data1_name, data2_name, analysis_type, top_n=12):
        """创建KDE曲线图"""
        print(f"\n📈 生成KDE曲线图: {data1_name} vs {data2_name}")
        
        # 获取KS统计量最大的特征
        if analysis_type == 'black_comparison':
            results_df = self.black_analysis_results
        else:
            results_df = self.white_analysis_results
            
        if results_df is None or len(results_df) == 0:
            print("    ❌ 没有分析结果，无法生成KDE图")
            return
        
        top_features = results_df.head(top_n)['feature'].tolist()
        
        # 创建子图
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'KDE分布对比: {data1_name} vs {data2_name}', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(top_features):
            row, col = i // 4, i % 4
            ax = axes[row, col]
            
            try:
                # 提取数据
                values1 = data1[feature].dropna()
                values2 = data2[feature].dropna()
                
                if len(values1) == 0 or len(values2) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(feature)
                    continue
                
                # 绘制KDE曲线
                if len(values1) > 1:
                    sns.kdeplot(data=values1, ax=ax, label=data1_name, alpha=0.7, color='red')
                if len(values2) > 1:
                    sns.kdeplot(data=values2, ax=ax, label=data2_name, alpha=0.7, color='blue')
                
                # 添加均值线
                ax.axvline(values1.mean(), color='red', linestyle='--', alpha=0.8, linewidth=1)
                ax.axvline(values2.mean(), color='blue', linestyle='--', alpha=0.8, linewidth=1)
                
                # 获取KS统计量
                ks_stat = results_df[results_df['feature'] == feature]['ks_statistic'].iloc[0]
                p_val = results_df[results_df['feature'] == feature]['p_value'].iloc[0]
                
                ax.set_title(f'{feature}\nKS={ks_stat:.3f}, p={p_val:.3e}', fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feature)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = self.output_dir / f'kde_plots_{analysis_type}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ KDE图已保存: {output_file}")
