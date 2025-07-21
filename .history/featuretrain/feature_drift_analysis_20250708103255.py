#!/usr/bin/env python3
"""
特征偏移深度分析脚本
专门分析train.csv和data文件夹数据之间的特征偏移问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FeatureDriftAnalyzer:
    def __init__(self):
        self.output_dir = Path('featuretrain')
        
    def load_data(self):
        """加载数据"""
        print("📊 加载数据进行偏移分析...")
        
        # 加载数据
        self.train_data = pd.read_csv('train.csv')
        self.train_white = self.train_data.iloc[:2939].copy()
        self.train_black = self.train_data.iloc[2939:].copy()
        self.good250623 = pd.read_csv('data/good250623_features.csv')
        self.bad250623 = pd.read_csv('data/bad250623_features.csv')
        
        self.feature_columns = self.train_data.columns[1:].tolist()
        
        print(f"✅ 数据加载完成")
        print(f"   - train白样本: {len(self.train_white)}")
        print(f"   - good250623: {len(self.good250623)}")
        print(f"   - train黑样本: {len(self.train_black)}")
        print(f"   - bad250623: {len(self.bad250623)}")
    
    def analyze_drift_severity(self):
        """分析偏移严重程度"""
        print("\n🔍 分析特征偏移严重程度...")
        
        # 读取之前的KS分析结果
        white_ks = pd.read_csv('featuretrain/ks_analysis_white_comparison.csv')
        black_ks = pd.read_csv('featuretrain/ks_analysis_black_comparison.csv')
        
        # 偏移严重程度分类
        def classify_drift(ks_stat):
            if ks_stat < 0.1:
                return "轻微偏移"
            elif ks_stat < 0.2:
                return "中等偏移"
            elif ks_stat < 0.4:
                return "严重偏移"
            else:
                return "极严重偏移"
        
        white_ks['drift_level'] = white_ks['ks_statistic'].apply(classify_drift)
        black_ks['drift_level'] = black_ks['ks_statistic'].apply(classify_drift)
        
        # 统计偏移程度
        print("\n📊 白样本偏移程度统计:")
        white_drift_stats = white_ks['drift_level'].value_counts()
        for level, count in white_drift_stats.items():
            print(f"   {level}: {count} 个特征 ({count/len(white_ks)*100:.1f}%)")
        
        print("\n📊 黑样本偏移程度统计:")
        black_drift_stats = black_ks['drift_level'].value_counts()
        for level, count in black_drift_stats.items():
            print(f"   {level}: {count} 个特征 ({count/len(black_ks)*100:.1f}%)")
        
        # 保存偏移分析结果
        drift_summary = {
            'white_samples': {
                'total_features': len(white_ks),
                'significant_drift': white_ks['significant'].sum(),
                'severe_drift': len(white_ks[white_ks['ks_statistic'] >= 0.2]),
                'extreme_drift': len(white_ks[white_ks['ks_statistic'] >= 0.4]),
                'avg_ks': white_ks['ks_statistic'].mean(),
                'max_ks': white_ks['ks_statistic'].max()
            },
            'black_samples': {
                'total_features': len(black_ks),
                'significant_drift': black_ks['significant'].sum(),
                'severe_drift': len(black_ks[black_ks['ks_statistic'] >= 0.2]),
                'extreme_drift': len(black_ks[black_ks['ks_statistic'] >= 0.4]),
                'avg_ks': black_ks['ks_statistic'].mean(),
                'max_ks': black_ks['ks_statistic'].max()
            }
        }
        
        return drift_summary, white_ks, black_ks
    
    def create_drift_visualization(self, white_ks, black_ks):
        """创建偏移可视化"""
        print("\n📈 生成偏移可视化图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. KS统计量分布对比
        axes[0,0].hist(white_ks['ks_statistic'], bins=30, alpha=0.7, label='白样本偏移', color='blue')
        axes[0,0].hist(black_ks['ks_statistic'], bins=30, alpha=0.7, label='黑样本偏移', color='red')
        axes[0,0].set_xlabel('KS统计量')
        axes[0,0].set_ylabel('特征数量')
        axes[0,0].set_title('特征偏移分布对比')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 偏移严重程度对比
        white_levels = white_ks['drift_level'].value_counts()
        black_levels = black_ks['drift_level'].value_counts()
        
        levels = ['轻微偏移', '中等偏移', '严重偏移', '极严重偏移']
        white_counts = [white_levels.get(level, 0) for level in levels]
        black_counts = [black_levels.get(level, 0) for level in levels]
        
        x = np.arange(len(levels))
        width = 0.35
        
        axes[0,1].bar(x - width/2, white_counts, width, label='白样本', color='blue', alpha=0.7)
        axes[0,1].bar(x + width/2, black_counts, width, label='黑样本', color='red', alpha=0.7)
        axes[0,1].set_xlabel('偏移程度')
        axes[0,1].set_ylabel('特征数量')
        axes[0,1].set_title('偏移严重程度对比')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(levels, rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 前20个偏移最严重的白样本特征
        top_white_drift = white_ks.head(20)
        axes[1,0].barh(range(len(top_white_drift)), top_white_drift['ks_statistic'], color='blue', alpha=0.7)
        axes[1,0].set_yticks(range(len(top_white_drift)))
        axes[1,0].set_yticklabels(top_white_drift['feature'], fontsize=8)
        axes[1,0].set_xlabel('KS统计量')
        axes[1,0].set_title('白样本偏移最严重的20个特征')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 效应大小对比
        axes[1,1].scatter(white_ks['ks_statistic'], white_ks['cohens_d'], alpha=0.6, label='白样本', color='blue')
        axes[1,1].scatter(black_ks['ks_statistic'], black_ks['cohens_d'], alpha=0.6, label='黑样本', color='red')
        axes[1,1].set_xlabel('KS统计量')
        axes[1,1].set_ylabel('Cohen\'s d (效应大小)')
        axes[1,1].set_title('偏移程度 vs 效应大小')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'feature_drift_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 偏移分析图已保存: {output_file}")
    
    def generate_drift_report(self, drift_summary):
        """生成偏移分析报告"""
        print("\n📄 生成偏移分析报告...")
        
        report_file = self.output_dir / 'feature_drift_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 特征偏移分析报告\n\n")
            
            f.write("## 🚨 偏移检测结果\n\n")
            f.write("### 白样本偏移情况 (train白样本 vs good250623)\n")
            white = drift_summary['white_samples']
            f.write(f"- **总特征数**: {white['total_features']}\n")
            f.write(f"- **显著偏移特征**: {white['significant_drift']} ({white['significant_drift']/white['total_features']*100:.1f}%)\n")
            f.write(f"- **严重偏移特征** (KS≥0.2): {white['severe_drift']} ({white['severe_drift']/white['total_features']*100:.1f}%)\n")
            f.write(f"- **极严重偏移特征** (KS≥0.4): {white['extreme_drift']} ({white['extreme_drift']/white['total_features']*100:.1f}%)\n")
            f.write(f"- **平均KS统计量**: {white['avg_ks']:.4f}\n")
            f.write(f"- **最大KS统计量**: {white['max_ks']:.4f}\n\n")
            
            f.write("### 黑样本偏移情况 (train黑样本 vs bad250623)\n")
            black = drift_summary['black_samples']
            f.write(f"- **总特征数**: {black['total_features']}\n")
            f.write(f"- **显著偏移特征**: {black['significant_drift']} ({black['significant_drift']/black['total_features']*100:.1f}%)\n")
            f.write(f"- **严重偏移特征** (KS≥0.2): {black['severe_drift']} ({black['severe_drift']/black['total_features']*100:.1f}%)\n")
            f.write(f"- **极严重偏移特征** (KS≥0.4): {black['extreme_drift']} ({black['extreme_drift']/black['total_features']*100:.1f}%)\n")
            f.write(f"- **平均KS统计量**: {black['avg_ks']:.4f}\n")
            f.write(f"- **最大KS统计量**: {black['max_ks']:.4f}\n\n")
            
            f.write("## 🔍 偏移原因分析\n\n")
            f.write("### 可能的原因:\n")
            f.write("1. **时间偏移**: 数据收集时间不同，恶意软件技术演进\n")
            f.write("2. **来源偏移**: 不同的数据源或收集方法\n")
            f.write("3. **标注偏移**: 标注标准或质量的变化\n")
            f.write("4. **预处理偏移**: 特征提取方法的差异\n\n")
            
            f.write("## 💡 建议的解决方案\n\n")
            f.write("### 短期解决方案:\n")
            f.write("1. **特征选择**: 移除偏移严重的特征\n")
            f.write("2. **数据标准化**: 对偏移特征进行标准化处理\n")
            f.write("3. **域适应**: 使用域适应技术减少偏移影响\n\n")
            
            f.write("### 长期解决方案:\n")
            f.write("1. **数据重新收集**: 确保数据收集标准一致\n")
            f.write("2. **特征工程**: 设计更稳定的特征\n")
            f.write("3. **持续监控**: 建立特征偏移监控机制\n")
        
        print(f"✅ 偏移分析报告已保存: {report_file}")
    
    def run_analysis(self):
        """运行完整的偏移分析"""
        print("🎯 特征偏移深度分析")
        print("=" * 50)
        
        self.load_data()
        drift_summary, white_ks, black_ks = self.analyze_drift_severity()
        self.create_drift_visualization(white_ks, black_ks)
        self.generate_drift_report(drift_summary)
        
        print(f"\n🎉 偏移分析完成！")
        print(f"📁 结果已保存到 featuretrain/ 文件夹")

if __name__ == "__main__":
    analyzer = FeatureDriftAnalyzer()
    analyzer.run_analysis()
