#!/usr/bin/env python3
"""
综合特征分析
结合KS检验和SHAP分析，全面分析误报原因
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_analysis import FeatureAnalyzer
from shap_analysis import SHAPAnalyzer
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveAnalyzer:
    def __init__(self):
        self.ks_analyzer = FeatureAnalyzer()
        self.shap_analyzer = SHAPAnalyzer()
        self.ks_results = None
        self.shap_importance = None
        
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("🎯 综合特征分析")
        print("=" * 60)
        
        # 1. 加载数据
        print("📊 步骤1: 加载数据")
        if not self.ks_analyzer.load_data():
            return False
        
        if not self.shap_analyzer.load_data():
            return False
        
        # 2. KS检验分析
        print("\n🔍 步骤2: KS检验分析")
        self.ks_results = self.ks_analyzer.ks_test_analysis()
        self.ks_analyzer.plot_ks_results(self.ks_results)
        self.ks_analyzer.plot_feature_distributions(self.ks_results)
        
        # 3. SHAP分析
        print("\n🤖 步骤3: SHAP分析")
        X, y = self.shap_analyzer.prepare_training_data()
        X_train, X_test, y_train, y_test = self.shap_analyzer.train_model(X, y)
        X_sample = self.shap_analyzer.create_shap_explainer(X_train)
        misc_scaled, shap_values, predictions, probabilities = self.shap_analyzer.analyze_misclassified_samples()
        
        self.shap_analyzer.plot_shap_summary(misc_scaled, shap_values)
        for i in range(min(3, len(misc_scaled))):
            self.shap_analyzer.plot_shap_waterfall(misc_scaled, shap_values, i)
        
        self.shap_importance = self.shap_analyzer.analyze_top_features(shap_values)
        
        # 4. 综合分析
        print("\n📈 步骤4: 综合分析")
        self.compare_ks_shap_results()
        self.generate_comprehensive_report()
        
        return True
    
    def compare_ks_shap_results(self):
        """比较KS检验和SHAP分析结果"""
        print("\n🔄 比较KS检验和SHAP分析结果...")
        
        # 合并结果
        ks_top20 = self.ks_results.head(20)[['feature', 'ks_statistic', 'p_value']].copy()
        shap_top20 = self.shap_importance.head(20)[['feature', 'importance']].copy()
        
        # 合并数据
        comparison = pd.merge(ks_top20, shap_top20, on='feature', how='outer')
        comparison['ks_rank'] = comparison['ks_statistic'].rank(ascending=False)
        comparison['shap_rank'] = comparison['importance'].rank(ascending=False)
        comparison['rank_diff'] = abs(comparison['ks_rank'] - comparison['shap_rank'])
        
        # 填充缺失值
        comparison = comparison.fillna(0)
        
        # 保存比较结果
        comparison.to_excel('ks_shap_comparison.xlsx', index=False)
        print("✅ KS-SHAP比较结果已保存到: ks_shap_comparison.xlsx")
        
        # 可视化比较
        self.plot_ks_shap_comparison(comparison)
        
        return comparison
    
    def plot_ks_shap_comparison(self, comparison):
        """绘制KS检验和SHAP结果比较图"""
        print("📊 绘制KS-SHAP比较图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('KS检验 vs SHAP分析结果比较', fontsize=16, fontweight='bold')
        
        # 1. 散点图：KS统计量 vs SHAP重要性
        ax1 = axes[0, 0]
        valid_data = comparison.dropna()
        scatter = ax1.scatter(valid_data['ks_statistic'], valid_data['importance'], 
                            alpha=0.7, s=60)
        ax1.set_xlabel('KS统计量')
        ax1.set_ylabel('SHAP重要性')
        ax1.set_title('KS统计量 vs SHAP重要性')
        ax1.grid(True, alpha=0.3)
        
        # 添加相关系数
        if len(valid_data) > 1:
            corr = valid_data['ks_statistic'].corr(valid_data['importance'])
            ax1.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax1.transAxes, 
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # 2. 排名比较
        ax2 = axes[0, 1]
        valid_ranks = comparison[comparison['ks_rank'] > 0]
        ax2.scatter(valid_ranks['ks_rank'], valid_ranks['shap_rank'], alpha=0.7)
        ax2.plot([0, 20], [0, 20], 'r--', alpha=0.5, label='完全一致线')
        ax2.set_xlabel('KS排名')
        ax2.set_ylabel('SHAP排名')
        ax2.set_title('特征重要性排名比较')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 共同重要特征
        ax3 = axes[1, 0]
        ks_top10 = set(self.ks_results.head(10)['feature'])
        shap_top10 = set(self.shap_importance.head(10)['feature'])
        
        overlap = len(ks_top10 & shap_top10)
        ks_only = len(ks_top10 - shap_top10)
        shap_only = len(shap_top10 - ks_top10)
        
        labels = [f'共同重要\n{overlap}个', f'仅KS重要\n{ks_only}个', f'仅SHAP重要\n{shap_only}个']
        sizes = [overlap, ks_only, shap_only]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Top10特征重叠情况')
        
        # 4. 排名差异分布
        ax4 = axes[1, 1]
        rank_diffs = comparison[comparison['rank_diff'] > 0]['rank_diff']
        ax4.hist(rank_diffs, bins=10, alpha=0.7, color='orange')
        ax4.set_xlabel('排名差异')
        ax4.set_ylabel('特征数量')
        ax4.set_title('KS与SHAP排名差异分布')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ks_shap_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ KS-SHAP比较图已保存到: ks_shap_comparison.png")
        plt.show()
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n📝 生成综合分析报告...")
        
        report = []
        report.append("# 误报样本特征分析综合报告\n")
        report.append("## 1. 数据概况\n")
        report.append(f"- 白样本数量: {len(self.ks_analyzer.white_data)}")
        report.append(f"- 黑样本数量: {len(self.ks_analyzer.black_data)}")
        report.append(f"- 误报样本数量: {len(self.ks_analyzer.misclassified_data)}")
        report.append(f"- 特征数量: {len(self.ks_analyzer.feature_names)}\n")
        
        report.append("## 2. KS检验分析结果\n")
        significant_features = self.ks_results[self.ks_results['significant']]
        report.append(f"- 显著差异特征数量: {len(significant_features)}")
        report.append(f"- 显著性比例: {len(significant_features)/len(self.ks_results)*100:.1f}%")
        
        report.append("\n### KS统计量最大的前10个特征:")
        for i, row in self.ks_results.head(10).iterrows():
            report.append(f"{i+1}. {row['feature']} (KS={row['ks_statistic']:.4f}, p={row['p_value']:.2e})")
        
        report.append("\n## 3. SHAP分析结果\n")
        report.append(f"- 模型类型: 随机森林")
        report.append(f"- 特征重要性分析基于: 误报样本")
        
        report.append("\n### SHAP重要性最高的前10个特征:")
        for i, row in self.shap_importance.head(10).iterrows():
            report.append(f"{i+1}. {row['feature']} (重要性={row['importance']:.6f})")
        
        report.append("\n## 4. 综合分析结论\n")
        
        # 找出共同重要的特征
        ks_top10 = set(self.ks_results.head(10)['feature'])
        shap_top10 = set(self.shap_importance.head(10)['feature'])
        common_features = ks_top10 & shap_top10
        
        report.append(f"### 4.1 关键发现")
        report.append(f"- KS检验和SHAP分析共同识别的重要特征: {len(common_features)}个")
        if common_features:
            report.append("- 共同重要特征列表:")
            for feature in sorted(common_features):
                report.append(f"  * {feature}")
        
        report.append(f"\n### 4.2 误报原因分析")
        report.append("基于KS检验和SHAP分析，误报可能的原因包括:")
        report.append("1. 某些特征在白样本和误报样本间存在显著分布差异")
        report.append("2. 模型过度依赖某些特征进行分类决策")
        report.append("3. 训练数据可能存在标签噪声或特征工程问题")
        
        report.append(f"\n### 4.3 改进建议")
        report.append("1. 重点关注KS检验显著且SHAP重要性高的特征")
        report.append("2. 考虑对重要特征进行重新工程或标准化")
        report.append("3. 增加更多样化的训练样本")
        report.append("4. 调整模型参数或尝试其他算法")
        
        # 保存报告
        with open('comprehensive_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("✅ 综合分析报告已保存到: comprehensive_analysis_report.md")
        
        # 打印关键结论
        print("\n📋 关键结论:")
        print(f"  - 显著差异特征: {len(significant_features)}/{len(self.ks_results)}")
        print(f"  - 共同重要特征: {len(common_features)}个")
        if common_features:
            print(f"  - 关键特征: {', '.join(list(common_features)[:5])}")

def main():
    """主函数"""
    analyzer = ComprehensiveAnalyzer()
    
    try:
        success = analyzer.run_full_analysis()
        if success:
            print("\n🎉 综合分析完成!")
            print("\n📄 生成的所有文件:")
            print("  KS检验分析:")
            print("    - ks_test_results.xlsx")
            print("    - ks_analysis_results.png")
            print("    - feature_distributions.png")
            print("  SHAP分析:")
            print("    - random_forest_model.pkl")
            print("    - feature_scaler.pkl")
            print("    - shap_feature_importance.xlsx")
            print("    - shap_summary_plot.png")
            print("    - shap_bar_plot.png")
            print("    - shap_waterfall_sample_*.png")
            print("  综合分析:")
            print("    - ks_shap_comparison.xlsx")
            print("    - ks_shap_comparison.png")
            print("    - comprehensive_analysis_report.md")
        else:
            print("❌ 分析失败")
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")

if __name__ == "__main__":
    main()
