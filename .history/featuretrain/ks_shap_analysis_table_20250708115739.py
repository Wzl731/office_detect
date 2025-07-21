#!/usr/bin/env python3
"""
KS-SHAP二维分析表
以KS统计量为横轴，SHAP重要性为纵轴
生成综合分析表格和可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class KSSHAPAnalysisTable:
    def __init__(self):
        """初始化KS-SHAP分析表生成器"""
        self.output_dir = Path('featuretrain')
        
    def load_comprehensive_data(self):
        """加载所有相关数据"""
        print("📊 加载KS和SHAP分析数据...")
        
        try:
            # 1. 加载KS漂移数据
            self.white_drift = pd.read_csv('featuretrain/ks_analysis_white_comparison.csv')
            self.black_drift = pd.read_csv('featuretrain/ks_analysis_black_comparison.csv')
            
            # 2. 加载SHAP分析数据
            self.shap_pattern = pd.read_excel('feature_analysis/shap_pattern_analysis.xlsx')
            
            # 3. 加载三维评分数据
            self.three_dim_scores = pd.read_csv('featuretrain/three_dimensional_feature_scores.csv')
            
            print(f"✅ 数据加载完成:")
            print(f"   - 白样本KS数据: {len(self.white_drift)} 个特征")
            print(f"   - 黑样本KS数据: {len(self.black_drift)} 个特征")
            print(f"   - SHAP模式数据: {len(self.shap_pattern)} 个特征")
            print(f"   - 三维评分数据: {len(self.three_dim_scores)} 个特征")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def create_ks_shap_analysis_table(self):
        """创建KS-SHAP综合分析表"""
        print("\n📋 生成KS-SHAP综合分析表...")
        
        try:
            analysis_table = []
            
            # 获取所有特征列表
            features = self.white_drift['feature'].tolist()
            
            for feature in features:
                # 获取KS数据
                white_row = self.white_drift[self.white_drift['feature'] == feature].iloc[0]
                black_row = self.black_drift[self.black_drift['feature'] == feature].iloc[0]
                
                # 计算KS指标
                white_ks = white_row['ks_statistic']
                black_ks = black_row['ks_statistic']
                max_ks = max(white_ks, black_ks)
                avg_ks = (white_ks + black_ks) / 2
                
                # 获取SHAP数据
                shap_row = self.shap_pattern[self.shap_pattern['feature'] == feature]
                if not shap_row.empty:
                    shap_importance = shap_row['misc_importance'].iloc[0]
                    if pd.isna(shap_importance):
                        shap_importance = 0
                    
                    # 获取其他SHAP指标
                    misc_mean_shap = shap_row.get('misc_mean_shap', pd.Series([0])).iloc[0]
                    white_mean_shap = shap_row.get('white_mean_shap', pd.Series([0])).iloc[0]
                    black_mean_shap = shap_row.get('black_mean_shap', pd.Series([0])).iloc[0]
                    misc_closer_to_black_shap = shap_row.get('misc_closer_to_black_shap', pd.Series([False])).iloc[0]
                else:
                    shap_importance = 0
                    misc_mean_shap = white_mean_shap = black_mean_shap = 0
                    misc_closer_to_black_shap = False
                
                # 获取三维评分
                three_dim_row = self.three_dim_scores[self.three_dim_scores['feature'] == feature]
                if not three_dim_row.empty:
                    stability_score = three_dim_row['stability_score'].iloc[0]
                    discriminative_power = three_dim_row['discriminative_power'].iloc[0]
                    interpretability_score = three_dim_row['interpretability_score'].iloc[0]
                    overall_score = three_dim_row['overall_score'].iloc[0]
                    shap_risk_flag = three_dim_row['shap_risk_flag'].iloc[0]
                else:
                    stability_score = discriminative_power = interpretability_score = overall_score = 0
                    shap_risk_flag = False
                
                # KS-SHAP象限分类
                ks_threshold = 0.2  # KS阈值
                shap_threshold = 0.01  # SHAP重要性阈值
                
                if max_ks >= ks_threshold and shap_importance >= shap_threshold:
                    quadrant = "高KS-高SHAP"
                    risk_level = "需要关注"
                elif max_ks >= ks_threshold and shap_importance < shap_threshold:
                    quadrant = "高KS-低SHAP"
                    risk_level = "漂移风险"
                elif max_ks < ks_threshold and shap_importance >= shap_threshold:
                    quadrant = "低KS-高SHAP"
                    risk_level = "核心特征"
                else:
                    quadrant = "低KS-低SHAP"
                    risk_level = "可考虑移除"
                
                # 综合建议
                if shap_risk_flag:
                    recommendation = "立即移除 - SHAP高风险"
                elif quadrant == "核心特征":
                    recommendation = "重点保留 - 稳定且重要"
                elif quadrant == "需要关注":
                    recommendation = "深度分析 - 重要但不稳定"
                elif quadrant == "漂移风险":
                    recommendation = "监控或移除 - 不稳定且不重要"
                else:
                    recommendation = "考虑移除 - 低价值"
                
                analysis_table.append({
                    'feature': feature,
                    'white_ks': white_ks,
                    'black_ks': black_ks,
                    'max_ks': max_ks,
                    'avg_ks': avg_ks,
                    'shap_importance': shap_importance,
                    'misc_mean_shap': misc_mean_shap,
                    'white_mean_shap': white_mean_shap,
                    'black_mean_shap': black_mean_shap,
                    'misc_closer_to_black_shap': misc_closer_to_black_shap,
                    'stability_score': stability_score,
                    'discriminative_power': discriminative_power,
                    'interpretability_score': interpretability_score,
                    'overall_score': overall_score,
                    'shap_risk_flag': shap_risk_flag,
                    'ks_shap_quadrant': quadrant,
                    'risk_level': risk_level,
                    'recommendation': recommendation
                })
            
            self.ks_shap_table = pd.DataFrame(analysis_table)
            
            # 按照综合重要性排序
            self.ks_shap_table['composite_score'] = (
                self.ks_shap_table['shap_importance'] * 0.4 +
                (1 - self.ks_shap_table['max_ks']) * 0.3 +  # KS越小越好
                self.ks_shap_table['discriminative_power'] * 0.3
            )
            
            self.ks_shap_table = self.ks_shap_table.sort_values('composite_score', ascending=False)
            
            # 保存分析表
            table_file = self.output_dir / 'ks_shap_comprehensive_analysis_table.csv'
            self.ks_shap_table.to_csv(table_file, index=False, encoding='utf-8')
            
            print(f"✅ KS-SHAP分析表已生成: {table_file}")
            print(f"   - 总特征数: {len(self.ks_shap_table)}")
            
            # 统计象限分布
            quadrant_counts = self.ks_shap_table['ks_shap_quadrant'].value_counts()
            print(f"📊 象限分布:")
            for quadrant, count in quadrant_counts.items():
                print(f"   - {quadrant}: {count} 个特征")
            
            return True
            
        except Exception as e:
            print(f"❌ 分析表生成失败: {e}")
            return False
    
    def create_ks_shap_visualizations(self):
        """创建KS-SHAP可视化图表"""
        print("\n📈 生成KS-SHAP可视化图表...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. KS vs SHAP散点图（主图）
            ax1 = axes[0, 0]
            
            # 根据象限设置颜色
            quadrant_colors = {
                '高KS-高SHAP': 'red',
                '高KS-低SHAP': 'orange', 
                '低KS-高SHAP': 'green',
                '低KS-低SHAP': 'gray'
            }
            
            for quadrant, color in quadrant_colors.items():
                data = self.ks_shap_table[self.ks_shap_table['ks_shap_quadrant'] == quadrant]
                ax1.scatter(data['max_ks'], data['shap_importance'], 
                           c=color, label=quadrant, alpha=0.7, s=50)
            
            # 添加象限分割线
            ax1.axhline(y=0.01, color='black', linestyle='--', alpha=0.5, label='SHAP阈值')
            ax1.axvline(x=0.2, color='black', linestyle='--', alpha=0.5, label='KS阈值')
            
            ax1.set_xlabel('最大KS统计量 (特征漂移程度)')
            ax1.set_ylabel('SHAP重要性')
            ax1.set_title('KS vs SHAP 二维分析图')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. 象限分布饼图
            ax2 = axes[0, 1]
            quadrant_counts = self.ks_shap_table['ks_shap_quadrant'].value_counts()
            colors = ['green', 'red', 'orange', 'gray']
            ax2.pie(quadrant_counts.values, labels=quadrant_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title('KS-SHAP象限分布')
            
            # 3. 风险等级分布
            ax3 = axes[1, 0]
            risk_counts = self.ks_shap_table['risk_level'].value_counts()
            risk_colors = {'核心特征': 'green', '需要关注': 'red', '漂移风险': 'orange', '可考虑移除': 'gray'}
            bar_colors = [risk_colors.get(risk, 'blue') for risk in risk_counts.index]
            
            bars = ax3.bar(risk_counts.index, risk_counts.values, color=bar_colors, alpha=0.7)
            ax3.set_xlabel('风险等级')
            ax3.set_ylabel('特征数量')
            ax3.set_title('特征风险等级分布')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, count in zip(bars, risk_counts.values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(count), ha='center', va='bottom')
            
            # 4. 综合评分vs KS散点图
            ax4 = axes[1, 1]
            scatter = ax4.scatter(self.ks_shap_table['max_ks'], 
                                 self.ks_shap_table['composite_score'],
                                 c=self.ks_shap_table['shap_importance'], 
                                 s=60, alpha=0.7, cmap='viridis')
            
            ax4.set_xlabel('最大KS统计量')
            ax4.set_ylabel('综合评分')
            ax4.set_title('KS vs 综合评分 (颜色=SHAP重要性)')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='SHAP重要性')
            
            plt.tight_layout()
            
            # 保存可视化
            viz_file = self.output_dir / 'ks_shap_analysis_visualization.png'
            plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ KS-SHAP可视化已保存: {viz_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 可视化生成失败: {e}")
            return False
    
    def create_detailed_quadrant_tables(self):
        """创建详细的象限分析表"""
        print("\n📋 生成详细象限分析表...")
        
        try:
            quadrants = self.ks_shap_table['ks_shap_quadrant'].unique()
            
            for quadrant in quadrants:
                quadrant_data = self.ks_shap_table[
                    self.ks_shap_table['ks_shap_quadrant'] == quadrant
                ].copy()
                
                # 按综合评分排序
                quadrant_data = quadrant_data.sort_values('composite_score', ascending=False)
                
                # 保存象限详细表
                quadrant_file = self.output_dir / f'ks_shap_{quadrant.replace("-", "_").replace(" ", "_")}_details.csv'
                quadrant_data.to_csv(quadrant_file, index=False, encoding='utf-8')
                
                print(f"   - {quadrant}: {len(quadrant_data)} 个特征 -> {quadrant_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 象限表生成失败: {e}")
            return False
    
    def generate_summary_report(self):
        """生成KS-SHAP分析摘要报告"""
        print("\n📄 生成KS-SHAP分析报告...")
        
        try:
            report_file = self.output_dir / 'ks_shap_analysis_report.md'
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# KS-SHAP二维分析报告\n\n")
                
                f.write("## 🎯 分析目标\n\n")
                f.write("本报告以KS统计量为横轴、SHAP重要性为纵轴，")
                f.write("对特征进行二维分析，识别不同象限的特征特性和处理策略。\n\n")
                
                f.write("## 📊 分析维度\n\n")
                f.write("- **横轴 (KS统计量)**: 反映特征在不同数据集间的分布差异（漂移程度）\n")
                f.write("- **纵轴 (SHAP重要性)**: 反映特征对模型决策的重要程度\n")
                f.write("- **象限划分**: KS阈值=0.2, SHAP阈值=0.01\n\n")
                
                # 象限分析
                f.write("## 🔍 四象限分析\n\n")
                
                quadrant_counts = self.ks_shap_table['ks_shap_quadrant'].value_counts()
                total_features = len(self.ks_shap_table)
                
                for quadrant, count in quadrant_counts.items():
                    percentage = count / total_features * 100
                    f.write(f"### {quadrant} ({count}个特征, {percentage:.1f}%)\n\n")
                    
                    # 获取该象限的特征
                    quadrant_features = self.ks_shap_table[
                        self.ks_shap_table['ks_shap_quadrant'] == quadrant
                    ].head(5)  # 显示前5个
                    
                    if quadrant == "低KS-高SHAP":
                        f.write("**特征特性**: 稳定且重要，是模型的核心特征\n")
                        f.write("**处理建议**: 重点保留，作为模型基础\n")
                    elif quadrant == "高KS-高SHAP":
                        f.write("**特征特性**: 重要但不稳定，存在漂移风险\n")
                        f.write("**处理建议**: 深度分析，考虑特征工程或监控\n")
                    elif quadrant == "高KS-低SHAP":
                        f.write("**特征特性**: 不稳定且不重要，漂移风险高\n")
                        f.write("**处理建议**: 监控或移除\n")
                    else:  # 低KS-低SHAP
                        f.write("**特征特性**: 稳定但不重要，价值较低\n")
                        f.write("**处理建议**: 可考虑移除以简化模型\n")
                    
                    f.write("\n**代表特征**:\n")
                    for _, row in quadrant_features.iterrows():
                        f.write(f"- `{row['feature']}`: KS={row['max_ks']:.3f}, SHAP={row['shap_importance']:.4f}\n")
                    f.write("\n")
                
                # 重点关注特征
                f.write("## ⭐ 重点关注特征\n\n")
                
                # 核心特征（低KS-高SHAP）
                core_features = self.ks_shap_table[
                    self.ks_shap_table['ks_shap_quadrant'] == '低KS-高SHAP'
                ].head(10)
                
                f.write("### 核心保留特征 (低KS-高SHAP)\n")
                for i, (_, row) in enumerate(core_features.iterrows()):
                    f.write(f"{i+1}. **`{row['feature']}`**\n")
                    f.write(f"   - KS统计量: {row['max_ks']:.3f}\n")
                    f.write(f"   - SHAP重要性: {row['shap_importance']:.4f}\n")
                    f.write(f"   - 综合评分: {row['composite_score']:.3f}\n\n")
                
                # 风险特征（高KS-高SHAP）
                risk_features = self.ks_shap_table[
                    self.ks_shap_table['ks_shap_quadrant'] == '高KS-高SHAP'
                ].head(5)
                
                if len(risk_features) > 0:
                    f.write("### 需要关注特征 (高KS-高SHAP)\n")
                    for i, (_, row) in enumerate(risk_features.iterrows()):
                        f.write(f"{i+1}. **`{row['feature']}`**\n")
                        f.write(f"   - KS统计量: {row['max_ks']:.3f} ⚠️\n")
                        f.write(f"   - SHAP重要性: {row['shap_importance']:.4f}\n")
                        f.write(f"   - 建议: {row['recommendation']}\n\n")
                
                f.write("## 📁 生成文件\n\n")
                f.write("- `ks_shap_comprehensive_analysis_table.csv`: 完整的KS-SHAP分析表\n")
                f.write("- `ks_shap_analysis_visualization.png`: KS-SHAP可视化图表\n")
                f.write("- `ks_shap_*_details.csv`: 各象限详细分析表\n")
                f.write("- `ks_shap_analysis_report.md`: 本报告\n")
            
            print(f"✅ KS-SHAP分析报告已保存: {report_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 报告生成失败: {e}")
            return False

    def run_ks_shap_analysis(self):
        """运行完整的KS-SHAP分析"""
        print("🎯 KS-SHAP二维分析表生成")
        print("=" * 60)
        print("横轴: KS统计量 | 纵轴: SHAP重要性")
        print("=" * 60)

        # 1. 加载数据
        if not self.load_comprehensive_data():
            return False

        # 2. 生成分析表
        if not self.create_ks_shap_analysis_table():
            return False

        # 3. 创建可视化
        if not self.create_ks_shap_visualizations():
            return False

        # 4. 生成象限详细表
        if not self.create_detailed_quadrant_tables():
            return False

        # 5. 生成摘要报告
        if not self.generate_summary_report():
            return False

        print(f"\n🎉 KS-SHAP二维分析完成！")
        print(f"📁 所有结果已保存到 featuretrain/ 文件夹")
        print(f"📊 已生成四象限分析表和可视化图表")

        # 显示关键统计信息
        print(f"\n📈 关键统计:")
        quadrant_counts = self.ks_shap_table['ks_shap_quadrant'].value_counts()
        for quadrant, count in quadrant_counts.items():
            print(f"   - {quadrant}: {count} 个特征")

        return True

def main():
    """主函数"""
    analyzer = KSSHAPAnalysisTable()
    success = analyzer.run_ks_shap_analysis()

    if success:
        print("\n✅ KS-SHAP二维分析表生成完成！")
        print("💡 请查看生成的表格和可视化文件")
        print("📋 主要文件:")
        print("   - ks_shap_comprehensive_analysis_table.csv (完整分析表)")
        print("   - ks_shap_analysis_visualization.png (可视化图表)")
        print("   - ks_shap_analysis_report.md (分析报告)")
    else:
        print("\n❌ KS-SHAP二维分析失败！")

if __name__ == "__main__":
    main()
