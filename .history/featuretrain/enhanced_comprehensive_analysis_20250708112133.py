#!/usr/bin/env python3
"""
增强版综合特征分析
整合KS检验、特征漂移、SHAP分析等所有数据源
提供更全面的特征洞察和模型优化建议
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

class EnhancedComprehensiveAnalysis:
    def __init__(self):
        """初始化增强版综合分析器"""
        self.output_dir = Path('featuretrain')

        # 数据存储
        self.drift_data = {}
        self.misclassification_data = {}
        self.shap_data = {}
        self.feature_insights = {}

    def load_all_data_sources(self):
        """加载所有数据源"""
        print("📊 加载所有分析数据源...")

        try:
            # 1. 加载特征漂移数据
            print("  📖 加载特征漂移数据...")
            self.drift_data['white_drift'] = pd.read_csv('featuretrain/ks_analysis_white_comparison.csv')
            self.drift_data['black_drift'] = pd.read_csv('featuretrain/ks_analysis_black_comparison.csv')

            # 2. 加载误报分析数据
            print("  📖 加载误报分析数据...")
            try:
                self.misclassification_data['comprehensive'] = pd.read_excel('feature_analysis/comprehensive_ks_analysis.xlsx')
            except:
                print("    ⚠️  comprehensive_ks_analysis.xlsx读取失败")

            try:
                self.misclassification_data['problematic'] = pd.read_excel('feature_analysis/problematic_features.xlsx')
            except:
                print("    ⚠️  problematic_features.xlsx读取失败")

            # 3. 加载SHAP分析数据
            print("  📖 加载SHAP分析数据...")
            try:
                self.shap_data['pattern_analysis'] = pd.read_excel('feature_analysis/shap_pattern_analysis.xlsx')
                print("    ✅ SHAP模式分析数据加载成功")
            except Exception as e:
                print(f"    ⚠️  SHAP模式分析数据读取失败: {e}")

            try:
                self.shap_data['problematic_features'] = pd.read_excel('feature_analysis/shap_problematic_features.xlsx')
                print("    ✅ SHAP问题特征数据加载成功")
            except Exception as e:
                print(f"    ⚠️  SHAP问题特征数据读取失败: {e}")

            # 4. 尝试加载其他可能的SHAP相关文件
            try:
                self.shap_data['integrated_results'] = pd.read_excel('feature_analysis/integrated_analysis_results.xlsx')
                print("    ✅ 整合分析结果加载成功")
            except Exception as e:
                print(f"    ⚠️  整合分析结果读取失败: {e}")

            print("✅ 数据加载完成")
            return True

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False

    def analyze_shap_insights(self):
        """分析SHAP洞察"""
        print("\n🔍 分析SHAP特征重要性洞察...")

        shap_insights = {}

        # 分析SHAP模式数据
        if 'pattern_analysis' in self.shap_data and self.shap_data['pattern_analysis'] is not None:
            pattern_data = self.shap_data['pattern_analysis']
            print(f"    📋 SHAP模式分析数据: {len(pattern_data)} 行")
            print(f"    📋 列名: {list(pattern_data.columns)}")

            # 如果有SHAP值相关列，进行分析
            shap_columns = [col for col in pattern_data.columns if 'shap' in col.lower()]
            if shap_columns:
                print(f"    📊 发现SHAP相关列: {shap_columns}")
                shap_insights['pattern_analysis'] = pattern_data

        # 分析SHAP问题特征
        if 'problematic_features' in self.shap_data and self.shap_data['problematic_features'] is not None:
            prob_data = self.shap_data['problematic_features']
            print(f"    📋 SHAP问题特征数据: {len(prob_data)} 行")
            print(f"    📋 列名: {list(prob_data.columns)}")
            shap_insights['problematic_features'] = prob_data

        # 分析整合结果
        if 'integrated_results' in self.shap_data and self.shap_data['integrated_results'] is not None:
            integrated_data = self.shap_data['integrated_results']
            print(f"    📋 整合分析结果: {len(integrated_data)} 行")
            print(f"    📋 列名: {list(integrated_data.columns)}")
            shap_insights['integrated_results'] = integrated_data

        return shap_insights

    def create_enhanced_feature_importance(self, shap_insights):
        """创建增强版特征重要性分析（结合SHAP）"""
        print("\n⭐ 创建增强版特征重要性分析...")

        # 基础重要性分析（来自之前的分析）
        white_drift = self.drift_data['white_drift']
        black_drift = self.drift_data['black_drift']

        enhanced_importance = []

        for feature in white_drift['feature']:
            white_row = white_drift[white_drift['feature'] == feature].iloc[0]
            black_row = black_drift[black_drift['feature'] == feature].iloc[0]

            # 基础指标
            white_mean = white_row['train_white_mean']
            black_mean = black_row['train_black_mean']
            white_std = white_row['train_white_std']
            black_std = black_row['train_black_std']

            # 分离度和稳定性
            mean_diff = abs(white_mean - black_mean)
            pooled_std = np.sqrt((white_std**2 + black_std**2) / 2)
            separation_score = mean_diff / pooled_std if pooled_std > 0 else 0
            stability_score = 1 / (1 + white_row['ks_statistic'] + black_row['ks_statistic'])
            basic_importance = separation_score * stability_score

            # 初始化SHAP相关指标
            shap_importance = 0
            shap_problematic = False
            shap_pattern_score = 0

            # 整合SHAP分析结果
            if 'pattern_analysis' in shap_insights:
                pattern_data = shap_insights['pattern_analysis']
                if 'feature' in pattern_data.columns:
                    feature_shap = pattern_data[pattern_data['feature'] == feature]
                    if not feature_shap.empty:
                        # 查找SHAP值相关列
                        shap_cols = [col for col in pattern_data.columns if 'shap' in col.lower() and 'value' in col.lower()]
                        if shap_cols:
                            shap_importance = abs(feature_shap[shap_cols[0]].iloc[0])

                        # 查找模式分数
                        pattern_cols = [col for col in pattern_data.columns if 'pattern' in col.lower() or 'score' in col.lower()]
                        if pattern_cols:
                            shap_pattern_score = feature_shap[pattern_cols[0]].iloc[0] if not pd.isna(feature_shap[pattern_cols[0]].iloc[0]) else 0

            # 检查是否为SHAP问题特征
            if 'problematic_features' in shap_insights:
                prob_data = shap_insights['problematic_features']
                if 'feature' in prob_data.columns:
                    shap_problematic = feature in prob_data['feature'].values

            # 综合重要性分数（结合基础分析和SHAP）
            if shap_importance > 0:
                # 如果有SHAP数据，结合使用
                combined_importance = (basic_importance * 0.6 + shap_importance * 0.4)
            else:
                # 如果没有SHAP数据，使用基础重要性
                combined_importance = basic_importance

            enhanced_importance.append({
                'feature': feature,
                'basic_importance': basic_importance,
                'shap_importance': shap_importance,
                'combined_importance': combined_importance,
                'separation_score': separation_score,
                'stability_score': stability_score,
                'white_drift_ks': white_row['ks_statistic'],
                'black_drift_ks': black_row['ks_statistic'],
                'shap_problematic': shap_problematic,
                'shap_pattern_score': shap_pattern_score,
                'white_mean': white_mean,
                'black_mean': black_mean,
                'mean_difference': mean_diff
            })

        enhanced_df = pd.DataFrame(enhanced_importance)
        enhanced_df = enhanced_df.sort_values('combined_importance', ascending=False)

        # 保存增强版重要性分析
        enhanced_file = self.output_dir / 'enhanced_feature_importance_with_shap.csv'
        enhanced_df.to_csv(enhanced_file, index=False, encoding='utf-8')

        print(f"✅ 增强版特征重要性分析已保存: {enhanced_file}")
        print(f"   - 有SHAP数据的特征: {len(enhanced_df[enhanced_df['shap_importance'] > 0])}")
        print(f"   - SHAP标记的问题特征: {enhanced_df['shap_problematic'].sum()}")

        return enhanced_df

    def generate_enhanced_recommendations(self, enhanced_df, shap_insights):
        """生成增强版优化建议（结合SHAP分析）"""
        print("\n💡 生成增强版模型优化建议...")

        enhanced_recommendations = {
            'shap_high_risk_features': [],
            'shap_important_features': [],
            'features_to_remove_enhanced': [],
            'features_to_engineer_enhanced': [],
            'features_to_keep_enhanced': [],
            'shap_specific_insights': [],
            'preprocessing_suggestions_enhanced': [],
            'model_suggestions_enhanced': []
        }

        # 1. SHAP高风险特征
        shap_high_risk = enhanced_df[enhanced_df['shap_problematic'] == True]
        enhanced_recommendations['shap_high_risk_features'] = shap_high_risk['feature'].tolist()

        # 2. SHAP重要特征
        shap_important = enhanced_df[enhanced_df['shap_importance'] > enhanced_df['shap_importance'].quantile(0.8)]
        enhanced_recommendations['shap_important_features'] = shap_important['feature'].tolist()

        # 3. 增强版移除建议（结合SHAP）
        # 高漂移 + 低综合重要性 + SHAP问题特征
        remove_candidates = enhanced_df[
            ((enhanced_df['white_drift_ks'] > 0.4) | (enhanced_df['black_drift_ks'] > 0.4)) &
            (enhanced_df['combined_importance'] < enhanced_df['combined_importance'].quantile(0.3))
        ]
        enhanced_recommendations['features_to_remove_enhanced'] = remove_candidates['feature'].tolist()

        # 4. 增强版工程建议
        # 中等漂移 + 高SHAP重要性
        engineer_candidates = enhanced_df[
            (((enhanced_df['white_drift_ks'] > 0.2) & (enhanced_df['white_drift_ks'] < 0.4)) |
             ((enhanced_df['black_drift_ks'] > 0.2) & (enhanced_df['black_drift_ks'] < 0.4))) &
            (enhanced_df['shap_importance'] > enhanced_df['shap_importance'].quantile(0.6))
        ]
        enhanced_recommendations['features_to_engineer_enhanced'] = engineer_candidates['feature'].tolist()

        # 5. 增强版保留建议
        # 低漂移 + 高综合重要性 + 非SHAP问题特征
        keep_candidates = enhanced_df[
            (enhanced_df['white_drift_ks'] < 0.2) &
            (enhanced_df['black_drift_ks'] < 0.2) &
            (enhanced_df['combined_importance'] > enhanced_df['combined_importance'].quantile(0.7)) &
            (enhanced_df['shap_problematic'] == False)
        ]
        enhanced_recommendations['features_to_keep_enhanced'] = keep_candidates['feature'].tolist()

        # 6. SHAP特定洞察
        if shap_insights:
            enhanced_recommendations['shap_specific_insights'] = [
                f"发现 {len(shap_high_risk)} 个SHAP标记的高风险特征",
                f"发现 {len(shap_important)} 个SHAP高重要性特征",
                "SHAP分析提供了特征贡献的可解释性视角",
                "建议重点关注SHAP值异常的特征"
            ]

        # 7. 增强版预处理建议
        enhanced_recommendations['preprocessing_suggestions_enhanced'] = [
            "基于SHAP分析结果进行特征重要性加权",
            "对SHAP标记的问题特征进行特殊处理",
            "使用SHAP值作为特征选择的参考指标",
            "结合SHAP可解释性优化特征工程策略"
        ]

        # 8. 增强版模型建议
        enhanced_recommendations['model_suggestions_enhanced'] = [
            "集成SHAP可解释性到模型监控中",
            "使用SHAP值检测模型决策的异常模式",
            "基于SHAP分析建立特征重要性动态调整机制",
            "结合SHAP和统计分析进行特征选择"
        ]

        return enhanced_recommendations

    def create_enhanced_visualization(self, enhanced_df, shap_insights):
        """创建增强版可视化（包含SHAP分析）"""
        print("\n📈 生成增强版可视化图表...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. 基础重要性 vs SHAP重要性对比
        valid_shap = enhanced_df[enhanced_df['shap_importance'] > 0]
        if len(valid_shap) > 0:
            axes[0,0].scatter(valid_shap['basic_importance'], valid_shap['shap_importance'],
                             alpha=0.6, color='blue', s=50)
            axes[0,0].set_xlabel('基础重要性分数')
            axes[0,0].set_ylabel('SHAP重要性分数')
            axes[0,0].set_title('基础重要性 vs SHAP重要性')
            axes[0,0].grid(True, alpha=0.3)

            # 添加对角线参考
            max_val = max(valid_shap['basic_importance'].max(), valid_shap['shap_importance'].max())
            axes[0,0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='完全一致线')
            axes[0,0].legend()
        else:
            axes[0,0].text(0.5, 0.5, 'No SHAP Data Available', ha='center', va='center',
                          transform=axes[0,0].transAxes, fontsize=14)
            axes[0,0].set_title('基础重要性 vs SHAP重要性')

        # 2. 综合重要性排名（前15个特征）
        top_features = enhanced_df.head(15)
        y_pos = range(len(top_features))

        # 分别显示基础重要性和SHAP重要性
        axes[0,1].barh(y_pos, top_features['basic_importance'], alpha=0.7,
                      color='lightblue', label='基础重要性')
        axes[0,1].barh(y_pos, top_features['shap_importance'], alpha=0.7,
                      color='orange', label='SHAP重要性')
        axes[0,1].set_yticks(y_pos)
        axes[0,1].set_yticklabels(top_features['feature'], fontsize=9)
        axes[0,1].set_xlabel('重要性分数')
        axes[0,1].set_title('前15个特征的重要性对比')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].invert_yaxis()

        # 3. SHAP问题特征标识
        problem_features = enhanced_df[enhanced_df['shap_problematic'] == True]
        normal_features = enhanced_df[enhanced_df['shap_problematic'] == False]

        axes[0,2].scatter(normal_features['combined_importance'],
                         normal_features['white_drift_ks'],
                         alpha=0.6, color='green', label='正常特征', s=30)
        if len(problem_features) > 0:
            axes[0,2].scatter(problem_features['combined_importance'],
                             problem_features['white_drift_ks'],
                             alpha=0.8, color='red', label='SHAP问题特征', s=50, marker='x')
        axes[0,2].set_xlabel('综合重要性分数')
        axes[0,2].set_ylabel('白样本漂移KS')
        axes[0,2].set_title('SHAP问题特征识别')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)

        # 4. 特征分类饼图（基于SHAP和漂移）
        categories = {'稳定重要': 0, 'SHAP重要但漂移': 0, 'SHAP问题': 0, '低重要性': 0}

        for _, row in enhanced_df.iterrows():
            if row['shap_problematic']:
                categories['SHAP问题'] += 1
            elif row['shap_importance'] > enhanced_df['shap_importance'].quantile(0.7) and \
                 (row['white_drift_ks'] > 0.2 or row['black_drift_ks'] > 0.2):
                categories['SHAP重要但漂移'] += 1
            elif row['combined_importance'] > enhanced_df['combined_importance'].quantile(0.7) and \
                 row['white_drift_ks'] < 0.2 and row['black_drift_ks'] < 0.2:
                categories['稳定重要'] += 1
            else:
                categories['低重要性'] += 1

        colors = ['green', 'orange', 'red', 'gray']
        axes[1,0].pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%',
                     colors=colors, startangle=90)
        axes[1,0].set_title('特征分类分布（基于SHAP+漂移）')

        # 5. 漂移 vs 综合重要性散点图
        axes[1,1].scatter(enhanced_df['combined_importance'],
                         enhanced_df['white_drift_ks'],
                         alpha=0.6, color='blue', label='白样本漂移', s=30)
        axes[1,1].scatter(enhanced_df['combined_importance'],
                         enhanced_df['black_drift_ks'],
                         alpha=0.6, color='red', label='黑样本漂移', s=30)
        axes[1,1].set_xlabel('综合重要性分数')
        axes[1,1].set_ylabel('KS漂移统计量')
        axes[1,1].set_title('综合重要性 vs 特征漂移')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        # 添加象限分割线
        axes[1,1].axhline(y=0.2, color='orange', linestyle='--', alpha=0.7)
        axes[1,1].axvline(x=enhanced_df['combined_importance'].median(),
                         color='green', linestyle='--', alpha=0.7)

        # 6. SHAP重要性分布
        shap_values = enhanced_df[enhanced_df['shap_importance'] > 0]['shap_importance']
        if len(shap_values) > 0:
            axes[1,2].hist(shap_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[1,2].set_xlabel('SHAP重要性分数')
            axes[1,2].set_ylabel('特征数量')
            axes[1,2].set_title('SHAP重要性分布')
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, 'No SHAP Data Available', ha='center', va='center',
                          transform=axes[1,2].transAxes, fontsize=14)
            axes[1,2].set_title('SHAP重要性分布')

        plt.tight_layout()

        # 保存图片
        output_file = self.output_dir / 'enhanced_comprehensive_analysis_with_shap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ 增强版综合分析图已保存: {output_file}")

    def generate_enhanced_report(self, enhanced_df, shap_insights, enhanced_recommendations):
        """生成增强版综合分析报告"""
        print("\n📄 生成增强版综合分析报告...")

        report_file = self.output_dir / 'enhanced_comprehensive_analysis_report.md'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 增强版综合特征分析报告（整合SHAP分析）\n\n")

            f.write("## 🎯 分析目标\n\n")
            f.write("本报告整合了KS检验、特征漂移分析和SHAP可解释性分析的所有结果，")
            f.write("提供更全面、更可解释的特征洞察和模型优化建议。\n\n")

            f.write("## 📊 数据源整合\n\n")
            f.write("### 使用的数据源：\n")
            f.write("1. **特征漂移数据**: featuretrain/ks_analysis_*_comparison.csv\n")
            f.write("2. **误报分析数据**: feature_analysis/comprehensive_ks_analysis.xlsx\n")
            f.write("3. **SHAP分析数据**: feature_analysis/shap_*.xlsx\n")
            f.write("4. **整合分析结果**: feature_analysis/integrated_analysis_results.xlsx\n\n")

            # SHAP分析结果
            f.write("## 🔍 SHAP分析洞察\n\n")
            shap_features_count = len(enhanced_df[enhanced_df['shap_importance'] > 0])
            shap_problematic_count = enhanced_df['shap_problematic'].sum()

            f.write(f"- **有SHAP数据的特征**: {shap_features_count} 个\n")
            f.write(f"- **SHAP标记的问题特征**: {shap_problematic_count} 个\n")

            if shap_features_count > 0:
                f.write(f"- **SHAP重要性最高的5个特征**:\n")
                top_shap = enhanced_df[enhanced_df['shap_importance'] > 0].nlargest(5, 'shap_importance')
                for i, row in top_shap.iterrows():
                    f.write(f"  {i+1}. `{row['feature']}` (SHAP: {row['shap_importance']:.3f})\n")
            f.write("\n")

            # 综合重要性分析
            f.write("## ⭐ 综合重要性分析（基础+SHAP）\n\n")
            f.write(f"- **综合重要性最高的5个特征**:\n")
            for i, row in enhanced_df.head(5).iterrows():
                f.write(f"  {i+1}. `{row['feature']}` (综合: {row['combined_importance']:.3f}, ")
                f.write(f"基础: {row['basic_importance']:.3f}, SHAP: {row['shap_importance']:.3f})\n")
            f.write("\n")

            # 增强版优化建议
            f.write("## 💡 增强版模型优化建议\n\n")

            f.write("### 🚨 SHAP高风险特征\n")
            f.write("以下特征被SHAP分析标记为高风险，建议重点关注:\n")
            for feature in enhanced_recommendations['shap_high_risk_features']:
                f.write(f"- `{feature}`\n")
            f.write("\n")

            f.write("### 🗑️ 增强版移除建议\n")
            f.write("结合SHAP分析，以下特征建议移除:\n")
            for feature in enhanced_recommendations['features_to_remove_enhanced'][:10]:
                f.write(f"- `{feature}`\n")
            f.write("\n")

            f.write("### 🔧 增强版特征工程建议\n")
            f.write("结合SHAP重要性，以下特征建议进行工程优化:\n")
            for feature in enhanced_recommendations['features_to_engineer_enhanced'][:10]:
                f.write(f"- `{feature}`\n")
            f.write("\n")

            f.write("### ✅ 增强版核心保留特征\n")
            f.write("结合所有分析，以下特征建议作为核心保留:\n")
            for feature in enhanced_recommendations['features_to_keep_enhanced'][:15]:
                f.write(f"- `{feature}`\n")
            f.write("\n")

            # SHAP特定洞察
            f.write("### 🧠 SHAP特定洞察\n")
            for insight in enhanced_recommendations['shap_specific_insights']:
                f.write(f"- {insight}\n")
            f.write("\n")

            # 增强版建议
            f.write("### 🔄 增强版预处理建议\n")
            for suggestion in enhanced_recommendations['preprocessing_suggestions_enhanced']:
                f.write(f"- {suggestion}\n")
            f.write("\n")

            f.write("### 🤖 增强版模型策略建议\n")
            for suggestion in enhanced_recommendations['model_suggestions_enhanced']:
                f.write(f"- {suggestion}\n")
            f.write("\n")

            f.write("## 📁 生成的文件\n\n")
            f.write("- `enhanced_feature_importance_with_shap.csv`: 增强版特征重要性分析\n")
            f.write("- `enhanced_comprehensive_analysis_with_shap.png`: 增强版可视化图表\n")
            f.write("- `enhanced_comprehensive_analysis_report.md`: 本报告\n")

        print(f"✅ 增强版综合分析报告已保存: {report_file}")

    def run_enhanced_analysis(self):
        """运行增强版综合分析"""
        print("🎯 增强版综合特征分析（整合SHAP）")
        print("=" * 70)

        # 1. 加载所有数据源
        if not self.load_all_data_sources():
            return False

        # 2. 分析SHAP洞察
        shap_insights = self.analyze_shap_insights()

        # 3. 创建增强版特征重要性分析
        enhanced_df = self.create_enhanced_feature_importance(shap_insights)

        # 4. 生成增强版优化建议
        enhanced_recommendations = self.generate_enhanced_recommendations(enhanced_df, shap_insights)

        # 5. 创建增强版可视化
        self.create_enhanced_visualization(enhanced_df, shap_insights)

        # 6. 生成增强版报告
        self.generate_enhanced_report(enhanced_df, shap_insights, enhanced_recommendations)

        print(f"\n🎉 增强版综合分析完成！")
        print(f"📁 所有结果已保存到 featuretrain/ 文件夹")
        print(f"🧠 SHAP分析已整合到特征重要性评估中")

        return True

def main():
    """主函数"""
    analyzer = EnhancedComprehensiveAnalysis()
    success = analyzer.run_enhanced_analysis()

    if success:
        print("\n✅ 增强版综合分析完成！")
        print("💡 现在的分析整合了KS检验、特征漂移和SHAP可解释性分析")
    else:
        print("\n❌ 增强版分析失败！")

if __name__ == "__main__":
    main()