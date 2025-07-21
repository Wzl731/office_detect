#!/usr/bin/env python3
"""
三维特征评估系统
结合SHAP可解释性、KS稳定性、判别力分析
为每个特征打出"解释性 + 稳定性 + 判别力"三维标签
输出特征处理策略和风险雷达图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import matplotlib
from math import pi
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class ThreeDimensionalFeatureAssessment:
    def __init__(self):
        """初始化三维特征评估系统"""
        self.output_dir = Path('featuretrain')
        
        # 数据存储
        self.feature_scores = {}
        self.feature_strategies = {}
        
    def load_comprehensive_data(self):
        """加载综合分析数据"""
        print("📊 加载三维特征评估数据...")
        
        try:
            # 1. 加载特征漂移数据（稳定性）
            self.white_drift = pd.read_csv('featuretrain/ks_analysis_white_comparison.csv')
            self.black_drift = pd.read_csv('featuretrain/ks_analysis_black_comparison.csv')
            
            # 2. 加载SHAP分析数据（可解释性）
            self.shap_pattern = pd.read_excel('feature_analysis/shap_pattern_analysis.xlsx')
            self.shap_problematic = pd.read_excel('feature_analysis/shap_problematic_features.xlsx')
            self.integrated_results = pd.read_excel('feature_analysis/integrated_analysis_results.xlsx')
            
            # 3. 加载误报分析数据（判别力）
            self.comprehensive_ks = pd.read_excel('feature_analysis/comprehensive_ks_analysis.xlsx')
            
            print("✅ 数据加载完成")
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def calculate_three_dimensional_scores(self):
        """计算三维评分：解释性 + 稳定性 + 判别力"""
        print("\n🎯 计算三维特征评分...")
        
        feature_assessments = []
        
        # 获取所有特征列表
        features = self.white_drift['feature'].tolist()
        
        for feature in features:
            try:
                # 1. 稳定性评分 (Stability Score)
                white_row = self.white_drift[self.white_drift['feature'] == feature].iloc[0]
                black_row = self.black_drift[self.black_drift['feature'] == feature].iloc[0]
                
                # 稳定性 = 1 - 平均KS漂移 (越小越稳定)
                avg_drift = (white_row['ks_statistic'] + black_row['ks_statistic']) / 2
                stability_score = max(0, 1 - avg_drift)  # 0-1范围
                
                # 2. 判别力评分 (Discriminative Power)
                # 基于黑白样本分离能力
                white_mean = white_row['train_white_mean']
                black_mean = black_row['train_black_mean']
                white_std = white_row['train_white_std']
                black_std = black_row['train_black_std']
                
                # 计算效应大小作为判别力指标
                mean_diff = abs(white_mean - black_mean)
                pooled_std = np.sqrt((white_std**2 + black_std**2) / 2)
                discriminative_power = min(1, mean_diff / pooled_std if pooled_std > 0 else 0)
                
                # 3. 可解释性评分 (Interpretability Score)
                interpretability_score = 0
                shap_risk_flag = False
                shap_importance = 0
                
                # 从SHAP模式分析获取可解释性
                shap_row = self.shap_pattern[self.shap_pattern['feature'] == feature]
                if not shap_row.empty:
                    # SHAP重要性
                    if 'misc_importance' in shap_row.columns:
                        shap_importance = abs(shap_row['misc_importance'].iloc[0])
                    
                    # SHAP可解释性评分
                    if 'misc_closer_to_black_shap' in shap_row.columns:
                        closer_to_black = shap_row['misc_closer_to_black_shap'].iloc[0]
                        if closer_to_black:
                            interpretability_score = 0.3  # 低可解释性（容易误导）
                        else:
                            interpretability_score = 0.8  # 高可解释性
                    
                    # SHAP相似度调整
                    if 'misc_black_shap_similarity' in shap_row.columns:
                        similarity = shap_row['misc_black_shap_similarity'].iloc[0]
                        if not pd.isna(similarity):
                            # 相似度越高，可解释性越低
                            interpretability_score = max(0.1, 1 - similarity)
                
                # 检查是否为SHAP问题特征
                if feature in self.shap_problematic['feature'].values:
                    shap_risk_flag = True
                    interpretability_score = min(interpretability_score, 0.2)  # 大幅降低可解释性
                
                # 4. 综合风险评估
                # 从integrated_results获取风险评分
                risk_score = 0
                integrated_row = self.integrated_results[self.integrated_results['feature'] == feature]
                if not integrated_row.empty and 'risk_score' in integrated_row.columns:
                    risk_score = integrated_row['risk_score'].iloc[0]
                    if pd.isna(risk_score):
                        risk_score = 0
                
                # 5. 特征分类
                feature_type = self.classify_feature_type(feature)
                
                # 6. 计算综合评分
                # 稳定性权重40%，判别力权重35%，可解释性权重25%
                overall_score = (stability_score * 0.4 + 
                               discriminative_power * 0.35 + 
                               interpretability_score * 0.25)
                
                feature_assessments.append({
                    'feature': feature,
                    'stability_score': stability_score,
                    'discriminative_power': discriminative_power,
                    'interpretability_score': interpretability_score,
                    'overall_score': overall_score,
                    'white_drift_ks': white_row['ks_statistic'],
                    'black_drift_ks': black_row['ks_statistic'],
                    'avg_drift': avg_drift,
                    'shap_importance': shap_importance,
                    'shap_risk_flag': shap_risk_flag,
                    'risk_score': risk_score,
                    'feature_type': feature_type,
                    'white_mean': white_mean,
                    'black_mean': black_mean,
                    'mean_difference': mean_diff
                })
                
            except Exception as e:
                print(f"    ⚠️  特征 {feature} 评估失败: {e}")
                continue
        
        self.feature_scores = pd.DataFrame(feature_assessments)
        self.feature_scores = self.feature_scores.sort_values('overall_score', ascending=False)
        
        # 保存三维评分结果
        scores_file = self.output_dir / 'three_dimensional_feature_scores.csv'
        self.feature_scores.to_csv(scores_file, index=False, encoding='utf-8')
        
        print(f"✅ 三维特征评分完成: {scores_file}")
        print(f"   - 总特征数: {len(self.feature_scores)}")
        print(f"   - 高风险特征: {self.feature_scores['shap_risk_flag'].sum()}")
        
        return self.feature_scores
    
    def classify_feature_type(self, feature):
        """分类特征类型"""
        feature_lower = feature.lower()
        
        if any(keyword in feature_lower for keyword in ['line', 'proc', 'num', 'cnt']):
            return '代码结构'
        elif any(keyword in feature_lower for keyword in ['shell', 'create', 'open', 'close', 'write']):
            return 'VBA函数'
        elif any(keyword in feature_lower for keyword in ['cmd', 'exe', 'dll', 'registry']):
            return '可疑关键词'
        else:
            return '其他'
    
    def generate_feature_strategies(self):
        """生成特征处理策略"""
        print("\n💡 生成特征处理策略...")
        
        strategies = {
            'keep_core': [],           # 保留核心特征
            'remove_risky': [],        # 移除高风险特征
            'combine_weak': [],        # 组合弱特征
            'rule_control': [],        # 规则控制特征
            'monitor_unstable': []     # 监控不稳定特征
        }
        
        for _, row in self.feature_scores.iterrows():
            feature = row['feature']
            stability = row['stability_score']
            discriminative = row['discriminative_power']
            interpretability = row['interpretability_score']
            overall = row['overall_score']
            shap_risk = row['shap_risk_flag']
            
            # 决策逻辑
            if shap_risk or interpretability < 0.3:
                # SHAP高风险或低可解释性 -> 移除
                strategies['remove_risky'].append({
                    'feature': feature,
                    'reason': 'SHAP高风险' if shap_risk else '低可解释性',
                    'scores': f"稳定性:{stability:.2f}, 判别力:{discriminative:.2f}, 可解释性:{interpretability:.2f}"
                })
            
            elif stability > 0.8 and discriminative > 0.6 and interpretability > 0.6:
                # 三维都高 -> 保留核心
                strategies['keep_core'].append({
                    'feature': feature,
                    'reason': '三维评分均优秀',
                    'scores': f"稳定性:{stability:.2f}, 判别力:{discriminative:.2f}, 可解释性:{interpretability:.2f}"
                })
            
            elif stability < 0.5:
                # 稳定性差 -> 监控
                strategies['monitor_unstable'].append({
                    'feature': feature,
                    'reason': '稳定性差，需要监控',
                    'scores': f"稳定性:{stability:.2f}, 判别力:{discriminative:.2f}, 可解释性:{interpretability:.2f}"
                })
            
            elif discriminative < 0.3:
                # 判别力差 -> 组合
                strategies['combine_weak'].append({
                    'feature': feature,
                    'reason': '判别力弱，建议组合',
                    'scores': f"稳定性:{stability:.2f}, 判别力:{discriminative:.2f}, 可解释性:{interpretability:.2f}"
                })
            
            elif interpretability > 0.7 and discriminative > 0.4:
                # 可解释性好，判别力中等 -> 规则控制
                strategies['rule_control'].append({
                    'feature': feature,
                    'reason': '适合规则控制',
                    'scores': f"稳定性:{stability:.2f}, 判别力:{discriminative:.2f}, 可解释性:{interpretability:.2f}"
                })
            
            else:
                # 其他情况 -> 根据综合评分决定
                if overall > 0.6:
                    strategies['keep_core'].append({
                        'feature': feature,
                        'reason': '综合评分良好',
                        'scores': f"稳定性:{stability:.2f}, 判别力:{discriminative:.2f}, 可解释性:{interpretability:.2f}"
                    })
                else:
                    strategies['combine_weak'].append({
                        'feature': feature,
                        'reason': '综合评分一般，建议组合',
                        'scores': f"稳定性:{stability:.2f}, 判别力:{discriminative:.2f}, 可解释性:{interpretability:.2f}"
                    })
        
        self.feature_strategies = strategies
        
        # 保存策略结果
        strategies_file = self.output_dir / 'feature_processing_strategies.json'
        import json
        with open(strategies_file, 'w', encoding='utf-8') as f:
            json.dump(strategies, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 特征处理策略生成完成: {strategies_file}")
        print(f"   - 保留核心: {len(strategies['keep_core'])}")
        print(f"   - 移除高风险: {len(strategies['remove_risky'])}")
        print(f"   - 组合弱特征: {len(strategies['combine_weak'])}")
        print(f"   - 规则控制: {len(strategies['rule_control'])}")
        print(f"   - 监控不稳定: {len(strategies['monitor_unstable'])}")
        
        return strategies
    
    def create_feature_risk_radar(self, top_n=20):
        """创建特征风险雷达图"""
        print(f"\n📈 生成特征风险雷达图 (前{top_n}个特征)...")
        
        # 选择前N个特征
        top_features = self.feature_scores.head(top_n)
        
        # 创建雷达图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        # 雷达图参数
        categories = ['稳定性', '判别力', '可解释性']
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # 闭合
        
        # 1. 高风险特征雷达图
        ax1 = axes[0]
        risky_features = self.feature_scores[self.feature_scores['shap_risk_flag'] == True].head(5)
        
        for i, (_, row) in enumerate(risky_features.iterrows()):
            values = [row['stability_score'], row['discriminative_power'], row['interpretability_score']]
            values += values[:1]  # 闭合
            
            ax1.plot(angles, values, 'o-', linewidth=2, label=row['feature'][:15], alpha=0.7)
            ax1.fill(angles, values, alpha=0.1)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('高风险特征雷达图', size=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax1.grid(True)
        
        # 2. 核心保留特征雷达图
        ax2 = axes[1]
        core_features = self.feature_scores[
            (self.feature_scores['stability_score'] > 0.7) & 
            (self.feature_scores['discriminative_power'] > 0.5) &
            (self.feature_scores['shap_risk_flag'] == False)
        ].head(5)
        
        for i, (_, row) in enumerate(core_features.iterrows()):
            values = [row['stability_score'], row['discriminative_power'], row['interpretability_score']]
            values += values[:1]
            
            ax2.plot(angles, values, 'o-', linewidth=2, label=row['feature'][:15], alpha=0.7)
            ax2.fill(angles, values, alpha=0.1)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 1)
        ax2.set_title('核心保留特征雷达图', size=14, fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)
        
        # 3. 不稳定特征雷达图
        ax3 = axes[2]
        unstable_features = self.feature_scores[self.feature_scores['stability_score'] < 0.5].head(5)
        
        for i, (_, row) in enumerate(unstable_features.iterrows()):
            values = [row['stability_score'], row['discriminative_power'], row['interpretability_score']]
            values += values[:1]
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=row['feature'][:15], alpha=0.7)
            ax3.fill(angles, values, alpha=0.1)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 1)
        ax3.set_title('不稳定特征雷达图', size=14, fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax3.grid(True)
        
        # 4. 综合评分最高特征雷达图
        ax4 = axes[3]
        top_overall = self.feature_scores.head(5)
        
        for i, (_, row) in enumerate(top_overall.iterrows()):
            values = [row['stability_score'], row['discriminative_power'], row['interpretability_score']]
            values += values[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=row['feature'][:15], alpha=0.7)
            ax4.fill(angles, values, alpha=0.1)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('综合评分最高特征雷达图', size=14, fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存雷达图
        radar_file = self.output_dir / 'feature_risk_radar_charts.png'
        plt.savefig(radar_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 特征风险雷达图已保存: {radar_file}")
        
        return radar_file

    def create_comprehensive_visualization(self):
        """创建综合可视化分析"""
        print("\n📊 生成综合三维分析可视化...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. 三维散点图
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.feature_scores['stability_score'],
                             self.feature_scores['discriminative_power'],
                             c=self.feature_scores['interpretability_score'],
                             s=60, alpha=0.7, cmap='RdYlGn')
        ax1.set_xlabel('稳定性评分')
        ax1.set_ylabel('判别力评分')
        ax1.set_title('三维特征评分散点图')
        plt.colorbar(scatter, ax=ax1, label='可解释性评分')
        ax1.grid(True, alpha=0.3)

        # 2. 特征策略分布饼图
        ax2 = axes[0, 1]
        strategy_counts = {
            '保留核心': len(self.feature_strategies['keep_core']),
            '移除高风险': len(self.feature_strategies['remove_risky']),
            '组合弱特征': len(self.feature_strategies['combine_weak']),
            '规则控制': len(self.feature_strategies['rule_control']),
            '监控不稳定': len(self.feature_strategies['monitor_unstable'])
        }

        colors = ['green', 'red', 'orange', 'blue', 'purple']
        ax2.pie(strategy_counts.values(), labels=strategy_counts.keys(),
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('特征处理策略分布')

        # 3. 风险评分分布
        ax3 = axes[0, 2]
        risk_categories = []
        for _, row in self.feature_scores.iterrows():
            if row['shap_risk_flag']:
                risk_categories.append('SHAP高风险')
            elif row['overall_score'] > 0.7:
                risk_categories.append('低风险')
            elif row['overall_score'] > 0.4:
                risk_categories.append('中等风险')
            else:
                risk_categories.append('高风险')

        risk_counts = pd.Series(risk_categories).value_counts()
        ax3.bar(risk_counts.index, risk_counts.values,
               color=['red', 'orange', 'yellow', 'green'])
        ax3.set_title('特征风险等级分布')
        ax3.set_ylabel('特征数量')
        plt.setp(ax3.get_xticklabels(), rotation=45)

        # 4. 特征类型vs评分
        ax4 = axes[1, 0]
        feature_types = self.feature_scores['feature_type'].unique()
        type_scores = []

        for ftype in feature_types:
            type_data = self.feature_scores[self.feature_scores['feature_type'] == ftype]
            type_scores.append([
                type_data['stability_score'].mean(),
                type_data['discriminative_power'].mean(),
                type_data['interpretability_score'].mean()
            ])

        x = np.arange(len(feature_types))
        width = 0.25

        ax4.bar(x - width, [scores[0] for scores in type_scores], width,
               label='稳定性', alpha=0.8)
        ax4.bar(x, [scores[1] for scores in type_scores], width,
               label='判别力', alpha=0.8)
        ax4.bar(x + width, [scores[2] for scores in type_scores], width,
               label='可解释性', alpha=0.8)

        ax4.set_xlabel('特征类型')
        ax4.set_ylabel('平均评分')
        ax4.set_title('不同类型特征的三维评分对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels(feature_types, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 综合评分排名
        ax5 = axes[1, 1]
        top_15 = self.feature_scores.head(15)
        y_pos = range(len(top_15))

        bars = ax5.barh(y_pos, top_15['overall_score'], alpha=0.7)

        # 根据风险标记颜色
        for i, (_, row) in enumerate(top_15.iterrows()):
            if row['shap_risk_flag']:
                bars[i].set_color('red')
            elif row['overall_score'] > 0.7:
                bars[i].set_color('green')
            else:
                bars[i].set_color('orange')

        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(top_15['feature'], fontsize=9)
        ax5.set_xlabel('综合评分')
        ax5.set_title('前15个特征综合评分排名')
        ax5.grid(True, alpha=0.3)
        ax5.invert_yaxis()

        # 6. 稳定性vs判别力象限图
        ax6 = axes[1, 2]

        # 根据SHAP风险标记颜色
        colors = ['red' if risk else 'blue' for risk in self.feature_scores['shap_risk_flag']]

        ax6.scatter(self.feature_scores['stability_score'],
                   self.feature_scores['discriminative_power'],
                   c=colors, alpha=0.6, s=50)

        # 添加象限分割线
        ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax6.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)

        # 添加象限标签
        ax6.text(0.75, 0.75, '理想区域\n(高稳定+高判别)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax6.text(0.25, 0.75, '需要工程\n(低稳定+高判别)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        ax6.text(0.75, 0.25, '需要组合\n(高稳定+低判别)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax6.text(0.25, 0.25, '考虑移除\n(低稳定+低判别)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

        ax6.set_xlabel('稳定性评分')
        ax6.set_ylabel('判别力评分')
        ax6.set_title('稳定性vs判别力象限分析\n(红色=SHAP高风险)')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存综合可视化
        viz_file = self.output_dir / 'three_dimensional_comprehensive_analysis.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ 综合三维分析可视化已保存: {viz_file}")

        return viz_file

    def generate_final_report(self):
        """生成最终三维评估报告"""
        print("\n📄 生成三维特征评估最终报告...")

        report_file = self.output_dir / 'three_dimensional_feature_assessment_report.md'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 三维特征评估最终报告\n\n")
            f.write("## 🎯 评估维度\n\n")
            f.write("本报告基于三个核心维度对每个特征进行全面评估：\n\n")
            f.write("1. **稳定性 (Stability)**: 基于KS检验的特征漂移分析\n")
            f.write("2. **判别力 (Discriminative Power)**: 基于效应大小的黑白样本分离能力\n")
            f.write("3. **可解释性 (Interpretability)**: 基于SHAP分析的模型决策可解释性\n\n")

            f.write("## 📊 评估结果统计\n\n")
            f.write(f"- **总特征数**: {len(self.feature_scores)}\n")
            f.write(f"- **SHAP高风险特征**: {self.feature_scores['shap_risk_flag'].sum()}\n")
            f.write(f"- **高稳定性特征** (>0.7): {len(self.feature_scores[self.feature_scores['stability_score'] > 0.7])}\n")
            f.write(f"- **高判别力特征** (>0.6): {len(self.feature_scores[self.feature_scores['discriminative_power'] > 0.6])}\n")
            f.write(f"- **高可解释性特征** (>0.7): {len(self.feature_scores[self.feature_scores['interpretability_score'] > 0.7])}\n\n")

            f.write("## 🏆 三维评分最优特征 (Top 10)\n\n")
            top_10 = self.feature_scores.head(10)
            for i, (_, row) in enumerate(top_10.iterrows()):
                f.write(f"{i+1}. **`{row['feature']}`** (综合: {row['overall_score']:.3f})\n")
                f.write(f"   - 稳定性: {row['stability_score']:.3f} | ")
                f.write(f"判别力: {row['discriminative_power']:.3f} | ")
                f.write(f"可解释性: {row['interpretability_score']:.3f}\n")
                f.write(f"   - 特征类型: {row['feature_type']}\n\n")

            f.write("## 💡 特征处理策略\n\n")

            # 保留核心特征
            f.write("### ✅ 保留核心特征\n")
            f.write("以下特征三维评分优秀，建议作为核心特征保留：\n")
            for item in self.feature_strategies['keep_core'][:15]:
                f.write(f"- **`{item['feature']}`**: {item['reason']} ({item['scores']})\n")
            f.write("\n")

            # 移除高风险特征
            f.write("### 🗑️ 移除高风险特征\n")
            f.write("以下特征存在SHAP风险或可解释性问题，建议移除：\n")
            for item in self.feature_strategies['remove_risky']:
                f.write(f"- **`{item['feature']}`**: {item['reason']} ({item['scores']})\n")
            f.write("\n")

            # 组合弱特征
            f.write("### 🔗 组合弱特征\n")
            f.write("以下特征判别力较弱，建议进行特征组合：\n")
            for item in self.feature_strategies['combine_weak'][:10]:
                f.write(f"- **`{item['feature']}`**: {item['reason']} ({item['scores']})\n")
            f.write("\n")

            # 规则控制特征
            f.write("### 📋 规则控制特征\n")
            f.write("以下特征适合通过规则进行控制：\n")
            for item in self.feature_strategies['rule_control'][:10]:
                f.write(f"- **`{item['feature']}`**: {item['reason']} ({item['scores']})\n")
            f.write("\n")

            # 监控不稳定特征
            f.write("### 👁️ 监控不稳定特征\n")
            f.write("以下特征稳定性较差，需要持续监控：\n")
            for item in self.feature_strategies['monitor_unstable'][:10]:
                f.write(f"- **`{item['feature']}`**: {item['reason']} ({item['scores']})\n")
            f.write("\n")

            f.write("## 🎯 实施建议\n\n")
            f.write("### 短期行动 (1-2周)\n")
            f.write("1. **立即移除** SHAP高风险特征\n")
            f.write("2. **重点保留** 三维评分优秀的核心特征\n")
            f.write("3. **建立监控** 不稳定特征的漂移情况\n\n")

            f.write("### 中期优化 (1-2月)\n")
            f.write("1. **特征组合** 弱判别力特征，提升整体效果\n")
            f.write("2. **规则引擎** 对适合的特征建立规则控制机制\n")
            f.write("3. **A/B测试** 验证特征策略的实际效果\n\n")

            f.write("### 长期维护 (持续)\n")
            f.write("1. **定期评估** 三维特征评分的变化\n")
            f.write("2. **动态调整** 特征处理策略\n")
            f.write("3. **持续监控** SHAP可解释性和特征稳定性\n\n")

            f.write("## 📁 生成文件\n\n")
            f.write("- `three_dimensional_feature_scores.csv`: 详细三维评分数据\n")
            f.write("- `feature_processing_strategies.json`: 特征处理策略\n")
            f.write("- `feature_risk_radar_charts.png`: 特征风险雷达图\n")
            f.write("- `three_dimensional_comprehensive_analysis.png`: 综合分析可视化\n")
            f.write("- `three_dimensional_feature_assessment_report.md`: 本报告\n")

        print(f"✅ 三维特征评估报告已保存: {report_file}")

        return report_file

    def run_three_dimensional_assessment(self):
        """运行完整的三维特征评估"""
        print("🎯 三维特征评估系统")
        print("=" * 70)
        print("评估维度: 解释性 + 稳定性 + 判别力")
        print("=" * 70)

        # 1. 加载数据
        if not self.load_comprehensive_data():
            return False

        # 2. 计算三维评分
        feature_scores = self.calculate_three_dimensional_scores()

        # 3. 生成处理策略
        strategies = self.generate_feature_strategies()

        # 4. 创建风险雷达图
        self.create_feature_risk_radar()

        # 5. 创建综合可视化
        self.create_comprehensive_visualization()

        # 6. 生成最终报告
        self.generate_final_report()

        print(f"\n🎉 三维特征评估完成！")
        print(f"📁 所有结果已保存到 featuretrain/ 文件夹")
        print(f"🎯 特征处理策略已生成，可直接用于模型优化")

        return True

def main():
    """主函数"""
    assessor = ThreeDimensionalFeatureAssessment()
    success = assessor.run_three_dimensional_assessment()

    if success:
        print("\n✅ 三维特征评估系统运行完成！")
        print("💡 请查看生成的雷达图和报告获取详细的特征处理建议")
    else:
        print("\n❌ 三维特征评估失败！")

if __name__ == "__main__":
    main()
