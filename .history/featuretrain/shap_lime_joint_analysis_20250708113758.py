#!/usr/bin/env python3
"""
SHAP-LIME联合分析优化策略
结合SHAP全局可解释性和LIME局部可解释性
为特征优化提供更全面、更可靠的策略建议
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular
from scipy import stats
import json

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class SHAPLIMEJointAnalysis:
    def __init__(self):
        """初始化SHAP-LIME联合分析系统"""
        self.output_dir = Path('featuretrain')
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据存储
        self.train_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
        # 模型和解释器
        self.model = None
        self.shap_explainer = None
        self.lime_explainer = None
        
        # 分析结果
        self.shap_values = None
        self.lime_explanations = []
        self.joint_insights = {}
        
    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("📊 加载并准备训练数据...")
        
        try:
            # 加载train.csv数据
            self.train_data = pd.read_csv('train.csv')
            
            # 准备特征和标签
            self.feature_names = self.train_data.columns[1:].tolist()  # 排除文件名列
            X = self.train_data[self.feature_names].values
            
            # 创建标签 (前2939个是良性=0，后面是恶意=1)
            y = np.concatenate([np.zeros(2939), np.ones(len(self.train_data) - 2939)])
            
            # 数据分割
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 数据标准化
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            
            print(f"✅ 数据准备完成:")
            print(f"   - 训练集: {len(self.X_train)} 样本")
            print(f"   - 测试集: {len(self.X_test)} 样本")
            print(f"   - 特征数: {len(self.feature_names)}")
            print(f"   - 良性样本: {int(np.sum(self.y_train == 0))} / 恶意样本: {int(np.sum(self.y_train == 1))}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据准备失败: {e}")
            return False
    
    def train_model(self):
        """训练基础模型"""
        print("\n🤖 训练随机森林模型...")
        
        try:
            # 使用随机森林作为基础模型
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(self.X_train, self.y_train)
            
            # 评估模型性能
            train_score = self.model.score(self.X_train, self.y_train)
            test_score = self.model.score(self.X_test, self.y_test)
            
            print(f"✅ 模型训练完成:")
            print(f"   - 训练准确率: {train_score:.4f}")
            print(f"   - 测试准确率: {test_score:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            return False
    
    def perform_shap_analysis(self, sample_size=1000):
        """执行SHAP分析"""
        print(f"\n🔍 执行SHAP全局可解释性分析 (样本数: {sample_size})...")
        
        try:
            # 创建SHAP解释器
            self.shap_explainer = shap.TreeExplainer(self.model)
            
            # 选择样本进行SHAP分析（为了计算效率）
            sample_indices = np.random.choice(len(self.X_test), min(sample_size, len(self.X_test)), replace=False)
            X_sample = self.X_test[sample_indices]
            
            # 计算SHAP值
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            
            # 如果是二分类，取正类的SHAP值
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # 恶意类的SHAP值
            
            print(f"✅ SHAP分析完成:")
            print(f"   - 分析样本数: {len(X_sample)}")
            print(f"   - SHAP值形状: {self.shap_values.shape}")
            
            return X_sample, sample_indices
            
        except Exception as e:
            print(f"❌ SHAP分析失败: {e}")
            return None, None
    
    def perform_lime_analysis(self, sample_indices, num_samples=100):
        """执行LIME分析"""
        print(f"\n🔍 执行LIME局部可解释性分析 (样本数: {num_samples})...")
        
        try:
            # 创建LIME解释器
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                class_names=['Benign', 'Malicious'],
                mode='classification',
                discretize_continuous=True
            )
            
            # 选择样本进行LIME分析
            lime_sample_indices = np.random.choice(sample_indices, min(num_samples, len(sample_indices)), replace=False)
            
            self.lime_explanations = []
            
            for i, idx in enumerate(lime_sample_indices):
                if i % 20 == 0:
                    print(f"    处理LIME样本: {i+1}/{len(lime_sample_indices)}")
                
                # 获取LIME解释
                explanation = self.lime_explainer.explain_instance(
                    self.X_test[idx],
                    self.model.predict_proba,
                    num_features=len(self.feature_names),
                    top_labels=1
                )
                
                # 提取特征重要性
                lime_weights = {}
                for feature_idx, weight in explanation.as_list():
                    # 解析特征名称（LIME返回的是特征名称字符串）
                    feature_name = feature_idx.split('<=')[0].split('>')[0].strip()
                    if feature_name in self.feature_names:
                        lime_weights[feature_name] = weight
                    else:
                        # 如果无法匹配，使用索引
                        try:
                            feat_idx = int(feature_idx.split('_')[-1]) if '_' in feature_idx else 0
                            if feat_idx < len(self.feature_names):
                                lime_weights[self.feature_names[feat_idx]] = weight
                        except:
                            continue
                
                self.lime_explanations.append({
                    'sample_idx': idx,
                    'prediction': self.model.predict_proba([self.X_test[idx]])[0],
                    'true_label': self.y_test[idx],
                    'lime_weights': lime_weights
                })
            
            print(f"✅ LIME分析完成:")
            print(f"   - 分析样本数: {len(self.lime_explanations)}")
            
            return True
            
        except Exception as e:
            print(f"❌ LIME分析失败: {e}")
            return False
    
    def analyze_shap_lime_consistency(self):
        """分析SHAP和LIME的一致性"""
        print("\n🔄 分析SHAP-LIME一致性...")
        
        try:
            # 计算SHAP全局重要性
            shap_importance = np.abs(self.shap_values).mean(axis=0)
            shap_feature_importance = dict(zip(self.feature_names, shap_importance))
            
            # 计算LIME平均重要性
            lime_feature_importance = {}
            for feature in self.feature_names:
                weights = []
                for exp in self.lime_explanations:
                    if feature in exp['lime_weights']:
                        weights.append(abs(exp['lime_weights'][feature]))
                lime_feature_importance[feature] = np.mean(weights) if weights else 0
            
            # 计算一致性指标
            consistency_analysis = []
            
            for feature in self.feature_names:
                shap_imp = shap_feature_importance.get(feature, 0)
                lime_imp = lime_feature_importance.get(feature, 0)
                
                # 归一化重要性分数
                shap_norm = shap_imp / max(shap_feature_importance.values()) if max(shap_feature_importance.values()) > 0 else 0
                lime_norm = lime_imp / max(lime_feature_importance.values()) if max(lime_feature_importance.values()) > 0 else 0
                
                # 计算一致性分数
                consistency_score = 1 - abs(shap_norm - lime_norm)
                
                # 计算重要性等级
                shap_rank = sorted(shap_feature_importance.values(), reverse=True).index(shap_imp) + 1
                lime_rank = sorted(lime_feature_importance.values(), reverse=True).index(lime_imp) + 1
                rank_diff = abs(shap_rank - lime_rank)
                
                consistency_analysis.append({
                    'feature': feature,
                    'shap_importance': shap_imp,
                    'lime_importance': lime_imp,
                    'shap_normalized': shap_norm,
                    'lime_normalized': lime_norm,
                    'consistency_score': consistency_score,
                    'shap_rank': shap_rank,
                    'lime_rank': lime_rank,
                    'rank_difference': rank_diff,
                    'agreement_level': self.classify_agreement(consistency_score, rank_diff)
                })
            
            self.consistency_df = pd.DataFrame(consistency_analysis)
            self.consistency_df = self.consistency_df.sort_values('consistency_score', ascending=False)
            
            # 保存一致性分析结果
            consistency_file = self.output_dir / 'shap_lime_consistency_analysis.csv'
            self.consistency_df.to_csv(consistency_file, index=False, encoding='utf-8')
            
            print(f"✅ 一致性分析完成: {consistency_file}")
            print(f"   - 高一致性特征 (>0.8): {len(self.consistency_df[self.consistency_df['consistency_score'] > 0.8])}")
            print(f"   - 中等一致性特征 (0.6-0.8): {len(self.consistency_df[(self.consistency_df['consistency_score'] > 0.6) & (self.consistency_df['consistency_score'] <= 0.8)])}")
            print(f"   - 低一致性特征 (<0.6): {len(self.consistency_df[self.consistency_df['consistency_score'] <= 0.6])}")
            
            return True
            
        except Exception as e:
            print(f"❌ 一致性分析失败: {e}")
            return False
    
    def classify_agreement(self, consistency_score, rank_diff):
        """分类一致性等级"""
        if consistency_score > 0.8 and rank_diff < 10:
            return "高度一致"
        elif consistency_score > 0.6 and rank_diff < 20:
            return "中等一致"
        elif consistency_score > 0.4:
            return "部分一致"
        else:
            return "不一致"
    
    def generate_joint_optimization_strategy(self):
        """生成SHAP-LIME联合优化策略"""
        print("\n💡 生成SHAP-LIME联合优化策略...")
        
        try:
            # 加载之前的三维评估结果
            three_dim_scores = pd.read_csv('featuretrain/three_dimensional_feature_scores.csv')
            
            # 合并分析结果
            joint_analysis = []
            
            for _, row in self.consistency_df.iterrows():
                feature = row['feature']
                
                # 获取三维评分
                three_dim_row = three_dim_scores[three_dim_scores['feature'] == feature]
                if not three_dim_row.empty:
                    stability = three_dim_row['stability_score'].iloc[0]
                    discriminative = three_dim_row['discriminative_power'].iloc[0]
                    interpretability = three_dim_row['interpretability_score'].iloc[0]
                    overall_score = three_dim_row['overall_score'].iloc[0]
                else:
                    stability = discriminative = interpretability = overall_score = 0
                
                # 计算联合可信度分数
                # 结合SHAP-LIME一致性和三维评分
                joint_reliability = (
                    row['consistency_score'] * 0.4 +  # SHAP-LIME一致性
                    interpretability * 0.3 +          # 原有可解释性
                    stability * 0.2 +                 # 稳定性
                    discriminative * 0.1               # 判别力
                )
                
                # 生成优化策略
                strategy = self.determine_joint_strategy(
                    row['consistency_score'],
                    row['shap_importance'],
                    row['lime_importance'],
                    stability,
                    discriminative,
                    interpretability,
                    joint_reliability
                )
                
                joint_analysis.append({
                    'feature': feature,
                    'shap_importance': row['shap_importance'],
                    'lime_importance': row['lime_importance'],
                    'consistency_score': row['consistency_score'],
                    'agreement_level': row['agreement_level'],
                    'stability_score': stability,
                    'discriminative_power': discriminative,
                    'interpretability_score': interpretability,
                    'joint_reliability': joint_reliability,
                    'optimization_strategy': strategy['action'],
                    'strategy_reason': strategy['reason'],
                    'priority_level': strategy['priority'],
                    'confidence_level': strategy['confidence']
                })
            
            self.joint_strategy_df = pd.DataFrame(joint_analysis)
            self.joint_strategy_df = self.joint_strategy_df.sort_values('joint_reliability', ascending=False)
            
            # 保存联合策略
            strategy_file = self.output_dir / 'shap_lime_joint_optimization_strategy.csv'
            self.joint_strategy_df.to_csv(strategy_file, index=False, encoding='utf-8')
            
            print(f"✅ 联合优化策略生成完成: {strategy_file}")
            
            # 统计策略分布
            strategy_counts = self.joint_strategy_df['optimization_strategy'].value_counts()
            print(f"📊 策略分布:")
            for strategy, count in strategy_counts.items():
                print(f"   - {strategy}: {count} 个特征")
            
            return True
            
        except Exception as e:
            print(f"❌ 联合策略生成失败: {e}")
            return False
    
    def determine_joint_strategy(self, consistency, shap_imp, lime_imp, stability, discriminative, interpretability, joint_reliability):
        """确定联合优化策略"""
        
        # 高可信度特征
        if joint_reliability > 0.8 and consistency > 0.7:
            if discriminative > 0.6:
                return {
                    'action': '核心保留',
                    'reason': 'SHAP-LIME高度一致且性能优秀',
                    'priority': '高',
                    'confidence': '很高'
                }
            else:
                return {
                    'action': '稳定保留',
                    'reason': 'SHAP-LIME一致但判别力一般',
                    'priority': '中',
                    'confidence': '高'
                }
        
        # 不一致但重要的特征
        elif consistency < 0.5 and (shap_imp > 0.1 or lime_imp > 0.1):
            return {
                'action': '深度分析',
                'reason': 'SHAP-LIME不一致但显示重要性，需要深入调查',
                'priority': '高',
                'confidence': '中'
            }
        
        # 一致性好但性能差的特征
        elif consistency > 0.7 and discriminative < 0.3:
            return {
                'action': '特征工程',
                'reason': 'SHAP-LIME一致但判别力弱，适合组合或变换',
                'priority': '中',
                'confidence': '高'
            }
        
        # 稳定性差的特征
        elif stability < 0.5:
            return {
                'action': '监控或移除',
                'reason': '稳定性差，存在漂移风险',
                'priority': '高',
                'confidence': '高'
            }
        
        # 低重要性特征
        elif shap_imp < 0.01 and lime_imp < 0.01:
            return {
                'action': '考虑移除',
                'reason': 'SHAP-LIME均显示低重要性',
                'priority': '低',
                'confidence': '高'
            }
        
        # 中等表现特征
        else:
            return {
                'action': '保持观察',
                'reason': '表现中等，继续观察',
                'priority': '低',
                'confidence': '中'
            }
    
    def create_joint_visualizations(self):
        """创建SHAP-LIME联合可视化"""
        print("\n📈 生成SHAP-LIME联合可视化...")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # 1. SHAP vs LIME重要性散点图
            ax1 = axes[0, 0]
            scatter = ax1.scatter(self.consistency_df['shap_normalized'], 
                                 self.consistency_df['lime_normalized'],
                                 c=self.consistency_df['consistency_score'],
                                 s=60, alpha=0.7, cmap='RdYlGn')
            ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='完全一致线')
            ax1.set_xlabel('SHAP重要性 (归一化)')
            ax1.set_ylabel('LIME重要性 (归一化)')
            ax1.set_title('SHAP vs LIME重要性对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='一致性分数')
            
            # 2. 一致性分数分布
            ax2 = axes[0, 1]
            ax2.hist(self.consistency_df['consistency_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('一致性分数')
            ax2.set_ylabel('特征数量')
            ax2.set_title('SHAP-LIME一致性分布')
            ax2.grid(True, alpha=0.3)
            
            # 3. 一致性等级饼图
            ax3 = axes[0, 2]
            agreement_counts = self.consistency_df['agreement_level'].value_counts()
            colors = ['green', 'yellow', 'orange', 'red']
            ax3.pie(agreement_counts.values, labels=agreement_counts.index, 
                   autopct='%1.1f%%', colors=colors[:len(agreement_counts)], startangle=90)
            ax3.set_title('SHAP-LIME一致性等级分布')
            
            # 4. 联合可信度vs一致性
            ax4 = axes[1, 0]
            ax4.scatter(self.joint_strategy_df['consistency_score'],
                       self.joint_strategy_df['joint_reliability'],
                       alpha=0.6, s=50)
            ax4.set_xlabel('SHAP-LIME一致性')
            ax4.set_ylabel('联合可信度分数')
            ax4.set_title('一致性 vs 联合可信度')
            ax4.grid(True, alpha=0.3)
            
            # 5. 优化策略分布
            ax5 = axes[1, 1]
            strategy_counts = self.joint_strategy_df['optimization_strategy'].value_counts()
            bars = ax5.bar(range(len(strategy_counts)), strategy_counts.values, alpha=0.7)
            ax5.set_xticks(range(len(strategy_counts)))
            ax5.set_xticklabels(strategy_counts.index, rotation=45, ha='right')
            ax5.set_ylabel('特征数量')
            ax5.set_title('SHAP-LIME联合优化策略分布')
            ax5.grid(True, alpha=0.3)
            
            # 为不同策略设置不同颜色
            colors = ['green', 'blue', 'orange', 'red', 'purple', 'brown']
            for i, bar in enumerate(bars):
                bar.set_color(colors[i % len(colors)])
            
            # 6. 前15个高可信度特征
            ax6 = axes[1, 2]
            top_15 = self.joint_strategy_df.head(15)
            y_pos = range(len(top_15))
            
            bars = ax6.barh(y_pos, top_15['joint_reliability'], alpha=0.7)
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(top_15['feature'], fontsize=9)
            ax6.set_xlabel('联合可信度分数')
            ax6.set_title('前15个高可信度特征')
            ax6.grid(True, alpha=0.3)
            ax6.invert_yaxis()
            
            # 根据策略设置颜色
            strategy_colors = {'核心保留': 'green', '稳定保留': 'blue', '深度分析': 'orange', 
                             '特征工程': 'yellow', '监控或移除': 'red', '考虑移除': 'darkred',
                             '保持观察': 'gray'}
            for i, (_, row) in enumerate(top_15.iterrows()):
                bars[i].set_color(strategy_colors.get(row['optimization_strategy'], 'gray'))
            
            plt.tight_layout()
            
            # 保存可视化
            viz_file = self.output_dir / 'shap_lime_joint_analysis_visualization.png'
            plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ 联合可视化已保存: {viz_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 可视化生成失败: {e}")
            return False

    def create_shap_summary_plots(self):
        """创建SHAP摘要图"""
        print("\n📊 生成SHAP摘要可视化...")

        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # 1. SHAP特征重要性条形图
            shap_importance = np.abs(self.shap_values).mean(axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': shap_importance
            }).sort_values('importance', ascending=False)

            top_20 = feature_importance_df.head(20)

            ax1 = axes[0]
            bars = ax1.barh(range(len(top_20)), top_20['importance'], alpha=0.7, color='skyblue')
            ax1.set_yticks(range(len(top_20)))
            ax1.set_yticklabels(top_20['feature'], fontsize=10)
            ax1.set_xlabel('SHAP重要性分数')
            ax1.set_title('SHAP特征重要性排名 (Top 20)')
            ax1.grid(True, alpha=0.3)
            ax1.invert_yaxis()

            # 2. SHAP值分布小提琴图
            ax2 = axes[1]

            # 选择前10个最重要的特征进行可视化
            top_10_features = top_20.head(10)['feature'].tolist()
            top_10_indices = [self.feature_names.index(f) for f in top_10_features]

            shap_data_for_violin = []
            labels_for_violin = []

            for i, feature_idx in enumerate(top_10_indices):
                shap_data_for_violin.append(self.shap_values[:, feature_idx])
                labels_for_violin.append(top_10_features[i][:15])  # 截断长特征名

            parts = ax2.violinplot(shap_data_for_violin, positions=range(len(shap_data_for_violin)),
                                  showmeans=True, showmedians=True)

            ax2.set_xticks(range(len(labels_for_violin)))
            ax2.set_xticklabels(labels_for_violin, rotation=45, ha='right')
            ax2.set_ylabel('SHAP值分布')
            ax2.set_title('前10个特征的SHAP值分布')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)

            plt.tight_layout()

            # 保存SHAP摘要图
            shap_file = self.output_dir / 'shap_summary_plots.png'
            plt.savefig(shap_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✅ SHAP摘要图已保存: {shap_file}")

            return True

        except Exception as e:
            print(f"❌ SHAP摘要图生成失败: {e}")
            return False

    def generate_comprehensive_report(self):
        """生成SHAP-LIME联合分析综合报告"""
        print("\n📄 生成SHAP-LIME联合分析报告...")

        try:
            report_file = self.output_dir / 'shap_lime_joint_analysis_report.md'

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# SHAP-LIME联合分析优化策略报告\n\n")

                f.write("## 🎯 分析目标\n\n")
                f.write("本报告结合SHAP全局可解释性和LIME局部可解释性分析，")
                f.write("通过两种方法的一致性验证，为特征优化提供更可靠的策略建议。\n\n")

                f.write("## 📊 方法论\n\n")
                f.write("### SHAP (SHapley Additive exPlanations)\n")
                f.write("- **优势**: 提供全局一致的特征重要性，理论基础扎实\n")
                f.write("- **应用**: 分析特征对模型整体决策的贡献\n")
                f.write("- **样本数**: 1000个测试样本\n\n")

                f.write("### LIME (Local Interpretable Model-agnostic Explanations)\n")
                f.write("- **优势**: 提供局部可解释性，关注个体样本的决策过程\n")
                f.write("- **应用**: 验证SHAP结果的局部一致性\n")
                f.write("- **样本数**: 100个代表性样本\n\n")

                f.write("### 联合分析策略\n")
                f.write("- **一致性验证**: 比较SHAP和LIME的特征重要性排名\n")
                f.write("- **可信度评估**: 结合一致性和三维特征评分\n")
                f.write("- **策略生成**: 基于联合分析结果制定优化策略\n\n")

                # 分析结果统计
                f.write("## 📈 分析结果统计\n\n")
                f.write(f"- **总特征数**: {len(self.consistency_df)}\n")

                high_consistency = len(self.consistency_df[self.consistency_df['consistency_score'] > 0.8])
                medium_consistency = len(self.consistency_df[(self.consistency_df['consistency_score'] > 0.6) &
                                                           (self.consistency_df['consistency_score'] <= 0.8)])
                low_consistency = len(self.consistency_df[self.consistency_df['consistency_score'] <= 0.6])

                f.write(f"- **高一致性特征** (>0.8): {high_consistency} ({high_consistency/len(self.consistency_df)*100:.1f}%)\n")
                f.write(f"- **中等一致性特征** (0.6-0.8): {medium_consistency} ({medium_consistency/len(self.consistency_df)*100:.1f}%)\n")
                f.write(f"- **低一致性特征** (<0.6): {low_consistency} ({low_consistency/len(self.consistency_df)*100:.1f}%)\n\n")

                # 一致性最高的特征
                f.write("## 🏆 SHAP-LIME高度一致特征 (Top 10)\n\n")
                top_consistent = self.consistency_df.head(10)
                for i, (_, row) in enumerate(top_consistent.iterrows()):
                    f.write(f"{i+1}. **`{row['feature']}`** (一致性: {row['consistency_score']:.3f})\n")
                    f.write(f"   - SHAP重要性: {row['shap_importance']:.4f} (排名: {row['shap_rank']})\n")
                    f.write(f"   - LIME重要性: {row['lime_importance']:.4f} (排名: {row['lime_rank']})\n")
                    f.write(f"   - 一致性等级: {row['agreement_level']}\n\n")

                # 联合优化策略
                f.write("## 💡 SHAP-LIME联合优化策略\n\n")

                strategy_counts = self.joint_strategy_df['optimization_strategy'].value_counts()

                for strategy, count in strategy_counts.items():
                    f.write(f"### {strategy} ({count}个特征)\n\n")

                    strategy_features = self.joint_strategy_df[
                        self.joint_strategy_df['optimization_strategy'] == strategy
                    ].head(5)  # 显示前5个

                    for _, row in strategy_features.iterrows():
                        f.write(f"- **`{row['feature']}`**: {row['strategy_reason']}\n")
                        f.write(f"  - 联合可信度: {row['joint_reliability']:.3f}\n")
                        f.write(f"  - 一致性: {row['consistency_score']:.3f}\n\n")

                f.write("## 📁 生成文件\n\n")
                f.write("- `shap_lime_consistency_analysis.csv`: SHAP-LIME一致性详细分析\n")
                f.write("- `shap_lime_joint_optimization_strategy.csv`: 联合优化策略\n")
                f.write("- `shap_lime_joint_analysis_visualization.png`: 联合分析可视化\n")
                f.write("- `shap_summary_plots.png`: SHAP摘要图\n")
                f.write("- `shap_lime_joint_analysis_report.md`: 本报告\n")

            print(f"✅ SHAP-LIME联合分析报告已保存: {report_file}")

            return True

        except Exception as e:
            print(f"❌ 报告生成失败: {e}")
            return False

    def run_joint_analysis(self):
        """运行完整的SHAP-LIME联合分析"""
        print("🎯 SHAP-LIME联合分析优化策略")
        print("=" * 70)
        print("结合全局可解释性(SHAP)和局部可解释性(LIME)")
        print("=" * 70)

        # 1. 数据准备
        if not self.load_and_prepare_data():
            return False

        # 2. 模型训练
        if not self.train_model():
            return False

        # 3. SHAP分析
        X_sample, sample_indices = self.perform_shap_analysis()
        if X_sample is None:
            return False

        # 4. LIME分析
        if not self.perform_lime_analysis(sample_indices):
            return False

        # 5. 一致性分析
        if not self.analyze_shap_lime_consistency():
            return False

        # 6. 联合优化策略
        if not self.generate_joint_optimization_strategy():
            return False

        # 7. 可视化
        if not self.create_joint_visualizations():
            return False

        # 8. SHAP摘要图
        if not self.create_shap_summary_plots():
            return False

        # 9. 综合报告
        if not self.generate_comprehensive_report():
            return False

        print(f"\n🎉 SHAP-LIME联合分析完成！")
        print(f"📁 所有结果已保存到 featuretrain/ 文件夹")
        print(f"🔍 联合分析提供了更可靠的特征优化策略")

        return True

def main():
    """主函数"""
    analyzer = SHAPLIMEJointAnalysis()
    success = analyzer.run_joint_analysis()

    if success:
        print("\n✅ SHAP-LIME联合分析完成！")
        print("💡 请查看生成的报告和可视化文件获取详细的优化建议")
    else:
        print("\n❌ SHAP-LIME联合分析失败！")

if __name__ == "__main__":
    main()
