#!/usr/bin/env python3
"""
检出率低原因分析
分析为什么黑样本数量多但检出率低的原因
包括数据质量、特征有效性、模型性能等多维度分析
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import json

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class DetectionRateAnalysis:
    def __init__(self):
        """初始化检出率分析"""
        self.output_dir = Path('featuretrain')
        
        # 数据存储
        self.train_data = None
        self.data_data = None
        self.feature_names = None
        
        # 分析结果
        self.analysis_results = {}
        
    def load_and_analyze_data_distribution(self):
        """加载并分析数据分布"""
        print("📊 分析数据分布和质量...")
        
        try:
            # 加载训练数据
            self.train_data = pd.read_csv('train.csv')
            
            # 加载测试数据
            try:
                self.data_data = pd.read_csv('data.csv')
                print(f"✅ 数据加载完成:")
                print(f"   - 训练数据: {len(self.train_data)} 样本")
                print(f"   - 测试数据: {len(self.data_data)} 样本")
            except:
                print(f"⚠️  data.csv未找到，仅分析train.csv")
                self.data_data = None
            
            # 分析训练数据分布
            print(f"\n📈 训练数据分析:")
            print(f"   - 总样本数: {len(self.train_data)}")
            
            # 根据文件名分析标签分布
            # 假设前2939个是良性，后面是恶意
            benign_count = 2939
            malicious_count = len(self.train_data) - benign_count
            
            print(f"   - 良性样本: {benign_count} ({benign_count/len(self.train_data)*100:.1f}%)")
            print(f"   - 恶意样本: {malicious_count} ({malicious_count/len(self.train_data)*100:.1f}%)")
            
            # 检查数据不平衡问题
            imbalance_ratio = malicious_count / benign_count
            print(f"   - 不平衡比例: {imbalance_ratio:.2f} (恶意/良性)")
            
            if imbalance_ratio > 2:
                print(f"   ⚠️  数据严重不平衡！恶意样本过多可能导致模型偏向")
            
            # 分析特征统计
            self.feature_names = self.train_data.columns[1:].tolist()  # 排除文件名
            print(f"   - 特征数量: {len(self.feature_names)}")
            
            # 分析特征值分布
            feature_data = self.train_data[self.feature_names]
            
            # 检查零值特征
            zero_features = []
            low_variance_features = []
            
            for feature in self.feature_names:
                values = feature_data[feature]
                zero_ratio = (values == 0).sum() / len(values)
                variance = values.var()
                
                if zero_ratio > 0.95:
                    zero_features.append((feature, zero_ratio))
                elif variance < 0.01:
                    low_variance_features.append((feature, variance))
            
            print(f"   - 零值特征 (>95%为0): {len(zero_features)}")
            print(f"   - 低方差特征 (<0.01): {len(low_variance_features)}")
            
            # 保存分析结果
            self.analysis_results['data_distribution'] = {
                'total_samples': len(self.train_data),
                'benign_count': benign_count,
                'malicious_count': malicious_count,
                'imbalance_ratio': imbalance_ratio,
                'feature_count': len(self.feature_names),
                'zero_features_count': len(zero_features),
                'low_variance_features_count': len(low_variance_features),
                'zero_features': zero_features[:10],  # 保存前10个
                'low_variance_features': low_variance_features[:10]
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 数据分析失败: {e}")
            return False
    
    def analyze_feature_effectiveness(self):
        """分析特征有效性"""
        print("\n🔍 分析特征有效性...")
        
        try:
            # 准备数据
            feature_data = self.train_data[self.feature_names].values
            
            # 创建标签
            benign_count = 2939
            labels = np.concatenate([np.zeros(benign_count), np.ones(len(self.train_data) - benign_count)])
            
            # 分析良性和恶意样本的特征差异
            benign_features = feature_data[:benign_count]
            malicious_features = feature_data[benign_count:]
            
            feature_effectiveness = []
            
            for i, feature_name in enumerate(self.feature_names):
                benign_values = benign_features[:, i]
                malicious_values = malicious_features[:, i]
                
                # 计算统计差异
                benign_mean = np.mean(benign_values)
                malicious_mean = np.mean(malicious_values)
                benign_std = np.std(benign_values)
                malicious_std = np.std(malicious_values)
                
                # 计算效应大小 (Cohen's d)
                pooled_std = np.sqrt((benign_std**2 + malicious_std**2) / 2)
                cohens_d = abs(malicious_mean - benign_mean) / pooled_std if pooled_std > 0 else 0
                
                # 计算重叠度
                min_max_benign = (np.min(benign_values), np.max(benign_values))
                min_max_malicious = (np.min(malicious_values), np.max(malicious_values))
                
                # 计算分布重叠
                overlap_start = max(min_max_benign[0], min_max_malicious[0])
                overlap_end = min(min_max_benign[1], min_max_malicious[1])
                overlap_ratio = max(0, overlap_end - overlap_start) / max(min_max_benign[1] - min_max_benign[0], 
                                                                         min_max_malicious[1] - min_max_malicious[0])
                
                # 计算非零比例
                benign_nonzero_ratio = np.count_nonzero(benign_values) / len(benign_values)
                malicious_nonzero_ratio = np.count_nonzero(malicious_values) / len(malicious_values)
                
                feature_effectiveness.append({
                    'feature': feature_name,
                    'benign_mean': benign_mean,
                    'malicious_mean': malicious_mean,
                    'mean_difference': abs(malicious_mean - benign_mean),
                    'cohens_d': cohens_d,
                    'overlap_ratio': overlap_ratio,
                    'benign_nonzero_ratio': benign_nonzero_ratio,
                    'malicious_nonzero_ratio': malicious_nonzero_ratio,
                    'effectiveness_score': cohens_d * (1 - overlap_ratio) * max(benign_nonzero_ratio, malicious_nonzero_ratio)
                })
            
            self.feature_effectiveness_df = pd.DataFrame(feature_effectiveness)
            self.feature_effectiveness_df = self.feature_effectiveness_df.sort_values('effectiveness_score', ascending=False)
            
            # 保存特征有效性分析
            effectiveness_file = self.output_dir / 'feature_effectiveness_analysis.csv'
            self.feature_effectiveness_df.to_csv(effectiveness_file, index=False, encoding='utf-8')
            
            # 统计有效特征
            highly_effective = len(self.feature_effectiveness_df[self.feature_effectiveness_df['effectiveness_score'] > 0.5])
            moderately_effective = len(self.feature_effectiveness_df[
                (self.feature_effectiveness_df['effectiveness_score'] > 0.1) & 
                (self.feature_effectiveness_df['effectiveness_score'] <= 0.5)
            ])
            low_effective = len(self.feature_effectiveness_df[self.feature_effectiveness_df['effectiveness_score'] <= 0.1])
            
            print(f"✅ 特征有效性分析完成:")
            print(f"   - 高效特征 (>0.5): {highly_effective}")
            print(f"   - 中效特征 (0.1-0.5): {moderately_effective}")
            print(f"   - 低效特征 (≤0.1): {low_effective}")
            
            # 保存分析结果
            self.analysis_results['feature_effectiveness'] = {
                'highly_effective': highly_effective,
                'moderately_effective': moderately_effective,
                'low_effective': low_effective,
                'top_10_features': self.feature_effectiveness_df.head(10).to_dict('records')
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 特征有效性分析失败: {e}")
            return False
    
    def train_and_evaluate_model(self):
        """训练模型并评估性能"""
        print("\n🤖 训练模型并评估检出率...")
        
        try:
            # 准备数据
            feature_data = self.train_data[self.feature_names].values
            benign_count = 2939
            labels = np.concatenate([np.zeros(benign_count), np.ones(len(self.train_data) - benign_count)])
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                feature_data, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # 数据标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练多个模型进行对比
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'RandomForest_Balanced': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            }
            
            model_results = {}
            
            for model_name, model in models.items():
                print(f"   训练 {model_name}...")
                
                # 训练模型
                if 'Balanced' in model_name:
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # 预测
                if 'Balanced' in model_name:
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # 计算指标
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)  # 这就是检出率
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # 混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                # 计算更多指标
                false_positive_rate = fp / (fp + tn)
                false_negative_rate = fn / (fn + tp)
                
                model_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,  # 检出率
                    'f1_score': f1,
                    'auc': auc,
                    'false_positive_rate': false_positive_rate,
                    'false_negative_rate': false_negative_rate,
                    'confusion_matrix': cm.tolist(),
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn)
                }
                
                print(f"     - 准确率: {accuracy:.3f}")
                print(f"     - 检出率 (召回率): {recall:.3f}")
                print(f"     - 精确率: {precision:.3f}")
                print(f"     - F1分数: {f1:.3f}")
                print(f"     - AUC: {auc:.3f}")
                print(f"     - 误报率: {false_positive_rate:.3f}")
                print(f"     - 漏报率: {false_negative_rate:.3f}")
            
            # 保存模型评估结果
            self.analysis_results['model_performance'] = model_results
            
            return True, models, scaler, X_test, y_test
            
        except Exception as e:
            print(f"❌ 模型训练评估失败: {e}")
            return False, None, None, None, None
    
    def analyze_detection_issues(self):
        """分析检出率低的具体原因"""
        print("\n🔍 分析检出率低的具体原因...")
        
        try:
            issues = []
            
            # 1. 数据不平衡问题
            imbalance_ratio = self.analysis_results['data_distribution']['imbalance_ratio']
            if imbalance_ratio > 2:
                issues.append({
                    'issue': '数据严重不平衡',
                    'description': f'恶意样本是良性样本的{imbalance_ratio:.1f}倍，模型可能过度拟合恶意样本',
                    'severity': '高',
                    'solution': '使用类别平衡技术、重采样、或调整类别权重'
                })
            
            # 2. 无效特征过多
            zero_features_count = self.analysis_results['data_distribution']['zero_features_count']
            total_features = self.analysis_results['data_distribution']['feature_count']
            if zero_features_count > total_features * 0.3:
                issues.append({
                    'issue': '无效特征过多',
                    'description': f'{zero_features_count}/{total_features} 个特征几乎全为零值',
                    'severity': '中',
                    'solution': '移除零值特征，进行特征选择'
                })
            
            # 3. 特征区分度不足
            highly_effective = self.analysis_results['feature_effectiveness']['highly_effective']
            if highly_effective < 10:
                issues.append({
                    'issue': '高效特征不足',
                    'description': f'只有{highly_effective}个高效特征，区分能力不足',
                    'severity': '高',
                    'solution': '特征工程、特征组合、或收集更多有效特征'
                })
            
            # 4. 模型性能问题
            if 'model_performance' in self.analysis_results:
                best_recall = max([result['recall'] for result in self.analysis_results['model_performance'].values()])
                if best_recall < 0.7:
                    issues.append({
                        'issue': '模型检出率过低',
                        'description': f'最佳模型检出率仅为{best_recall:.3f}',
                        'severity': '高',
                        'solution': '尝试其他算法、调参、或集成学习'
                    })
            
            # 5. 特征重叠度高
            avg_overlap = np.mean([f['overlap_ratio'] for f in self.analysis_results['feature_effectiveness']['top_10_features']])
            if avg_overlap > 0.8:
                issues.append({
                    'issue': '特征分布重叠严重',
                    'description': f'良性和恶意样本特征重叠度达{avg_overlap:.3f}',
                    'severity': '高',
                    'solution': '寻找更具区分性的特征，或使用非线性模型'
                })
            
            print(f"✅ 发现 {len(issues)} 个主要问题:")
            for i, issue in enumerate(issues):
                print(f"   {i+1}. {issue['issue']} (严重程度: {issue['severity']})")
                print(f"      {issue['description']}")
                print(f"      建议: {issue['solution']}")
            
            self.analysis_results['detection_issues'] = issues
            
            return True
            
        except Exception as e:
            print(f"❌ 问题分析失败: {e}")
            return False
