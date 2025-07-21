#!/usr/bin/env python3
"""
智能后处理机制
基于KS检验和SHAP分析结果，对检测结果进行智能后处理
减少误报，提高检测准确性
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class IntelligentPostProcessor:
    def __init__(self, analysis_results_dir='./'):
        """
        初始化后处理器
        
        Args:
            analysis_results_dir: 分析结果文件所在目录
        """
        self.analysis_dir = Path(analysis_results_dir)
        self.load_analysis_results()
        self.build_post_processing_rules()
        
    def load_analysis_results(self):
        """加载所有分析结果"""
        print("📊 加载分析结果...")
        
        try:
            # 1. 加载整合分析结果
            self.integrated_results = pd.read_excel(
                self.analysis_dir / 'integrated_analysis_results.xlsx'
            )
            print("✅ 整合分析结果已加载")
            
            # 2. 加载KS检验结果
            self.ks_results = pd.read_excel(
                self.analysis_dir / 'comprehensive_ks_analysis.xlsx'
            )
            print("✅ KS检验结果已加载")
            
            # 3. 加载SHAP分析结果
            self.shap_results = pd.read_excel(
                self.analysis_dir / 'shap_pattern_analysis.xlsx'
            )
            print("✅ SHAP分析结果已加载")
            
            # 4. 加载特征工程建议
            self.recommendations = pd.read_excel(
                self.analysis_dir / 'feature_engineering_recommendations.xlsx'
            )
            print("✅ 特征工程建议已加载")
            
        except FileNotFoundError as e:
            print(f"❌ 缺少分析结果文件: {e}")
            print("请先运行完整的特征分析流程")
            raise
    
    def build_post_processing_rules(self):
        """基于分析结果构建后处理规则"""
        print("\n🔧 构建后处理规则...")
        
        # 1. 识别高风险特征（容易导致误报的特征）
        self.high_risk_features = self.integrated_results[
            self.integrated_results['risk_level'] == '高风险'
        ]['feature'].tolist()
        
        # 2. 识别可靠特征（区分度好的特征）
        # 检查列名是否存在
        if 'ks_white_black' in self.integrated_results.columns:
            ks_column = 'ks_white_black'
        elif 'ks_statistic_white_black' in self.integrated_results.columns:
            ks_column = 'ks_statistic_white_black'
        else:
            # 如果没有找到对应列，使用其他指标
            ks_column = None

        if ks_column:
            self.reliable_features = self.integrated_results[
                (self.integrated_results['risk_level'] == '低风险') &
                (self.integrated_results[ks_column] > 0.5)  # 白黑样本区分度好
            ]['feature'].tolist()
        else:
            # 备用方案：只使用风险等级
            self.reliable_features = self.integrated_results[
                self.integrated_results['risk_level'] == '低风险'
            ]['feature'].tolist()
        
        # 3. 构建特征权重调整规则
        self.feature_weights = {}
        for _, row in self.integrated_results.iterrows():
            feature = row['feature']
            risk_score = row['risk_score']
            
            if risk_score > 0.7:
                self.feature_weights[feature] = 0.3  # 高风险特征降权
            elif risk_score > 0.5:
                self.feature_weights[feature] = 0.6  # 中风险特征适度降权
            else:
                self.feature_weights[feature] = 1.0  # 正常权重
        
        # 4. 构建阈值调整规则
        self.threshold_adjustments = {}
        for _, row in self.recommendations.iterrows():
            if row['priority'] == '高':
                feature = row['feature']
                if '阈值' in row['recommendation']:
                    self.threshold_adjustments[feature] = 1.5  # 提高阈值
        
        print(f"✅ 识别高风险特征: {len(self.high_risk_features)} 个")
        print(f"✅ 识别可靠特征: {len(self.reliable_features)} 个")
        print(f"✅ 构建权重调整规则: {len(self.feature_weights)} 个")
    
    def post_process_prediction(self, features, original_prediction, original_probability):
        """
        对单个样本的预测结果进行后处理
        
        Args:
            features: dict, 特征值字典 {feature_name: value}
            original_prediction: int, 原始预测结果 (0=良性, 1=恶意)
            original_probability: float, 原始预测概率
            
        Returns:
            dict: 后处理结果
        """
        result = {
            'original_prediction': original_prediction,
            'original_probability': original_probability,
            'adjusted_prediction': original_prediction,
            'adjusted_probability': original_probability,
            'confidence_level': 'medium',
            'post_processing_actions': [],
            'risk_factors': [],
            'protective_factors': []
        }
        
        # 1. 计算调整后的风险评分
        adjusted_score = self._calculate_adjusted_risk_score(features)
        result['adjusted_risk_score'] = adjusted_score
        
        # 2. 应用特征权重调整
        weighted_score = self._apply_feature_weights(features, adjusted_score)
        result['weighted_score'] = weighted_score
        
        # 3. 检查高风险特征模式
        risk_factors = self._check_risk_patterns(features)
        result['risk_factors'] = risk_factors
        
        # 4. 检查保护性因素
        protective_factors = self._check_protective_patterns(features)
        result['protective_factors'] = protective_factors
        
        # 5. 综合决策
        final_decision = self._make_final_decision(
            weighted_score, risk_factors, protective_factors, original_probability
        )
        
        result['adjusted_prediction'] = final_decision['prediction']
        result['adjusted_probability'] = final_decision['probability']
        result['confidence_level'] = final_decision['confidence']
        result['post_processing_actions'] = final_decision['actions']
        
        return result
    
    def _calculate_adjusted_risk_score(self, features):
        """基于分析结果计算调整后的风险评分"""
        risk_score = 0.0
        total_weight = 0.0
        
        for feature_name, feature_value in features.items():
            if feature_name in self.integrated_results['feature'].values:
                # 获取该特征的分析结果
                feature_data = self.integrated_results[
                    self.integrated_results['feature'] == feature_name
                ].iloc[0]
                
                # 基于特征的风险评分和当前值计算贡献
                feature_risk = feature_data['risk_score']
                
                # 标准化特征值（简单的min-max标准化）
                if feature_value > 0:
                    normalized_value = min(feature_value / 10.0, 1.0)  # 简单标准化
                    contribution = feature_risk * normalized_value
                    
                    risk_score += contribution
                    total_weight += feature_risk
        
        return risk_score / (total_weight + 1e-8) if total_weight > 0 else 0.0
    
    def _apply_feature_weights(self, features, base_score):
        """应用特征权重调整"""
        weighted_score = base_score
        
        # 检查高风险特征的激活情况
        high_risk_activation = 0
        for feature in self.high_risk_features:
            if feature in features and features[feature] > 0:
                high_risk_activation += 1
        
        # 如果高风险特征大量激活，降低整体评分
        if high_risk_activation > len(self.high_risk_features) * 0.5:
            weighted_score *= 0.7  # 降低30%
        
        return weighted_score
    
    def _check_risk_patterns(self, features):
        """检查风险模式"""
        risk_factors = []
        
        # 1. 检查高风险特征组合
        high_risk_active = [f for f in self.high_risk_features 
                           if f in features and features[f] > 0]
        
        if len(high_risk_active) >= 3:
            risk_factors.append({
                'type': 'high_risk_feature_combination',
                'description': f'激活了{len(high_risk_active)}个高风险特征',
                'features': high_risk_active,
                'severity': 'high'
            })
        
        # 2. 检查特定的问题模式
        # 基于SHAP分析结果识别的问题模式
        if 'NUM_PROC' in features and features['NUM_PROC'] > 10:
            if 'AutoOpen' in features and features['AutoOpen'] > 0:
                risk_factors.append({
                    'type': 'complex_autoopen_pattern',
                    'description': '复杂VBA代码 + 自动执行',
                    'severity': 'medium'
                })
        
        return risk_factors
    
    def _check_protective_patterns(self, features):
        """检查保护性模式（降低误报风险的因素）"""
        protective_factors = []
        
        # 1. 检查可靠特征的支持
        reliable_active = [f for f in self.reliable_features 
                          if f in features and features[f] == 0]  # 可靠特征为0表示良性
        
        if len(reliable_active) >= 2:
            protective_factors.append({
                'type': 'reliable_features_support',
                'description': f'{len(reliable_active)}个可靠特征支持良性判断',
                'features': reliable_active,
                'strength': 'medium'
            })
        
        # 2. 检查合法复杂代码模式
        # 基于分析结果，某些特征组合可能表示合法的复杂工具
        if ('NUM_PROC' in features and features['NUM_PROC'] > 5 and
            'CreateObject' in features and features['CreateObject'] > 0):
            
            # 如果没有明显的恶意行为特征，可能是合法工具
            malicious_indicators = ['Shell', 'cmd.exe', 'powershell.exe']
            has_malicious = any(f in features and features[f] > 0 
                              for f in malicious_indicators if f in features)
            
            if not has_malicious:
                protective_factors.append({
                    'type': 'legitimate_complex_tool',
                    'description': '可能是合法的复杂Excel工具',
                    'strength': 'high'
                })
        
        return protective_factors
    
    def _make_final_decision(self, weighted_score, risk_factors, protective_factors, original_prob):
        """综合所有因素做出最终决策"""
        actions = []
        
        # 基础调整
        adjusted_prob = original_prob
        
        # 应用风险因素
        for risk in risk_factors:
            if risk['severity'] == 'high':
                adjusted_prob *= 1.2  # 增加20%
                actions.append(f"应用高风险调整: {risk['description']}")
            elif risk['severity'] == 'medium':
                adjusted_prob *= 1.1  # 增加10%
                actions.append(f"应用中风险调整: {risk['description']}")
        
        # 应用保护性因素
        for protection in protective_factors:
            if protection['strength'] == 'high':
                adjusted_prob *= 0.6  # 降低40%
                actions.append(f"应用强保护调整: {protection['description']}")
            elif protection['strength'] == 'medium':
                adjusted_prob *= 0.8  # 降低20%
                actions.append(f"应用中保护调整: {protection['description']}")
        
        # 限制概率范围
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))
        
        # 确定最终预测
        final_prediction = 1 if adjusted_prob > 0.5 else 0
        
        # 确定置信度
        if abs(adjusted_prob - 0.5) > 0.4:
            confidence = 'high'
        elif abs(adjusted_prob - 0.5) > 0.2:
            confidence = 'medium'
        else:
            confidence = 'low'
            actions.append("低置信度预测，建议人工审核")
        
        return {
            'prediction': final_prediction,
            'probability': adjusted_prob,
            'confidence': confidence,
            'actions': actions
        }
    
    def batch_post_process(self, predictions_df):
        """批量后处理"""
        print("\n🔄 批量后处理...")
        
        results = []
        for idx, row in predictions_df.iterrows():
            # 假设输入格式包含特征值和原始预测
            features = {col: row[col] for col in row.index 
                       if col not in ['prediction', 'probability', 'filename']}
            
            result = self.post_process_prediction(
                features, 
                row.get('prediction', 0), 
                row.get('probability', 0.5)
            )
            
            result['filename'] = row.get('filename', f'sample_{idx}')
            results.append(result)
        
        return pd.DataFrame(results)
    
    def save_post_processor(self, filepath='intelligent_post_processor.pkl'):
        """保存后处理器配置"""
        config = {
            'high_risk_features': self.high_risk_features,
            'reliable_features': self.reliable_features,
            'feature_weights': self.feature_weights,
            'threshold_adjustments': self.threshold_adjustments
        }
        
        joblib.dump(config, filepath)
        print(f"✅ 后处理器配置已保存到: {filepath}")
    
    def generate_post_processing_report(self, results_df):
        """生成后处理报告"""
        print("\n📋 生成后处理报告...")
        
        # 统计调整效果
        original_malicious = (results_df['original_prediction'] == 1).sum()
        adjusted_malicious = (results_df['adjusted_prediction'] == 1).sum()
        
        # 计算调整的样本数
        changed_predictions = (results_df['original_prediction'] != results_df['adjusted_prediction']).sum()
        
        report = f"""
# 智能后处理报告

## 处理统计
- 总样本数: {len(results_df)}
- 原始恶意预测: {original_malicious}
- 调整后恶意预测: {adjusted_malicious}
- 预测改变样本: {changed_predictions}
- 调整率: {changed_predictions/len(results_df)*100:.1f}%

## 调整方向
- 恶意→良性: {((results_df['original_prediction'] == 1) & (results_df['adjusted_prediction'] == 0)).sum()}
- 良性→恶意: {((results_df['original_prediction'] == 0) & (results_df['adjusted_prediction'] == 1)).sum()}

## 置信度分布
- 高置信度: {(results_df['confidence_level'] == 'high').sum()}
- 中置信度: {(results_df['confidence_level'] == 'medium').sum()}
- 低置信度: {(results_df['confidence_level'] == 'low').sum()}
"""
        
        with open('post_processing_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ 后处理报告已保存到: post_processing_report.md")

def main():
    """主函数 - 演示用法"""
    print("🎯 智能后处理机制")
    print("=" * 60)
    
    # 初始化后处理器
    processor = IntelligentPostProcessor()
    
    # 保存配置
    processor.save_post_processor()
    
    print("\n✅ 智能后处理器已准备就绪!")
    print("\n📋 使用方法:")
    print("1. 对单个样本: processor.post_process_prediction(features, pred, prob)")
    print("2. 批量处理: processor.batch_post_process(df)")
    print("3. 生成报告: processor.generate_post_processing_report(results)")

if __name__ == "__main__":
    main()
