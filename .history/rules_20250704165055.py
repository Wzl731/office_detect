#!/usr/bin/env python3
"""
后处理规则应用模块
加载预生成的规则文件，提供快速的后处理功能
"""

import json
import pickle
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PostProcessingRules:
    def __init__(self, rules_file='feature_analysis/post_processing_rules.pkl'):
        """
        初始化规则应用器
        
        Args:
            rules_file: 规则文件路径（支持.json或.pkl格式）
        """
        self.rules_file = Path(rules_file)
        self.rules = None
        self.load_rules()
        
    def load_rules(self):
        """加载规则文件"""
        try:
            if self.rules_file.suffix == '.pkl':
                # 加载pickle格式（更快）
                with open(self.rules_file, 'rb') as f:
                    self.rules = pickle.load(f)
            elif self.rules_file.suffix == '.json':
                # 加载JSON格式
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    self.rules = json.load(f)
            else:
                # 尝试pickle格式
                pkl_file = self.rules_file.with_suffix('.pkl')
                if pkl_file.exists():
                    with open(pkl_file, 'rb') as f:
                        self.rules = pickle.load(f)
                else:
                    raise FileNotFoundError(f"规则文件不存在: {self.rules_file}")
            
            print(f"✅ 规则已加载: {self.rules_file}")
            return True
            
        except Exception as e:
            print(f"❌ 规则加载失败: {e}")
            self.rules = None
            return False
    
    def is_available(self):
        """检查规则是否可用"""
        return self.rules is not None
    
    def get_feature_weight(self, feature_name):
        """获取特征权重"""
        if not self.is_available():
            return 1.0
        
        weights = self.rules.get('weight_adjustment', {}).get('feature_weights', {})
        return weights.get(feature_name, 1.0)
    
    def get_threshold_adjustment(self, feature_name):
        """获取阈值调整"""
        if not self.is_available():
            return 1.0
        
        adjustments = self.rules.get('threshold_adjustment', {}).get('threshold_adjustments', {})
        return adjustments.get(feature_name, 1.0)
    
    def check_risk_patterns(self, features):
        """检查风险模式"""
        if not self.is_available():
            return []
        
        risk_factors = []
        risk_patterns = self.rules.get('pattern_recognition', {}).get('risk_patterns', [])
        
        for pattern in risk_patterns:
            if self._evaluate_pattern_condition(pattern['condition'], features):
                risk_factors.append({
                    'type': pattern['name'],
                    'description': pattern['description'],
                    'severity': pattern['severity'],
                    'weight_multiplier': pattern.get('weight_multiplier', 1.0)
                })
        
        return risk_factors
    
    def check_protective_patterns(self, features):
        """检查保护模式"""
        if not self.is_available():
            return []
        
        protective_factors = []
        protective_patterns = self.rules.get('pattern_recognition', {}).get('protective_patterns', [])
        
        for pattern in protective_patterns:
            if self._evaluate_pattern_condition(pattern['condition'], features):
                protective_factors.append({
                    'type': pattern['name'],
                    'description': pattern['description'],
                    'strength': pattern['strength'],
                    'weight_multiplier': pattern.get('weight_multiplier', 1.0)
                })
        
        return protective_factors
    
    def _evaluate_pattern_condition(self, condition, features):
        """评估模式条件"""
        try:
            if condition['type'] == 'feature_count':
                # 计算激活的特征数量
                active_count = sum(1 for f in condition['features'] if features.get(f, 0) > 0)
                threshold = condition['threshold']
                operator = condition['operator']
                
                if operator == '>=':
                    return active_count >= threshold
                elif operator == '>':
                    return active_count > threshold
                elif operator == '==':
                    return active_count == threshold
                
            elif condition['type'] == 'feature_zero_count':
                # 计算为0的特征数量
                zero_count = sum(1 for f in condition['features'] if features.get(f, 0) == 0)
                threshold = condition['threshold']
                operator = condition['operator']
                
                if operator == '>=':
                    return zero_count >= threshold
                elif operator == '>':
                    return zero_count > threshold
                elif operator == '==':
                    return zero_count == threshold
                    
            elif condition['type'] == 'complex_condition':
                # 复杂条件（如合法工具检测）
                has_complexity = condition.get('has_complexity', {})
                lacks_malicious = condition.get('lacks_malicious', {})
                
                # 检查复杂性条件
                complexity_met = True
                if has_complexity:
                    active_count = sum(1 for f in has_complexity['features'] if features.get(f, 0) > 0)
                    complexity_met = active_count >= has_complexity['threshold']
                
                # 检查缺乏恶意条件
                malicious_met = True
                if lacks_malicious:
                    malicious_count = sum(1 for f in lacks_malicious['features'] if features.get(f, 0) > 0)
                    malicious_met = malicious_count == lacks_malicious['threshold']
                
                return complexity_met and malicious_met
                
        except Exception as e:
            print(f"⚠️  模式条件评估失败: {e}")
            return False
        
        return False
    
    def calculate_adjusted_risk_score(self, features):
        """计算调整后的风险评分"""
        if not self.is_available():
            return 0.5
        
        # 获取特征分类
        fc = self.rules.get('feature_classification', {})
        high_risk_features = fc.get('high_risk_features', [])
        reliable_features = fc.get('reliable_features', [])
        
        # 计算基础风险评分
        risk_score = 0.0
        total_weight = 0.0
        
        for feature_name, feature_value in features.items():
            if feature_value > 0:
                # 基础权重
                base_weight = 1.0
                if feature_name in high_risk_features:
                    base_weight = 0.5  # 高风险特征降权
                
                # 应用规则权重
                rule_weight = self.get_feature_weight(feature_name)
                final_weight = base_weight * rule_weight
                
                # 标准化特征值
                normalized_value = min(feature_value / 10.0, 1.0)
                contribution = final_weight * normalized_value
                
                risk_score += contribution
                total_weight += final_weight
        
        # 标准化评分
        if total_weight > 0:
            risk_score = risk_score / total_weight
        else:
            risk_score = 0.0
        
        return min(max(risk_score, 0.0), 1.0)
    
    def assess_confidence(self, adjusted_probability, risk_factor_count, protective_factor_count):
        """评估置信度"""
        if not self.is_available():
            return 'medium'
        
        confidence_rules = self.rules.get('confidence_assessment', {})
        
        # 检查高置信度条件
        for condition in confidence_rules.get('high_confidence_conditions', []):
            if self._evaluate_confidence_condition(condition['condition'], 
                                                 adjusted_probability, 
                                                 risk_factor_count, 
                                                 protective_factor_count):
                return 'high'
        
        # 检查低置信度条件
        for condition in confidence_rules.get('low_confidence_conditions', []):
            if self._evaluate_confidence_condition(condition['condition'], 
                                                 adjusted_probability, 
                                                 risk_factor_count, 
                                                 protective_factor_count):
                return 'low'
        
        # 默认中等置信度
        return 'medium'
    
    def _evaluate_confidence_condition(self, condition_str, adjusted_probability, risk_factor_count, protective_factor_count):
        """评估置信度条件"""
        try:
            # 创建安全的评估环境
            safe_dict = {
                'adjusted_probability': adjusted_probability,
                'risk_factor_count': risk_factor_count,
                'protective_factor_count': protective_factor_count
            }
            
            # 安全评估条件
            return eval(condition_str, {"__builtins__": {}}, safe_dict)
        except:
            return False
    
    def post_process_prediction(self, features, original_prediction, original_probability):
        """
        对预测结果进行后处理
        
        Args:
            features: dict, 特征值字典
            original_prediction: int, 原始预测结果 (0=良性, 1=恶意)
            original_probability: float, 原始预测概率
            
        Returns:
            dict: 后处理结果
        """
        if not self.is_available():
            # 规则不可用时返回原始结果
            return {
                'original_prediction': original_prediction,
                'original_probability': original_probability,
                'adjusted_prediction': original_prediction,
                'adjusted_probability': original_probability,
                'confidence_level': 'medium',
                'post_processing_applied': False,
                'risk_factors': [],
                'protective_factors': [],
                'post_processing_actions': ['规则不可用，使用原始结果']
            }
        
        # 1. 检查风险和保护模式
        risk_factors = self.check_risk_patterns(features)
        protective_factors = self.check_protective_patterns(features)
        
        # 2. 计算调整后的概率
        adjusted_probability = original_probability
        actions = []
        
        # 应用风险因素调整
        for risk in risk_factors:
            multiplier = risk.get('weight_multiplier', 1.0)
            adjusted_probability *= multiplier
            actions.append(f"应用{risk['severity']}风险调整: {risk['description']}")
        
        # 应用保护因素调整
        for protection in protective_factors:
            multiplier = protection.get('weight_multiplier', 1.0)
            adjusted_probability *= multiplier
            actions.append(f"应用{protection['strength']}保护调整: {protection['description']}")
        
        # 限制概率范围
        adjusted_probability = max(0.01, min(0.99, adjusted_probability))
        
        # 3. 确定最终预测
        adjusted_prediction = 1 if adjusted_probability > 0.5 else 0
        
        # 4. 评估置信度
        confidence_level = self.assess_confidence(
            adjusted_probability, len(risk_factors), len(protective_factors)
        )
        
        if confidence_level == 'low':
            actions.append("低置信度预测，建议人工审核")
        
        return {
            'original_prediction': original_prediction,
            'original_probability': original_probability,
            'adjusted_prediction': adjusted_prediction,
            'adjusted_probability': adjusted_probability,
            'confidence_level': confidence_level,
            'post_processing_applied': True,
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'post_processing_actions': actions
        }
    
    def get_rules_summary(self):
        """获取规则摘要"""
        if not self.is_available():
            return "规则不可用"
        
        metadata = self.rules.get('metadata', {})
        generation_info = metadata.get('generation_info', {})
        
        summary = f"""
规则摘要:
- 总特征数: {generation_info.get('total_features', 'N/A')}
- 高风险特征: {generation_info.get('high_risk_features_count', 'N/A')} 个
- 可靠特征: {generation_info.get('reliable_features_count', 'N/A')} 个
- 严重问题特征: {generation_info.get('critical_problematic_count', 'N/A')} 个
- 生成时间: {generation_info.get('generation_timestamp', 'N/A')}
"""
        return summary.strip()

# 全局规则实例（单例模式）
_global_rules = None

def get_post_processing_rules(rules_file='feature_analysis/post_processing_rules.pkl'):
    """获取全局规则实例"""
    global _global_rules
    if _global_rules is None:
        _global_rules = PostProcessingRules(rules_file)
    return _global_rules

def apply_post_processing(features, original_prediction, original_probability, rules_file=None):
    """
    便捷的后处理函数
    
    Args:
        features: dict, 特征值字典
        original_prediction: int, 原始预测结果
        original_probability: float, 原始预测概率
        rules_file: str, 可选的规则文件路径
        
    Returns:
        dict: 后处理结果
    """
    if rules_file:
        rules = PostProcessingRules(rules_file)
    else:
        rules = get_post_processing_rules()
    
    return rules.post_process_prediction(features, original_prediction, original_probability)

if __name__ == "__main__":
    # 测试规则加载
    rules = PostProcessingRules()
    if rules.is_available():
        print("✅ 规则模块测试成功")
        print(rules.get_rules_summary())
    else:
        print("❌ 规则模块测试失败")
        print("请先运行 feature_analysis/generate_post_processing_rules.py 生成规则文件")
