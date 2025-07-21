#!/usr/bin/env python3
"""
评估智能后处理规则对恶意文件检测率的影响
"""

import sys
import os
from pathlib import Path
import pandas as pd

# 添加当前目录到路径
sys.path.append('.')

try:
    from rules import get_post_processing_rules, apply_post_processing
    from original_feature_extractor import extract_features
    import joblib
    import numpy as np
    POST_PROCESSOR_AVAILABLE = True
    print("✅ 后处理规则模块已加载")
except ImportError as e:
    POST_PROCESSOR_AVAILABLE = False
    print(f"❌ 后处理规则模块加载失败: {e}")

def test_sample_files(folder_path, max_files=20):
    """测试样本文件，比较原始预测和后处理结果"""
    
    if not POST_PROCESSOR_AVAILABLE:
        print("❌ 后处理模块不可用")
        return
    
    # 初始化检测器
    detector = VBAMalwareDetector(use_post_processing=False)  # 先不使用后处理
    if not detector.load_models():
        print("❌ 模型加载失败")
        return
    
    # 获取规则
    rules = get_post_processing_rules()
    if not rules.is_available():
        print("❌ 规则不可用")
        return
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder}")
        return
    
    # 获取文件列表
    files = []
    for ext in ['*.doc', '*.docx', '*.xls', '*.xlsx', '*.ppt', '*.pptx']:
        files.extend(folder.glob(ext))
    
    files = files[:max_files]  # 限制文件数量
    
    print(f"🔍 测试 {len(files)} 个文件...")
    print("=" * 80)
    
    results = []
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] 📄 {file_path.name}")
        
        try:
            # 提取特征
            features = extract_features(str(file_path))
            if features is None:
                print("  ❌ 特征提取失败")
                continue
            
            # 转换为字典格式
            features_dict = {}
            for j, feature_name in enumerate(detector.feature_names):
                features_dict[feature_name] = features[j] if j < len(features) else 0
            
            # RandomForest原始预测
            rf_model = detector.models['RandomForest']
            rf_prob = rf_model.predict_proba([features])[0]
            rf_malicious_prob = rf_prob[1]
            rf_prediction = 1 if rf_malicious_prob > 0.5 else 0
            
            # 应用后处理
            post_result = apply_post_processing(
                features_dict, rf_prediction, rf_malicious_prob
            )
            
            # 记录结果
            result = {
                'file': file_path.name,
                'original_prediction': rf_prediction,
                'original_probability': rf_malicious_prob,
                'adjusted_prediction': post_result['adjusted_prediction'],
                'adjusted_probability': post_result['adjusted_probability'],
                'confidence_level': post_result['confidence_level'],
                'risk_factors': len(post_result['risk_factors']),
                'protective_factors': len(post_result['protective_factors']),
                'changed': rf_prediction != post_result['adjusted_prediction']
            }
            results.append(result)
            
            # 显示结果
            print(f"  🤖 原始: {'恶意' if rf_prediction else '良性'} ({rf_malicious_prob:.3f})")
            print(f"  🧠 调整: {'恶意' if post_result['adjusted_prediction'] else '良性'} ({post_result['adjusted_probability']:.3f})")
            print(f"  📊 置信度: {post_result['confidence_level']}")
            print(f"  🚨 风险因素: {len(post_result['risk_factors'])}")
            print(f"  🛡️  保护因素: {len(post_result['protective_factors'])}")
            
            if result['changed']:
                print(f"  ⚠️  预测改变: {'恶意→良性' if rf_prediction else '良性→恶意'}")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    # 统计分析
    if results:
        print(f"\n{'='*80}")
        print("📊 统计分析")
        print(f"{'='*80}")
        
        total = len(results)
        changed_count = sum(1 for r in results if r['changed'])
        malicious_to_benign = sum(1 for r in results if r['original_prediction'] == 1 and r['adjusted_prediction'] == 0)
        benign_to_malicious = sum(1 for r in results if r['original_prediction'] == 0 and r['adjusted_prediction'] == 1)
        
        original_malicious = sum(1 for r in results if r['original_prediction'] == 1)
        adjusted_malicious = sum(1 for r in results if r['adjusted_prediction'] == 1)
        
        print(f"📈 总文件数: {total}")
        print(f"🔄 预测改变: {changed_count} ({changed_count/total*100:.1f}%)")
        print(f"📉 恶意→良性: {malicious_to_benign} ({malicious_to_benign/total*100:.1f}%)")
        print(f"📈 良性→恶意: {benign_to_malicious} ({benign_to_malicious/total*100:.1f}%)")
        print(f"🎯 原始恶意检出: {original_malicious}/{total} ({original_malicious/total*100:.1f}%)")
        print(f"🎯 调整后恶意检出: {adjusted_malicious}/{total} ({adjusted_malicious/total*100:.1f}%)")
        
        if original_malicious > 0:
            detection_rate_change = (adjusted_malicious - original_malicious) / original_malicious * 100
            print(f"📊 检测率变化: {detection_rate_change:+.1f}%")
        
        # 保存详细结果
        df = pd.DataFrame(results)
        df.to_csv('detection_impact_analysis.csv', index=False)
        print(f"\n💾 详细结果已保存到: detection_impact_analysis.csv")

def main():
    """主函数"""
    print("🎯 智能后处理规则对恶意文件检测率影响评估")
    print("=" * 80)
    
    # 测试恶意文件
    print("\n🦠 测试恶意文件 (data/bad250623):")
    test_sample_files('data/bad250623', max_files=30)
    
    # 测试良性文件
    print("\n🟢 测试良性文件 (data/good250623):")
    test_sample_files('data/good250623', max_files=30)

if __name__ == "__main__":
    main()
