#!/usr/bin/env python3
"""
快速测试恶意样本的检测效果
"""

import sys
import os
from pathlib import Path
import glob

# 添加当前目录到路径
sys.path.append('.')

try:
    from rules import get_post_processing_rules, apply_post_processing
    from original_feature_extractor import OriginalVBAFeatureExtractor
    import joblib
    import numpy as np
    POST_PROCESSOR_AVAILABLE = True
    print("✅ 后处理规则模块已加载")
except ImportError as e:
    POST_PROCESSOR_AVAILABLE = False
    print(f"❌ 后处理规则模块加载失败: {e}")

def test_malicious_samples(folder_path, max_files=20):
    """测试恶意样本"""
    
    if not POST_PROCESSOR_AVAILABLE:
        print("❌ 后处理模块不可用")
        return
    
    # 加载RandomForest模型
    try:
        rf_model = joblib.load('models/RandomForest_model.pkl')
        print("✅ RandomForest模型已加载")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载特征名称
    try:
        with open('models/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"✅ 特征名称已加载 ({len(feature_names)} 个)")
    except Exception as e:
        print(f"❌ 特征名称加载失败: {e}")
        return
    
    # 初始化特征提取器
    try:
        extractor = VBAFeatureExtractor()
        print("✅ 特征提取器已初始化")
    except Exception as e:
        print(f"❌ 特征提取器初始化失败: {e}")
        return

    # 获取规则
    rules = get_post_processing_rules()
    if not rules.is_available():
        print("❌ 规则不可用")
        return
    
    # 获取文件列表
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder}")
        return
    
    # 获取恶意文件
    files = []
    for ext in ['*.doc', '*.docx', '*.xls', '*.xlsx']:
        files.extend(folder.glob(ext))
    
    files = files[:max_files]  # 限制文件数量
    
    print(f"\n🦠 测试恶意样本: {len(files)} 个文件")
    print("=" * 80)
    
    results = {
        'original_malicious': 0,
        'adjusted_malicious': 0,
        'malicious_to_benign': 0,
        'benign_to_malicious': 0,
        'total_files': 0,
        'failed_files': 0
    }
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] 📄 {file_path.name}")
        
        try:
            # 提取特征
            features = extractor.extract_features_from_file(str(file_path))
            if features is None:
                print("  ❌ 特征提取失败")
                results['failed_files'] += 1
                continue
            
            # 转换为字典格式
            features_dict = {}
            for j, feature_name in enumerate(feature_names):
                features_dict[feature_name] = features[j] if j < len(features) else 0
            
            # RandomForest原始预测
            rf_prob = rf_model.predict_proba([features])[0]
            rf_malicious_prob = rf_prob[1]
            rf_prediction = 1 if rf_malicious_prob > 0.5 else 0
            
            # 应用后处理
            post_result = apply_post_processing(
                features_dict, rf_prediction, rf_malicious_prob
            )
            
            # 统计结果
            results['total_files'] += 1
            if rf_prediction == 1:
                results['original_malicious'] += 1
            if post_result['adjusted_prediction'] == 1:
                results['adjusted_malicious'] += 1
            
            if rf_prediction == 1 and post_result['adjusted_prediction'] == 0:
                results['malicious_to_benign'] += 1
                print("  ⚠️  恶意→良性 (可能的漏报)")
            elif rf_prediction == 0 and post_result['adjusted_prediction'] == 1:
                results['benign_to_malicious'] += 1
                print("  📈 良性→恶意")
            
            # 显示结果
            print(f"  🤖 原始: {'恶意' if rf_prediction else '良性'} ({rf_malicious_prob:.3f})")
            print(f"  🧠 调整: {'恶意' if post_result['adjusted_prediction'] else '良性'} ({post_result['adjusted_probability']:.3f})")
            print(f"  📊 置信度: {post_result['confidence_level']}")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            results['failed_files'] += 1
    
    # 统计分析
    print(f"\n{'='*80}")
    print("📊 恶意样本检测结果统计")
    print(f"{'='*80}")
    
    total_processed = results['total_files']
    if total_processed > 0:
        original_detection_rate = results['original_malicious'] / total_processed * 100
        adjusted_detection_rate = results['adjusted_malicious'] / total_processed * 100
        
        print(f"📈 处理文件数: {total_processed}")
        print(f"❌ 失败文件数: {results['failed_files']}")
        print(f"🎯 原始检测率: {results['original_malicious']}/{total_processed} ({original_detection_rate:.1f}%)")
        print(f"🎯 调整后检测率: {results['adjusted_malicious']}/{total_processed} ({adjusted_detection_rate:.1f}%)")
        print(f"📉 恶意→良性: {results['malicious_to_benign']} 个 (潜在漏报)")
        print(f"📈 良性→恶意: {results['benign_to_malicious']} 个")
        
        detection_rate_change = adjusted_detection_rate - original_detection_rate
        print(f"📊 检测率变化: {detection_rate_change:+.1f}%")
        
        if results['malicious_to_benign'] > 0:
            miss_rate = results['malicious_to_benign'] / results['original_malicious'] * 100 if results['original_malicious'] > 0 else 0
            print(f"⚠️  漏报率: {miss_rate:.1f}% ({results['malicious_to_benign']}/{results['original_malicious']})")
    
    return results

def main():
    """主函数"""
    print("🦠 恶意样本检测效果测试")
    print("=" * 80)
    
    # 测试恶意文件
    results = test_malicious_samples('data/bad250623', max_files=30)
    
    if results and results['total_files'] > 0:
        print(f"\n💡 分析结论:")
        print("=" * 60)
        
        if results['malicious_to_benign'] > 0:
            print(f"⚠️  智能后处理确实降低了对恶意文件的检测率")
            print(f"   原因: {results['malicious_to_benign']} 个恶意文件被误判为良性")
            print(f"   建议: 需要调整后处理规则的阈值或条件")
        else:
            print(f"✅ 智能后处理没有影响恶意文件的检测")
        
        if results['adjusted_malicious'] > results['original_malicious']:
            print(f"📈 后处理还提高了 {results['benign_to_malicious']} 个文件的检测")

if __name__ == "__main__":
    main()
