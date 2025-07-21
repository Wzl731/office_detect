#!/usr/bin/env python3
"""
Random Forest性能分析和误报降低建议
基于RF算法原理提供具体的改进方案
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from original_feature_extractor import OriginalVBAFeatureExtractor

def analyze_current_performance():
    """分析当前RF模型性能"""
    print("🎯 Random Forest误报降低分析")
    print("=" * 60)
    
    # 加载模型
    models_dir = Path('models')
    model_path = models_dir / 'randomforest_model.pkl'
    feature_path = models_dir / 'feature_columns.pkl'
    
    if not model_path.exists():
        print("❌ Random Forest模型不存在")
        return
    
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    with open(feature_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"✅ 加载模型成功: {len(feature_names)} 个特征")

    # 分析特征重要性
    print("\n🔍 特征重要性分析:")
    importances = rf_model.feature_importances_

    # 确保特征名和重要性数量匹配
    feature_names_clean = feature_names[1:] if len(feature_names) > len(importances) else feature_names
    print(f"特征名数量: {len(feature_names_clean)}, 重要性数量: {len(importances)}")

    # 创建特征重要性DataFrame
    feature_df = pd.DataFrame({
        'feature': feature_names_clean[:len(importances)],  # 确保长度匹配
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Top 15 重要特征:")
    for i, (_, row) in enumerate(feature_df.head(15).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # 分析特征类型分布
    print(f"\n📊 特征类型重要性分析:")
    
    # 分类特征
    obfuscation_features = feature_df[feature_df['feature'].str.startswith('FEATURE_')]
    suspicious_features = feature_df[feature_df['feature'].str.startswith('SUSPICIOUS_')]
    
    print(f"  混淆特征 (77个):")
    print(f"    平均重要性: {obfuscation_features['importance'].mean():.4f}")
    print(f"    最高重要性: {obfuscation_features['importance'].max():.4f}")
    print(f"    重要性>0.01的特征数: {len(obfuscation_features[obfuscation_features['importance'] > 0.01])}")
    
    print(f"  可疑特征 (46个):")
    print(f"    平均重要性: {suspicious_features['importance'].mean():.4f}")
    print(f"    最高重要性: {suspicious_features['importance'].max():.4f}")
    print(f"    重要性>0.01的特征数: {len(suspicious_features[suspicious_features['importance'] > 0.01])}")
    
    return rf_model, feature_df

def suggest_improvements():
    """基于RF原理提出改进建议"""
    print(f"\n💡 基于Random Forest原理的误报降低建议:")
    print("=" * 60)
    
    print("🎯 1. 特征工程改进:")
    print("   ✅ 添加良性特征平衡器")
    print("      - Excel良性操作特征 (Worksheet, Range, Cells)")
    print("      - 用户界面操作特征 (MsgBox, UserForm)")
    print("      - 数据处理特征 (循环, 条件判断)")
    print("      - 文档操作特征 (Save, Copy, Paste)")
    print("      - 良性比例特征 (注释比例, 良性密度)")
    
    print("\n🎯 2. 类别权重调整:")
    print("   ✅ 使用class_weight='balanced'")
    print("      - 自动平衡良性和恶意样本权重")
    print("      - 减少对少数类的过拟合")
    
    print("\n🎯 3. 决策阈值优化:")
    print("   ✅ 提高分类阈值 (0.5 → 0.7)")
    print("      - 降低误报率，可能略微降低检测率")
    print("      - 基于业务需求平衡两者")
    
    print("\n🎯 4. 模型参数调优:")
    print("   ✅ max_depth=15 (防止过拟合)")
    print("   ✅ min_samples_split=5 (增加泛化能力)")
    print("   ✅ min_samples_leaf=2 (平滑决策边界)")
    print("   ✅ n_estimators=200 (提高稳定性)")
    
    print("\n🎯 5. 后处理策略:")
    print("   ✅ 复杂表格识别")
    print("      - 检测Excel特有的VBA模式")
    print("      - 良性分数 > 恶意分数 → 改判良性")
    print("   ✅ 置信度过滤")
    print("      - 低置信度预测 → 人工审核")

def demonstrate_feature_engineering():
    """演示特征工程改进"""
    print(f"\n🔧 特征工程改进演示:")
    print("=" * 60)
    
    print("📋 原始特征集 (123维):")
    print("  - 混淆特征: 77个 (过程数量, 行长度, 字符串操作等)")
    print("  - 可疑特征: 46个 (Shell, CreateObject, cmd.exe等)")
    
    print("\n📋 增强特征集 (138维):")
    print("  - 原始特征: 123个")
    print("  - 良性特征: 15个 (新增)")
    
    print("\n🔍 良性特征详细说明:")
    benign_features = [
        "Excel良性操作 (5个)",
        "  - Worksheet/Workbook对象操作",
        "  - Cells/Range单元格操作", 
        "  - SUM/AVERAGE/COUNT函数使用",
        "  - PivotTable/Chart数据分析",
        "  - Application对象调用",
        "",
        "用户界面操作 (3个)",
        "  - MsgBox/InputBox用户交互",
        "  - Show/Hide界面显示",
        "  - Button/TextBox控件操作",
        "",
        "数据处理特征 (3个)",
        "  - Value/Text/Formula赋值",
        "  - For循环数据处理",
        "  - If/Select条件判断",
        "",
        "文档操作特征 (2个)",
        "  - Save/Open文档操作",
        "  - Copy/Paste编辑操作",
        "",
        "良性比例特征 (2个)",
        "  - 注释比例 (良性代码通常有更多注释)",
        "  - 良性关键词密度"
    ]
    
    for feature in benign_features:
        print(f"    {feature}")

def show_rf_algorithm_insights():
    """展示RF算法对误报降低的作用机制"""
    print(f"\n🧠 Random Forest算法误报降低机制:")
    print("=" * 60)
    
    print("🌳 1. 决策树集成投票:")
    print("   - 多个决策树独立判断")
    print("   - 投票机制减少单树偏差")
    print("   - 良性特征增加 → 更多树投票'良性'")
    
    print("\n📊 2. 特征重要性自动选择:")
    print("   - RF自动计算每个特征的重要性")
    print("   - 重要的良性特征会被优先使用")
    print("   - 噪声特征影响被自动降低")
    
    print("\n⚖️  3. 类别权重平衡:")
    print("   - class_weight='balanced'自动调整")
    print("   - 防止模型偏向多数类")
    print("   - 提高少数类(恶意)的准确性")
    
    print("\n🎯 4. 阈值调整策略:")
    print("   - 默认阈值0.5 → 调整到0.7")
    print("   - 需要更高置信度才判断为恶意")
    print("   - 误报率↓, 检测率可能略微↓")
    
    print("\n📈 5. 预期改进效果:")
    print("   - 误报率: 从~10% 降低到 ~5%")
    print("   - 检测率: 保持在 ~80% 以上")
    print("   - 整体F1分数提升")

def main():
    """主函数"""
    # 分析当前性能
    rf_model, feature_df = analyze_current_performance()
    
    # 提出改进建议
    suggest_improvements()
    
    # 演示特征工程
    demonstrate_feature_engineering()
    
    # 展示RF算法洞察
    show_rf_algorithm_insights()
    
    print(f"\n🚀 下一步行动建议:")
    print("=" * 60)
    print("1. 运行 train_enhanced_rf.py 训练增强模型")
    print("2. 使用新模型测试 data/sample 文件夹")
    print("3. 比较原始模型和增强模型的误报率")
    print("4. 根据结果调整分类阈值")
    print("5. 收集更多良性样本进行进一步训练")

if __name__ == "__main__":
    main()
