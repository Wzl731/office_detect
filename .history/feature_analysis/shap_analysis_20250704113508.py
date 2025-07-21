#!/usr/bin/env python3
"""
SHAP可视化解释分析
用于解释模型对误报样本的预测结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SHAPAnalyzer:
    def __init__(self):
        self.white_data = None
        self.black_data = None
        self.misclassified_data = None
        self.feature_names = None
        self.model = None
        self.scaler = None
        self.explainer = None
        
    def load_data(self):
        """加载数据文件"""
        print("📊 正在加载数据文件...")
        
        try:
            # 加载数据
            self.white_data = pd.read_excel('../data/good250623_features.xlsx')
            self.black_data = pd.read_excel('../data/bad250623_features.xlsx')
            self.misclassified_data = pd.read_excel('../data/good2bad_features.xlsx')
            
            # 获取特征名称
            self.feature_names = [col for col in self.white_data.columns if col != 'FILENAME']
            
            print(f"✅ 白样本: {self.white_data.shape}")
            print(f"✅ 黑样本: {self.black_data.shape}")
            print(f"✅ 误报样本: {self.misclassified_data.shape}")
            print(f"✅ 特征数量: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def prepare_training_data(self):
        """准备训练数据"""
        print("\n🔧 准备训练数据...")
        
        # 提取特征
        white_features = self.white_data[self.feature_names].fillna(0)
        black_features = self.black_data[self.feature_names].fillna(0)
        
        # 合并数据
        X = pd.concat([white_features, black_features], ignore_index=True)
        y = np.concatenate([np.zeros(len(white_features)), np.ones(len(black_features))])
        
        # 数据标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ 训练数据形状: {X_scaled.shape}")
        print(f"✅ 标签分布: 良性={np.sum(y==0)}, 恶意={np.sum(y==1)}")
        
        return X_scaled, y
    
    def train_model(self, X, y):
        """训练随机森林模型"""
        print("\n🤖 训练随机森林模型...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估模型
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"✅ 训练准确率: {train_score:.4f}")
        print(f"✅ 测试准确率: {test_score:.4f}")
        
        # 保存模型
        joblib.dump(self.model, 'random_forest_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        print("✅ 模型已保存")
        
        return X_train, X_test, y_train, y_test
    
    def create_shap_explainer(self, X_train):
        """创建SHAP解释器"""
        print("\n🔍 创建SHAP解释器...")
        
        # 使用TreeExplainer for RandomForest
        self.explainer = shap.TreeExplainer(self.model)
        
        # 计算SHAP值 (使用训练数据的子集以提高速度)
        sample_size = min(1000, len(X_train))
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[sample_indices]
        
        print(f"✅ 使用 {sample_size} 个样本计算SHAP值...")
        
        return X_sample
    
    def analyze_misclassified_samples(self):
        """分析误报样本的SHAP值"""
        print("\n🎯 分析误报样本...")
        
        # 准备误报样本数据
        misc_features = self.misclassified_data[self.feature_names].fillna(0)
        misc_scaled = self.scaler.transform(misc_features)
        
        # 预测误报样本
        predictions = self.model.predict(misc_scaled)
        probabilities = self.model.predict_proba(misc_scaled)
        
        print(f"✅ 误报样本预测结果:")
        print(f"  - 预测为恶意: {np.sum(predictions == 1)}")
        print(f"  - 预测为良性: {np.sum(predictions == 0)}")
        print(f"  - 平均恶意概率: {probabilities[:, 1].mean():.4f}")
        
        # 计算误报样本的SHAP值
        print("🔍 计算误报样本SHAP值...")
        shap_values = self.explainer.shap_values(misc_scaled)
        
        # 如果是二分类，取恶意类的SHAP值
        if isinstance(shap_values, list):
            shap_values_malicious = shap_values[1]  # 恶意类
        else:
            shap_values_malicious = shap_values
        
        return misc_scaled, shap_values_malicious, predictions, probabilities
    
    def plot_shap_summary(self, misc_scaled, shap_values):
        """绘制SHAP摘要图"""
        print("\n📈 绘制SHAP摘要图...")
        
        # 创建特征名称DataFrame
        feature_df = pd.DataFrame(misc_scaled, columns=self.feature_names)
        
        # 1. SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, feature_df, max_display=20, show=False)
        plt.title('SHAP特征重要性摘要 (误报样本)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        print("✅ SHAP摘要图已保存到: shap_summary_plot.png")
        plt.show()
        
        # 2. SHAP条形图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, feature_df, plot_type="bar", max_display=20, show=False)
        plt.title('SHAP特征重要性条形图 (误报样本)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
        print("✅ SHAP条形图已保存到: shap_bar_plot.png")
        plt.show()
    
    def plot_shap_waterfall(self, misc_scaled, shap_values, sample_idx=0):
        """绘制单个样本的SHAP瀑布图"""
        print(f"\n🌊 绘制样本 {sample_idx} 的SHAP瀑布图...")
        
        # 创建特征名称DataFrame
        feature_df = pd.DataFrame(misc_scaled, columns=self.feature_names)
        
        # 创建Explanation对象
        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
            data=feature_df.iloc[sample_idx].values,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title(f'SHAP瀑布图 - 误报样本 {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
        print(f"✅ SHAP瀑布图已保存到: shap_waterfall_sample_{sample_idx}.png")
        plt.show()
    
    def analyze_top_features(self, shap_values):
        """分析最重要的特征"""
        print("\n🔍 分析最重要的特征...")
        
        # 计算特征重要性
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # 保存结果
        importance_df.to_excel('shap_feature_importance.xlsx', index=False)
        print("✅ SHAP特征重要性已保存到: shap_feature_importance.xlsx")
        
        # 显示前20个最重要的特征
        print("\n📋 SHAP重要性最高的前20个特征:")
        print("-" * 50)
        for i, row in importance_df.head(20).iterrows():
            print(f"{row['feature']:25} | {row['importance']:.6f}")
        
        return importance_df

def main():
    """主函数"""
    print("🎯 SHAP可视化解释分析")
    print("=" * 60)
    print("📋 功能: 解释模型对误报样本的预测结果")
    print("🤖 模型: 随机森林分类器")
    print()
    
    # 创建分析器
    analyzer = SHAPAnalyzer()
    
    # 加载数据
    if not analyzer.load_data():
        return
    
    # 准备训练数据
    X, y = analyzer.prepare_training_data()
    
    # 训练模型
    X_train, X_test, y_train, y_test = analyzer.train_model(X, y)
    
    # 创建SHAP解释器
    X_sample = analyzer.create_shap_explainer(X_train)
    
    # 分析误报样本
    misc_scaled, shap_values, predictions, probabilities = analyzer.analyze_misclassified_samples()
    
    # 绘制SHAP图表
    analyzer.plot_shap_summary(misc_scaled, shap_values)
    
    # 绘制瀑布图 (前3个样本)
    for i in range(min(3, len(misc_scaled))):
        analyzer.plot_shap_waterfall(misc_scaled, shap_values, i)
    
    # 分析最重要的特征
    importance_df = analyzer.analyze_top_features(shap_values)
    
    print("\n🎉 SHAP分析完成!")
    print("📄 生成的文件:")
    print("  - random_forest_model.pkl: 训练好的模型")
    print("  - feature_scaler.pkl: 特征标准化器")
    print("  - shap_feature_importance.xlsx: SHAP特征重要性")
    print("  - shap_summary_plot.png: SHAP摘要图")
    print("  - shap_bar_plot.png: SHAP条形图")
    print("  - shap_waterfall_sample_*.png: SHAP瀑布图")

if __name__ == "__main__":
    main()
