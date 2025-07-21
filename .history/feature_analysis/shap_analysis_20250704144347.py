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
    
    def analyze_all_samples(self):
        """分析误报样本、白样本和黑样本的SHAP值进行对比"""
        print("\n🎯 分析所有样本类型...")

        # 1. 准备误报样本数据
        misc_features = self.misclassified_data[self.feature_names].fillna(0)
        misc_scaled = self.scaler.transform(misc_features)

        # 2. 准备白样本数据（随机采样，避免计算量过大）
        white_features = self.white_data[self.feature_names].fillna(0)
        white_sample_size = min(200, len(white_features))  # 最多200个样本
        white_sample_idx = np.random.choice(len(white_features), white_sample_size, replace=False)
        white_sample = white_features.iloc[white_sample_idx]
        white_scaled = self.scaler.transform(white_sample)

        # 3. 准备黑样本数据（随机采样）
        black_features = self.black_data[self.feature_names].fillna(0)
        black_sample_size = min(200, len(black_features))  # 最多200个样本
        black_sample_idx = np.random.choice(len(black_features), black_sample_size, replace=False)
        black_sample = black_features.iloc[black_sample_idx]
        black_scaled = self.scaler.transform(black_sample)

        print(f"✅ 样本数量:")
        print(f"  - 误报样本: {len(misc_scaled)}")
        print(f"  - 白样本: {len(white_scaled)}")
        print(f"  - 黑样本: {len(black_scaled)}")

        # 预测所有样本
        misc_pred = self.model.predict(misc_scaled)
        misc_prob = self.model.predict_proba(misc_scaled)
        white_pred = self.model.predict(white_scaled)
        white_prob = self.model.predict_proba(white_scaled)
        black_pred = self.model.predict(black_scaled)
        black_prob = self.model.predict_proba(black_scaled)

        print(f"\n📊 预测结果:")
        print(f"  误报样本 - 预测为恶意: {np.sum(misc_pred == 1)}/{len(misc_pred)} ({np.sum(misc_pred == 1)/len(misc_pred)*100:.1f}%)")
        print(f"  白样本   - 预测为恶意: {np.sum(white_pred == 1)}/{len(white_pred)} ({np.sum(white_pred == 1)/len(white_pred)*100:.1f}%)")
        print(f"  黑样本   - 预测为恶意: {np.sum(black_pred == 1)}/{len(black_pred)} ({np.sum(black_pred == 1)/len(black_pred)*100:.1f}%)")

        # 计算SHAP值
        print("\n🔍 计算SHAP值...")
        print("  - 计算误报样本SHAP值...")
        misc_shap = self.explainer.shap_values(misc_scaled)
        print("  - 计算白样本SHAP值...")
        white_shap = self.explainer.shap_values(white_scaled)
        print("  - 计算黑样本SHAP值...")
        black_shap = self.explainer.shap_values(black_scaled)

        # 如果是二分类，取恶意类的SHAP值
        if isinstance(misc_shap, list):
            misc_shap_malicious = misc_shap[1]
            white_shap_malicious = white_shap[1]
            black_shap_malicious = black_shap[1]
        else:
            misc_shap_malicious = misc_shap
            white_shap_malicious = white_shap
            black_shap_malicious = black_shap

        return {
            'misc': {'scaled': misc_scaled, 'shap': misc_shap_malicious, 'pred': misc_pred, 'prob': misc_prob},
            'white': {'scaled': white_scaled, 'shap': white_shap_malicious, 'pred': white_pred, 'prob': white_prob},
            'black': {'scaled': black_scaled, 'shap': black_shap_malicious, 'pred': black_pred, 'prob': black_prob}
        }
    
    def plot_shap_comparison(self, all_results):
        """绘制SHAP对比分析图"""
        print("\n📈 绘制SHAP对比分析图...")

        misc_data = all_results['misc']
        white_data = all_results['white']
        black_data = all_results['black']

        # 1. 误报样本SHAP摘要图
        plt.figure(figsize=(12, 8))
        misc_df = pd.DataFrame(misc_data['scaled'], columns=self.feature_names)
        shap.summary_plot(misc_data['shap'], misc_df, max_display=20, show=False)
        plt.title('SHAP特征重要性摘要 - 误报样本', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_summary_misclassified.png', dpi=300, bbox_inches='tight')
        print("✅ 误报样本SHAP摘要图已保存到: shap_summary_misclassified.png")
        plt.show()

        # 2. 白样本SHAP摘要图
        plt.figure(figsize=(12, 8))
        white_df = pd.DataFrame(white_data['scaled'], columns=self.feature_names)
        shap.summary_plot(white_data['shap'], white_df, max_display=20, show=False)
        plt.title('SHAP特征重要性摘要 - 白样本', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_summary_white.png', dpi=300, bbox_inches='tight')
        print("✅ 白样本SHAP摘要图已保存到: shap_summary_white.png")
        plt.show()

        # 3. 黑样本SHAP摘要图
        plt.figure(figsize=(12, 8))
        black_df = pd.DataFrame(black_data['scaled'], columns=self.feature_names)
        shap.summary_plot(black_data['shap'], black_df, max_display=20, show=False)
        plt.title('SHAP特征重要性摘要 - 黑样本', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_summary_black.png', dpi=300, bbox_inches='tight')
        print("✅ 黑样本SHAP摘要图已保存到: shap_summary_black.png")
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
    
    def analyze_shap_comparison(self, all_results):
        """对比分析不同样本类型的SHAP值"""
        print("\n🔍 对比分析SHAP值...")

        misc_shap = all_results['misc']['shap']
        white_shap = all_results['white']['shap']
        black_shap = all_results['black']['shap']

        # 计算各类样本的特征重要性
        misc_importance = np.abs(misc_shap).mean(axis=0)
        white_importance = np.abs(white_shap).mean(axis=0)
        black_importance = np.abs(black_shap).mean(axis=0)

        # 计算平均SHAP值（带符号）
        misc_mean_shap = misc_shap.mean(axis=0)
        white_mean_shap = white_shap.mean(axis=0)
        black_mean_shap = black_shap.mean(axis=0)

        # 创建对比DataFrame
        comparison_df = pd.DataFrame({
            'feature': self.feature_names,
            'misc_importance': misc_importance,
            'white_importance': white_importance,
            'black_importance': black_importance,
            'misc_mean_shap': misc_mean_shap,
            'white_mean_shap': white_mean_shap,
            'black_mean_shap': black_mean_shap,
        })

        # 计算误报样本与黑样本的相似度
        comparison_df['misc_black_similarity'] = 1 - np.abs(comparison_df['misc_mean_shap'] - comparison_df['black_mean_shap']) / (np.abs(comparison_df['misc_mean_shap']) + np.abs(comparison_df['black_mean_shap']) + 1e-8)

        # 按误报样本重要性排序
        comparison_df = comparison_df.sort_values('misc_importance', ascending=False)

        # 保存结果
        comparison_df.to_excel('shap_comparison_analysis.xlsx', index=False)
        print("✅ SHAP对比分析已保存到: shap_comparison_analysis.xlsx")

        # 显示关键发现
        print("\n📋 关键发现:")
        print("=" * 80)

        # 1. 误报样本中最重要的特征
        print("� 误报样本中最重要的前10个特征:")
        for idx, row in comparison_df.head(10).iterrows():
            print(f"  {row['feature']:25} | 误报重要性:{row['misc_importance']:.4f} | 黑样本重要性:{row['black_importance']:.4f}")

        # 2. 误报样本与黑样本SHAP值相似的特征
        similar_features = comparison_df[comparison_df['misc_black_similarity'] > 0.7].head(10)
        print(f"\n🎯 误报样本与黑样本SHAP值相似的特征 (相似度>0.7):")
        for idx, row in similar_features.iterrows():
            print(f"  {row['feature']:25} | 相似度:{row['misc_black_similarity']:.3f} | 误报SHAP:{row['misc_mean_shap']:.4f} | 黑样本SHAP:{row['black_mean_shap']:.4f}")

        # 3. 可能导致误报的关键特征
        problematic_features = comparison_df[
            (comparison_df['misc_importance'] > comparison_df['misc_importance'].quantile(0.8)) &
            (comparison_df['misc_mean_shap'] * comparison_df['black_mean_shap'] > 0) &  # 同号
            (np.abs(comparison_df['misc_mean_shap'] - comparison_df['black_mean_shap']) < 0.1)  # 数值接近
        ]

        print(f"\n⚠️  可能导致误报的关键特征 (高重要性且与黑样本相似):")
        for idx, row in problematic_features.head(10).iterrows():
            print(f"  {row['feature']:25} | 误报SHAP:{row['misc_mean_shap']:.4f} | 黑样本SHAP:{row['black_mean_shap']:.4f} | 差异:{abs(row['misc_mean_shap']-row['black_mean_shap']):.4f}")

        return comparison_df

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
