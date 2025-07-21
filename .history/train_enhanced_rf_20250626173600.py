#!/usr/bin/env python3
"""
使用增强特征集训练Random Forest模型
添加良性特征以降低误报率
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from original_feature_extractor import OriginalVBAFeatureExtractor

class EnhancedRFTrainer:
    def __init__(self):
        """初始化增强RF训练器"""
        self.extractor = OriginalVBAFeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        
    def extract_features_from_folder(self, folder_path, label):
        """从文件夹提取特征"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return None, None
        
        # 获取所有Office文件
        office_files = []
        for ext in ['*.xls', '*.xlsx', '*.doc', '*.docx']:
            office_files.extend(folder_path.glob(ext))
        
        # 处理无扩展名文件
        for file_path in folder_path.iterdir():
            if file_path.is_file() and not file_path.suffix:
                office_files.append(file_path)
        
        if not office_files:
            print(f"❌ 在 {folder_path} 中未找到Office文件")
            return None, None
        
        print(f"🔍 处理 {folder_path.name}: {len(office_files)} 个文件")
        
        features_list = []
        labels_list = []
        successful_count = 0
        
        for i, file_path in enumerate(office_files, 1):
            if i % 100 == 0:
                print(f"  处理进度: {i}/{len(office_files)}")
            
            # 提取特征
            features = self.extractor.extract_features_from_file(file_path)
            if features and len(features) == 139:  # 新的特征维度
                features_list.append(features[1:])  # 去掉文件名
                labels_list.append(label)
                successful_count += 1
        
        print(f"✅ 成功提取 {successful_count}/{len(office_files)} 个文件的特征")
        
        if not features_list:
            return None, None
        
        return np.array(features_list), np.array(labels_list)
    
    def prepare_training_data(self, benign_folder, malicious_folder):
        """准备训练数据"""
        print("📊 准备训练数据...")
        
        # 提取良性样本特征
        benign_features, benign_labels = self.extract_features_from_folder(benign_folder, 0)
        
        # 提取恶意样本特征
        malicious_features, malicious_labels = self.extract_features_from_folder(malicious_folder, 1)
        
        if benign_features is None or malicious_features is None:
            print("❌ 特征提取失败")
            return None, None
        
        # 合并数据
        X = np.vstack([benign_features, malicious_features])
        y = np.hstack([benign_labels, malicious_labels])
        
        print(f"📈 数据集统计:")
        print(f"  总样本数: {len(X)}")
        print(f"  良性样本: {len(benign_labels)} ({len(benign_labels)/len(y)*100:.1f}%)")
        print(f"  恶意样本: {len(malicious_labels)} ({len(malicious_labels)/len(y)*100:.1f}%)")
        print(f"  特征维度: {X.shape[1]}")
        
        # 创建特征名称
        self.feature_names = (['FILENAME'] + 
                             [f'FEATURE_{i+1}' for i in range(77)] + 
                             [f'SUSPICIOUS_{i+1}' for i in range(46)] +
                             [f'BENIGN_{i+1}' for i in range(15)])
        
        return X, y
    
    def train_enhanced_rf(self, X, y, optimize_params=True):
        """训练增强版Random Forest模型"""
        print("🚀 训练增强版Random Forest模型...")
        
        # 分割训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if optimize_params:
            # 参数优化
            print("🔧 进行参数优化...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_rf = grid_search.best_estimator_
            print(f"✅ 最佳参数: {grid_search.best_params_}")
        else:
            # 使用默认参数但针对误报优化
            best_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',  # 平衡类别权重
                random_state=42
            )
            best_rf.fit(X_train, y_train)
        
        # 评估模型
        self._evaluate_model(best_rf, X_train, X_test, y_train, y_test)
        
        # 分析特征重要性
        self._analyze_feature_importance(best_rf)
        
        self.models['RandomForest'] = best_rf
        return best_rf
    
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """评估模型性能"""
        print("\n📊 模型性能评估:")
        
        # 训练集性能
        train_pred = model.predict(X_train)
        train_score = model.score(X_train, y_train)
        print(f"  训练集准确率: {train_score:.4f}")
        
        # 测试集性能
        test_pred = model.predict(X_test)
        test_score = model.score(X_test, y_test)
        print(f"  测试集准确率: {test_score:.4f}")
        
        # 详细分类报告
        print("\n📋 分类报告:")
        print(classification_report(y_test, test_pred, 
                                  target_names=['良性', '恶意']))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, test_pred)
        print(f"\n📊 混淆矩阵:")
        print(f"  真负例(TN): {cm[0,0]}, 假正例(FP): {cm[0,1]}")
        print(f"  假负例(FN): {cm[1,0]}, 真正例(TP): {cm[1,1]}")
        
        # 误报率和检测率
        fp_rate = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        detection_rate = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
        
        print(f"  误报率: {fp_rate:.4f} ({fp_rate*100:.2f}%)")
        print(f"  检测率: {detection_rate:.4f} ({detection_rate*100:.2f}%)")
        
        # ROC AUC
        test_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, test_proba)
        print(f"  ROC AUC: {auc_score:.4f}")
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"  交叉验证F1: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    
    def _analyze_feature_importance(self, model):
        """分析特征重要性"""
        print("\n🔍 特征重要性分析:")
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names[1:],  # 去掉文件名
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Top 20 重要特征
        print("Top 20 重要特征:")
        for i, (_, row) in enumerate(feature_importance_df.head(20).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # 分析特征类型
        obfuscation_imp = feature_importance_df[
            feature_importance_df['feature'].str.startswith('FEATURE_')
        ]['importance'].mean()
        
        suspicious_imp = feature_importance_df[
            feature_importance_df['feature'].str.startswith('SUSPICIOUS_')
        ]['importance'].mean()
        
        benign_imp = feature_importance_df[
            feature_importance_df['feature'].str.startswith('BENIGN_')
        ]['importance'].mean()
        
        print(f"\n📊 特征类型平均重要性:")
        print(f"  混淆特征: {obfuscation_imp:.4f}")
        print(f"  可疑特征: {suspicious_imp:.4f}")
        print(f"  良性特征: {benign_imp:.4f}")
        
        return feature_importance_df
    
    def save_models(self, output_dir='models_enhanced'):
        """保存训练好的模型"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存Random Forest模型
        if 'RandomForest' in self.models:
            model_path = output_dir / 'randomforest_enhanced_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.models['RandomForest'], f)
            print(f"✅ Random Forest模型已保存: {model_path}")
        
        # 保存特征名称
        if self.feature_names:
            feature_path = output_dir / 'enhanced_feature_columns.pkl'
            with open(feature_path, 'wb') as f:
                pickle.dump(self.feature_names, f)
            print(f"✅ 特征名称已保存: {feature_path}")

def main():
    """主函数"""
    print("🚀 增强版Random Forest训练器")
    print("=" * 50)
    
    trainer = EnhancedRFTrainer()
    
    # 准备训练数据
    X, y = trainer.prepare_training_data('data/good250623', 'data/bad250623')
    
    if X is None:
        print("❌ 数据准备失败")
        return
    
    # 训练模型
    rf_model = trainer.train_enhanced_rf(X, y, optimize_params=False)
    
    # 保存模型
    trainer.save_models()
    
    print("\n✅ 训练完成！")
    print("💡 建议:")
    print("  1. 使用新模型测试误报率")
    print("  2. 调整分类阈值进一步优化")
    print("  3. 收集更多良性样本进行训练")

if __name__ == "__main__":
    main()
