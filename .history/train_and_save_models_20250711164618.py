#!/usr/bin/env python3
"""
训练VBA恶意宏检测模型并保存
基于原始项目的机器学习模型，训练后保存到models文件夹
"""

import numpy as np
import pandas as pd
import pickle
import os
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 动态导入特征提取模块
def import_feature_extractor(module_name='feature222'):
    """动态导入特征提取模块"""
    try:
        if module_name == 'original':
            from original_feature_extractor import OriginalVBAFeatureExtractor
        elif module_name == 'feature222':
            from feature222 import OriginalVBAFeatureExtractor
        elif module_name == 'feature_11111':
            from feature_11111 import OriginalVBAFeatureExtractor
        else:
            raise ImportError(f"未知的特征提取模块: {module_name}")

        print(f"✅ 成功导入特征提取模块: {module_name}")
        return OriginalVBAFeatureExtractor
    except ImportError as e:
        print(f"❌ 导入特征提取模块失败: {e}")
        return None


class DatasetProcessor:
    """数据集处理器 - 负责特征提取和数据集生成"""

    def __init__(self, feature_extractor_name='feature222'):
        """初始化数据集处理器"""
        self.feature_extractor_class = import_feature_extractor(feature_extractor_name)
        if self.feature_extractor_class is None:
            raise RuntimeError("特征提取模块导入失败")

        self.extractor = self.feature_extractor_class()

    def extract_features_from_folder(self, folder_path, label, output_file=None):
        """从文件夹提取特征并保存"""
        folder_path = Path(folder_path)

        if not folder_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return None

        # 获取所有文件
        files = [f for f in folder_path.iterdir() if f.is_file()]
        if not files:
            print(f"❌ 文件夹为空: {folder_path}")
            return None

        print(f"📁 处理文件夹: {folder_path} (标签: {'恶意' if label == 1 else '良性'})")
        print(f"📄 文件数量: {len(files)}")

        features_list = []
        successful_count = 0

        for i, file_path in enumerate(files, 1):
            if i % 100 == 0:
                print(f"  处理进度: {i}/{len(files)}")

            try:
                # 提取特征
                features = self.extractor.extract_features_from_file(file_path)
                if features is not None:
                    # 添加标签
                    features.append(label)
                    features_list.append(features)
                    successful_count += 1
            except Exception as e:
                print(f"  ❌ 处理失败 {file_path.name}: {e}")

        print(f"✅ 成功处理: {successful_count}/{len(files)} 个文件")

        if not features_list:
            print("❌ 没有成功提取任何特征")
            return None

        # 转换为DataFrame
        feature_names = self.extractor.get_feature_names() + ['label']
        df = pd.DataFrame(features_list, columns=feature_names)

        # 保存到文件
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(output_path, index=False)
            print(f"💾 特征数据已保存到: {output_path}")

        return df

    def process_datasets(self, benign_folder, malicious_folder, output_file='processed_dataset.xls'):
        """处理良性和恶意数据集"""
        print("🔄 开始处理数据集...")

        # 处理良性样本
        benign_df = self.extract_features_from_folder(benign_folder, label=0)
        if benign_df is None:
            return None

        # 处理恶意样本
        malicious_df = self.extract_features_from_folder(malicious_folder, label=1)
        if malicious_df is None:
            return None

        # 合并数据集
        combined_df = pd.concat([benign_df, malicious_df], ignore_index=True)

        # 保存合并后的数据集
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_excel(output_path, index=False)

        print(f"✅ 数据集处理完成:")
        print(f"  📊 良性样本: {len(benign_df)}")
        print(f"  📊 恶意样本: {len(malicious_df)}")
        print(f"  📊 总样本数: {len(combined_df)}")
        print(f"  💾 已保存到: {output_path}")

        return combined_df

class VBAMalwareModelTrainer:
    def __init__(self):
        """初始化模型训练器"""
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        
        # 数据集配置
        self.DS1_BENIGN_SAMPLES_CNT = 2939
        self.DS1_MAL_SAMPLES_CNT = 13734
        
        # 模型配置 
        # TODO 优化参数以提高性能 
        self.model_configs = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),
                'need_scaling': False
            },
            'MLP': {
                'model': MLPClassifier(
                    hidden_layer_sizes=(150,),
                    max_iter=2000,  # 增加迭代次数
                    alpha=1e-4,
                    learning_rate='adaptive',
                    early_stopping=True,  # 早停
                    validation_fraction=0.1,
                    random_state=42
                ),
                'need_scaling': True
            },
            'KNN': {
                'model': KNeighborsClassifier(
                    n_neighbors=3,
                    weights='distance',  # 使用距离权重
                    n_jobs=-1
                ),
                'need_scaling': True
            },
            'SVM': {
                'model': SVC(
                    kernel='rbf',  # 使用RBF核
                    C=1.0,
                    gamma='scale',
                    random_state=42,
                    probability=True
                ),
                'need_scaling': True
            }
        }
    
    def load_dataset(self):
        """加载数据集"""
        print("📊 加载数据集...")
        
        # 加载数据集1
        try:
            self.dataset1 = pd.read_excel('ds1.xls')
            print(f"  ✅ 数据集1加载成功: {len(self.dataset1)} 条记录")
        except Exception as e:
            print(f"  ❌ 数据集1加载失败: {e}")
            return False
        
        # 创建标签
        self.labels = [0] * self.DS1_BENIGN_SAMPLES_CNT + [1] * self.DS1_MAL_SAMPLES_CNT
        
        # 保存特征列名
        self.feature_columns = self.dataset1.columns[1:124].tolist()  # 排除文件名列
        
        print(f"  📋 特征维度: {len(self.feature_columns)}")
        print(f"  📋 良性样本: {self.DS1_BENIGN_SAMPLES_CNT}")
        print(f"  📋 恶意样本: {self.DS1_MAL_SAMPLES_CNT}")
        
        return True
    
    def prepare_data(self):
        """准备训练数据"""
        print("🔧 准备训练数据...")
        
        # 提取特征和标签
        X = self.dataset1.iloc[:, 1:124].values  # 123维特征
        y = np.array(self.labels)
        
        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  📊 训练集大小: {len(self.X_train)}")
        print(f"  📊 测试集大小: {len(self.X_test)}")
        
        return True
    
    def train_models(self):
        """训练所有模型"""
        print("🚀 开始训练模型...")
        
        for model_name, config in self.model_configs.items():
            print(f"\n   训练 {model_name} 模型...")
            
            try:
                model = config['model']
                
                # 准备训练数据
                X_train = self.X_train.copy()
                X_test = self.X_test.copy()
                
                # 如果需要标准化
                if config['need_scaling']:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    self.scalers[model_name] = scaler
                    print(f"    ✅ 数据标准化完成")
                
                # 训练模型
                model.fit(X_train, self.y_train)
                
                # 评估模型
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                
                print(f"    ✅ {model_name} 训练完成")
                print(f"    📈 测试集准确率: {accuracy:.4f}")
                
                # 保存模型
                self.models[model_name] = model
                
            except Exception as e:
                print(f"    ❌ {model_name} 训练失败: {e}")
    
    def save_models(self):
        """保存训练好的模型"""
        print("\n💾 保存模型...")
        
        # 创建models目录
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # 保存每个模型
        for model_name, model in self.models.items():
            try:
                # 保存模型
                model_path = models_dir / f'{model_name.lower()}_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  ✅ {model_name} 模型已保存: {model_path}")
                
                # 保存对应的标准化器（如果有）
                if model_name in self.scalers:
                    scaler_path = models_dir / f'{model_name.lower()}_scaler.pkl'
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[model_name], f)
                    print(f"  ✅ {model_name} 标准化器已保存: {scaler_path}")
                    
            except Exception as e:
                print(f"  ❌ 保存 {model_name} 失败: {e}")
        
        # 保存特征列名
        try:
            feature_path = models_dir / 'feature_columns.pkl'
            with open(feature_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)
            print(f"  ✅ 特征列名已保存: {feature_path}")
        except Exception as e:
            print(f"  ❌ 保存特征列名失败: {e}")
    
    def generate_detailed_report(self):
        """生成详细的模型评估报告"""
        print("\n📊 生成模型评估报告...")
        
        report_path = Path('models') / 'model_evaluation_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("VBA恶意宏检测模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"数据集信息:\n")
            f.write(f"  - 总样本数: {len(self.dataset1)}\n")
            f.write(f"  - 良性样本: {self.DS1_BENIGN_SAMPLES_CNT}\n")
            f.write(f"  - 恶意样本: {self.DS1_MAL_SAMPLES_CNT}\n")
            f.write(f"  - 特征维度: {len(self.feature_columns)}\n\n")
            
            for model_name, model in self.models.items():
                f.write(f"{model_name} 模型评估:\n")
                f.write("-" * 30 + "\n")
                
                try:
                    # 准备测试数据
                    X_test = self.X_test.copy()
                    if model_name in self.scalers:
                        X_test = self.scalers[model_name].transform(X_test)
                    
                    # 预测
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(self.y_test, y_pred)
                    
                    f.write(f"准确率: {accuracy:.4f}\n")
                    f.write(f"分类报告:\n{classification_report(self.y_test, y_pred)}\n")
                    f.write(f"混淆矩阵:\n{confusion_matrix(self.y_test, y_pred)}\n\n")
                    
                except Exception as e:
                    f.write(f"评估失败: {e}\n\n")
        
        print(f"  ✅ 评估报告已保存: {report_path}")
    
    def run(self):
        """运行完整的训练流程"""
        print("🎯 VBA恶意宏检测模型训练器")
        print("=" * 50)
        
        # 加载数据集
        if not self.load_dataset():
            return False
        
        # 准备数据
        if not self.prepare_data():
            return False
        
        # 训练模型
        self.train_models()
        
        # 保存模型
        self.save_models()
        
        # 生成报告
        self.generate_detailed_report()
        
        print("\n🎉 模型训练完成！")
        print(f"📁 模型文件保存在: ./models/")
        
        return True

def main():
    """主函数"""
    trainer = VBAMalwareModelTrainer()
    success = trainer.run()
    
    if success:
        print("\n✅ 所有模型训练并保存成功！")
    else:
        print("\n❌ 模型训练失败！")

if __name__ == "__main__":
    main()

#python detector.py --no-save
#python detector.py --save-type benign
#python detector.py --save-type malicious