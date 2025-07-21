#!/usr/bin/env python3
"""
训练模型并保存
训练后保存到models文件夹
"""

import numpy as np
import pandas as pd
import pickle
import os
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



class VBAMalwareModelTrainer:
    def __init__(self, dataset_file='ds_date/combined_dataset.csv', models_dir=None, selected_models=None):
        """初始化模型训练器"""
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.dataset_file = dataset_file

        # 选择要训练的模型
        if selected_models is None:
            self.selected_models = ['RandomForest', 'MLP', 'KNN', 'SVM']
        else:
            self.selected_models = selected_models

        # 创建带时间戳的模型目录
        if models_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_suffix = "_".join(self.selected_models).lower()
            self.models_dir = Path(f"models_{models_suffix}_{timestamp}")
        else:
            self.models_dir = Path(models_dir)

        # 动态分析的数据集配置（将在load_dataset中设置）
        self.benign_samples_cnt = 0
        self.malicious_samples_cnt = 0
        self.total_samples_cnt = 0
        
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
        """动态加载和分析数据集"""
        print("📊 加载数据集...")

        # 加载 CSV 数据集
        try:
            self.dataset = pd.read_csv(self.dataset_file)
            print(f"  ✅ 数据集加载成功: {len(self.dataset)} 条记录")
        except Exception as e:
            print(f"  ❌ 数据集加载失败: {e}")
            return False

        # 动态分析数据集结构
        if 'label' not in self.dataset.columns:
            print("  ❌ 数据集中未找到'label'列")
            return False

        # 统计样本数量
        label_counts = self.dataset['label'].value_counts()
        self.benign_samples_cnt = label_counts.get(0, 0)
        self.malicious_samples_cnt = label_counts.get(1, 0)
        self.total_samples_cnt = len(self.dataset)

        # 获取特征列（排除文件名和标签列）
        exclude_columns = ['label']
        if 'filename' in self.dataset.columns:
            exclude_columns.append('filename')
        elif self.dataset.columns[0].lower() in ['file', 'name', 'filename']:
            exclude_columns.append(self.dataset.columns[0])

        self.feature_columns = [col for col in self.dataset.columns if col not in exclude_columns]

        print(f"  📋 特征维度: {len(self.feature_columns)}")
        print(f"  📋 良性样本: {self.benign_samples_cnt}")
        print(f"  📋 恶意样本: {self.malicious_samples_cnt}")
        print(f"  📋 总样本数: {self.total_samples_cnt}")

        # 验证数据集
        if self.benign_samples_cnt == 0 or self.malicious_samples_cnt == 0:
            print("  ⚠️  警告: 数据集中缺少良性或恶意样本")

        return True
    
    def prepare_data(self):
        """准备训练数据"""
        print("🔧 准备训练数据...")

        # 提取特征和标签
        X = self.dataset[self.feature_columns].values
        y = self.dataset['label'].values

        print(f"  📊 特征矩阵形状: {X.shape}")
        print(f"  📊 标签数组形状: {y.shape}")

        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"  📊 训练集大小: {len(self.X_train)}")
        print(f"  📊 测试集大小: {len(self.X_test)}")
        print(f"  📊 训练集标签分布: 良性={sum(self.y_train==0)}, 恶意={sum(self.y_train==1)}")
        print(f"  📊 测试集标签分布: 良性={sum(self.y_test==0)}, 恶意={sum(self.y_test==1)}")

        return True
    
    def train_models(self):
        """训练选定的模型"""
        print(f"🚀 开始训练模型: {', '.join(self.selected_models)}")

        for model_name, config in self.model_configs.items():
            # 只训练选定的模型
            if model_name not in self.selected_models:
                print(f"   ⏭️  跳过 {model_name} 模型")
                continue
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
        print(f"\n💾 保存模型到: {self.models_dir}")

        # 创建带时间戳的models目录
        self.models_dir.mkdir(exist_ok=True)
        
        # 保存每个模型
        for model_name, model in self.models.items():
            try:
                # 保存模型
                model_path = self.models_dir / f'{model_name.lower()}_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  ✅ {model_name} 模型已保存: {model_path}")

                # 保存对应的标准化器（如果有）
                if model_name in self.scalers:
                    scaler_path = self.models_dir / f'{model_name.lower()}_scaler.pkl'
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[model_name], f)
                    print(f"  ✅ {model_name} 标准化器已保存: {scaler_path}")

            except Exception as e:
                print(f"  ❌ 保存 {model_name} 失败: {e}")

        # 保存特征列名
        try:
            feature_path = self.models_dir / 'feature_columns.pkl'
            with open(feature_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)
            print(f"  ✅ 特征列名已保存: {feature_path}")
        except Exception as e:
            print(f"  ❌ 保存特征列名失败: {e}")
    
    def generate_detailed_report(self):
        """生成详细的模型评估报告"""
        print("\n📊 生成模型评估报告...")

        report_path = self.models_dir / 'model_evaluation_report.txt'
        
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
        print(f"📁 模型文件保存在: ./{self.models_dir}/")
        print(f"📋 模型目录: {self.models_dir.absolute()}")

        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VBA恶意宏检测模型训练器')
    parser.add_argument('--dataset', '-d', default='ds_date/combined_dataset.csv', help='训练数据集文件路径')
    parser.add_argument('--models-dir', '-m', help='模型保存目录（默认自动生成时间戳目录）')
    parser.add_argument('--models', nargs='+',
                       choices=['RandomForest', 'MLP', 'KNN', 'SVM', 'all'],
                       default=['all'],
                       help='选择要训练的模型 (默认: all)')

    args = parser.parse_args()

    # 处理模型选择
    if 'all' in args.models:
        selected_models = ['RandomForest', 'MLP', 'KNN', 'SVM']
    else:
        selected_models = args.models

    print(f"🚀 开始训练模型: {', '.join(selected_models)}")
    trainer = VBAMalwareModelTrainer(args.dataset, args.models_dir, selected_models)
    success = trainer.run()

    if success:
        print("✅ 模型训练完成！")
    else:
        print("❌ 模型训练失败！")

if __name__ == "__main__":
    main()