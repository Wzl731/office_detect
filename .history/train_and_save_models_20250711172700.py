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
    def __init__(self, dataset_file='ds_date/combined_dataset.csv', models_dir=None):
        """初始化模型训练器"""
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.dataset_file = dataset_file

        # 创建带时间戳的模型目录
        if models_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.models_dir = Path(f"models_{timestamp}")
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

        # 加载数据集（支持 CSV 和 Excel 格式）
        try:
            if self.dataset_file.endswith('.csv'):
                self.dataset = pd.read_csv(self.dataset_file)
                print(f"  ✅ CSV数据集加载成功: {len(self.dataset)} 条记录")
            else:
                self.dataset = pd.read_excel(self.dataset_file)
                print(f"  ✅ Excel数据集加载成功: {len(self.dataset)} 条记录")
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
    parser = argparse.ArgumentParser(description='VBA恶意宏检测模型训练器')

    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 特征提取命令
    extract_parser = subparsers.add_parser('extract', help='从原始文件提取特征')
    extract_parser.add_argument('--benign-folder', '-b', required=True, help='良性样本文件夹路径')
    extract_parser.add_argument('--malicious-folder', '-m', required=True, help='恶意样本文件夹路径')
    extract_parser.add_argument('--output', '-o', default='processed_dataset.xls', help='输出数据集文件路径')
    extract_parser.add_argument('--feature-extractor', '-f', default='feature222',
                               choices=['original', 'feature222', 'feature_11111'],
                               help='特征提取模块选择')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--dataset', '-d', default='ds_date/combined_dataset.csv', help='训练数据集文件路径')
    train_parser.add_argument('--models-dir', default='models', help='模型保存目录')

    # 完整流程命令
    full_parser = subparsers.add_parser('full', help='完整流程：特征提取 + 训练')
    full_parser.add_argument('--benign-folder', '-b', required=True, help='良性样本文件夹路径')
    full_parser.add_argument('--malicious-folder', '-m', required=True, help='恶意样本文件夹路径')
    full_parser.add_argument('--feature-extractor', '-f', default='feature222',
                            choices=['original', 'feature222', 'feature_11111'],
                            help='特征提取模块选择')
    full_parser.add_argument('--models-dir', default='models', help='模型保存目录')

    args = parser.parse_args()

    if args.command == 'extract':
        # 特征提取
        print("🔄 开始特征提取...")
        processor = DatasetProcessor(args.feature_extractor)
        dataset = processor.process_datasets(args.benign_folder, args.malicious_folder, args.output)

        if dataset is not None:
            print("✅ 特征提取完成！")
        else:
            print("❌ 特征提取失败！")

    elif args.command == 'train':
        # 模型训练
        print("🚀 开始模型训练...")
        trainer = VBAMalwareModelTrainer(args.dataset)
        success = trainer.run()

        if success:
            print("✅ 模型训练完成！")
        else:
            print("❌ 模型训练失败！")

    elif args.command == 'full':
        # 完整流程
        print("🎯 开始完整训练流程...")

        # 1. 特征提取
        print("\n📊 步骤1: 特征提取")
        processor = DatasetProcessor(args.feature_extractor)
        dataset_file = 'temp_dataset.xls'
        dataset = processor.process_datasets(args.benign_folder, args.malicious_folder, dataset_file)

        if dataset is None:
            print("❌ 特征提取失败！")
            return

        # 2. 模型训练
        print("\n🚀 步骤2: 模型训练")
        trainer = VBAMalwareModelTrainer(dataset_file)
        success = trainer.run()

        if success:
            print("✅ 完整流程执行成功！")
        else:
            print("❌ 模型训练失败！")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()