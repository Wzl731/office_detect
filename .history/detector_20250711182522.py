import numpy as np
import pandas as pd
import pickle
import argparse
import shutil
from pathlib import Path
from feature222 import OriginalVBAFeatureExtractor as extractor
import warnings
warnings.filterwarnings('ignore')

# 模型配置常量
MODEL_FILES = {
    'RandomForest': 'randomforest_model.pkl',
    'MLP': 'mlp_model.pkl',
    'KNN': 'knn_model.pkl',
    'SVM': 'svm_model.pkl'
}

SCALER_FILES = {
    'MLP': 'mlp_scaler.pkl',
    'KNN': 'knn_scaler.pkl',
    'SVM': 'svm_scaler.pkl'
}

class OfficeDetector:
    def __init__(self, models_dir='models_randomforest_mlp_knn_svm_20250711_175424'):
        """初始化检测器 - 只负责核心检测功能"""
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}

    def load_models(self):
        """加载训练好的模型"""
        # 直接加载模型
        for model_name, model_file in MODEL_FILES.items():
            model_path = self.models_dir / model_file
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)

            # 加载对应的标准化器
            if model_name in SCALER_FILES:
                scaler_path = self.models_dir / SCALER_FILES[model_name]
                with open(scaler_path, 'rb') as f:
                    self.scalers[model_name] = pickle.load(f)

        print(f"成功加载 {len(self.models)} 个模型")

        # 获取特征列名并验证
        feature_names = extractor().get_feature_names()[1:]  # 跳过文件名
        print(f"获取特征列名: {len(feature_names)} 个特征")

        return True

    def extract_features(self, file_path):
        """提取文件特征"""
        try:
            features_list = extractor().extract_features_from_file(file_path)
            if features_list is None:
                return None

            # 提取数值特征 (跳过文件名)
            features = np.array(features_list[1:]).reshape(1, -1)
            return features

        except Exception as e:
            print(f"  ❌ {file_path.name}: 特征提取失败 - {e}")
            return None

    def predict_file(self, file_path, features=None):
        """预测单个文件"""
        # 如果没有提供特征，则使用类内的特征提取方法
        if features is None:
            features = self.extract_features(file_path)
            if features is None:
                return None

        results = {}

        for model_name, model in self.models.items():
            try:
                # 准备特征数据
                X = features.copy()

                # 如果需要标准化
                if model_name in self.scalers:
                    X = self.scalers[model_name].transform(X)

                # 预测
                prediction = model.predict(X)[0]

                # 获取预测概率（如果支持）
                try:
                    probabilities = model.predict_proba(X)[0]
                    confidence = max(probabilities)
                except:
                    confidence = None

                results[model_name] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'label': '恶意' if prediction == 1 else '良性'
                }

            except Exception as e:
                print(f"  ❌ {model_name} 预测失败: {e}")
                results[model_name] = {
                    'prediction': None,
                    'confidence': None,
                    'label': '错误'
                }

        # 集成投票决策
        malicious_votes = 0
        for result in results.values():
            if result['prediction'] == 1:
                malicious_votes += 1

        # 如果认为恶意的模型大于等于3个，则判定为恶意
        if malicious_votes >= 3:
            ensemble_result = '恶意'
        else:
            ensemble_result = '良性'

        # 添加集成结果到返回值
        results['ensemble'] = {
            'prediction': 1 if ensemble_result == '恶意' else 0,
            'malicious_votes': malicious_votes,
            'total_votes': len(self.models),  # 使用总模型数量
            'label': ensemble_result
        }

        return results

    def predict_folder(self, folder_path, save_files=True, save_type='all'):
        """预测文件夹中的所有文件"""
        folder_path = Path(folder_path)

        if not folder_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return

        # 获取所有文件
        all_files = [f for f in folder_path.iterdir() if f.is_file()]

        if not all_files:
            print(f"❌ 在 {folder_path} 中未找到文件")
            return

        print(f"开始测试 {len(all_files)} 个文件...")
        print("=" * 80)

        # 初始化统计
        total_files = len(all_files)
        successful_predictions = 0
        model_stats = {model_name: {'malicious': 0, 'benign': 0, 'errors': 0}
                      for model_name in self.models.keys()}
        ensemble_stats = {'malicious': 0, 'benign': 0, 'errors': 0}

        # 分类文件列表
        malicious_files = []
        benign_files = []

        # 处理每个文件
        for i, file_path in enumerate(all_files, 1):
            print(f"\n[{i}/{total_files}]  分析: {file_path.name}")

            # 直接使用已验证的 predict_file 方法
            results = self.predict_file(file_path)

            if results:
                successful_predictions += 1

                # 显示预测结果
                print("   预测结果:")
                for model_name, result in results.items():
                    if model_name == 'ensemble':
                        continue

                    if result['prediction'] is not None:
                        label = result['label']
                        confidence = f" (置信度: {result['confidence']:.3f})" if result['confidence'] else ""
                        print(f"    {model_name:12}: {label}{confidence}")

                        # 更新统计
                        if result['prediction'] == 1:
                            model_stats[model_name]['malicious'] += 1
                        else:
                            model_stats[model_name]['benign'] += 1
                    else:
                        model_stats[model_name]['errors'] += 1
                        print(f"    {model_name:12}: 预测失败")

                # 显示集成结果
                ensemble = results['ensemble']
                consensus_label = f"🚨 {ensemble['label']}" if ensemble['label'] == '恶意' else f"✅ {ensemble['label']}"
                print(f"   集成结果: {consensus_label} ({ensemble['malicious_votes']}/{ensemble['total_votes']}票)")

                # 根据集成结果分类文件
                if results['ensemble']['prediction'] == 1:
                    malicious_files.append(file_path)
                    ensemble_stats['malicious'] += 1
                else:
                    benign_files.append(file_path)
                    ensemble_stats['benign'] += 1
            else:
                print("  ❌ 分析失败")
                ensemble_stats['errors'] += 1

        # 显示总结
        print_test_summary(total_files, successful_predictions, model_stats, ensemble_stats)

        # 保存文件
        if save_type in ['all', 'malicious']:
            save_files_by_type(malicious_files, '恶意', save_files)
        if save_type in ['all', 'benign']:
            save_files_by_type(benign_files, '良性', save_files)




# 文件保存函数
def save_files_by_type(files_list, file_type, save_files=True):
    """通用文件保存方法"""
    if not files_list:
        print(f"\n📁 未检测到{file_type}文件，无需保存")
        return

    if not save_files:
        print(f"\n📁 检测出 {len(files_list)} 个{file_type}文件 (已禁用保存功能)")
        return

    # 根据文件类型确定目标目录
    target_dirs = {
        '恶意': 'data/good2bad2',
        '良性': 'data/bad2good2'
    }

    target_dir = Path(target_dirs.get(file_type, f'data/{file_type}'))
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 正在保存 {len(files_list)} 个{file_type}文件到: {target_dir}")

    saved_count = 0
    skipped_count = 0
    error_count = 0

    for source_file in files_list:
        try:
            # 目标文件路径
            target_file = target_dir / source_file.name

            # 如果目标文件已存在，跳过
            if target_file.exists():
                skipped_count += 1
                continue

            # 复制文件
            shutil.copy2(source_file, target_file)
            saved_count += 1

        except Exception as e:
            error_count += 1

    # 简洁的总结信息
    print(f"✅ 保存完成: {saved_count} 个新文件, {skipped_count} 个已存在, {error_count} 个失败")


def display_prediction_results(results, model_stats):
    """显示单个文件的预测结果"""
    print("   预测结果:")

    for model_name, result in results.items():
        # 跳过集成结果，只显示单个模型结果
        if model_name == 'ensemble':
            continue

        if result['prediction'] is not None:
            label = result['label']
            confidence = f" (置信度: {result['confidence']:.3f})" if result['confidence'] else ""
            print(f"    {model_name:12}: {label}{confidence}")

            # 统计
            if result['prediction'] == 1:
                model_stats[model_name]['malicious'] += 1
            else:
                model_stats[model_name]['benign'] += 1
        else:
            model_stats[model_name]['errors'] += 1
            print(f"    {model_name:12}: 预测失败")

    # 显示集成结果
    ensemble = results['ensemble']
    consensus_label = f"🚨 {ensemble['label']}" if ensemble['label'] == '恶意' else f"✅ {ensemble['label']}"
    print(f"   集成结果: {consensus_label} ({ensemble['malicious_votes']}/{ensemble['total_votes']}票)")

    return ensemble['label']


# 结果处理相关的独立函数

def print_test_summary(total_files, successful_predictions, model_stats, ensemble_stats):
    """打印测试总结"""
    print("\n" + "=" * 80)
    print("📈 测试总结")
    print("=" * 80)

    print(f"📁 总文件数: {total_files}")
    print(f"✅ 成功分析: {successful_predictions}")
    print(f"❌ 分析失败: {total_files - successful_predictions}")

    print("\n📊 各模型检测统计:")
    for model_name, stats in model_stats.items():
        total = stats['malicious'] + stats['benign'] + stats['errors']
        if total > 0:
            malicious_rate = stats['malicious'] / total * 100
            benign_rate = stats['benign'] / total * 100
            error_rate = stats['errors'] / total * 100

            print(f"  {model_name:12}: 恶意 {stats['malicious']:3d} ({malicious_rate:5.1f}%) | "
                  f"良性 {stats['benign']:3d} ({benign_rate:5.1f}%) | "
                  f"错误 {stats['errors']:3d} ({error_rate:5.1f}%)")

    # 添加集成结果统计
    print("\n🎯 集成预测统计:")
    ensemble_total = ensemble_stats['malicious'] + ensemble_stats['benign'] + ensemble_stats['errors']
    if ensemble_total > 0:
        malicious_rate = ensemble_stats['malicious'] / ensemble_total * 100
        benign_rate = ensemble_stats['benign'] / ensemble_total * 100
        error_rate = ensemble_stats['errors'] / ensemble_total * 100

        print(f"  {'集成结果':12}: 恶意 {ensemble_stats['malicious']:3d} ({malicious_rate:5.1f}%) | "
              f"良性 {ensemble_stats['benign']:3d} ({benign_rate:5.1f}%) | "
              f"错误 {ensemble_stats['errors']:3d} ({error_rate:5.1f}%)")




def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='VBA恶意宏检测器 - 使用训练好的模型检测Office文件')
    parser.add_argument('--folder', '-f', default='data/bad', help='要测试的文件夹路径')
    parser.add_argument('--models-dir', '-m', default='models_0711', help='模型文件夹路径 (默认: models)')
    parser.add_argument('--no-save', action='store_true', help='禁用保存所有文件')
    parser.add_argument('--save-type', choices=['all', 'malicious', 'benign'], default='all',
                       help='选择保存文件类型: all(全部), malicious(仅恶意), benign(仅良性) (默认: all)')

    args = parser.parse_args()

    print("🎯 VBA恶意宏检测器 (原始特征格式)")
    print("=" * 50)
    print(f"📁 测试文件夹: {args.folder}")
    print(f"🤖 模型目录: {args.models_dir}")
    print()

    # 初始化检测器
    detector = OfficeDetector(models_dir=args.models_dir)

    # 加载模型
    detector.load_models()

    # 预测指定文件夹
    detector.predict_folder(args.folder, save_files=not args.no_save, save_type=args.save_type)

    print("\n🎉 测试完成！")

if __name__ == "__main__":
    main()