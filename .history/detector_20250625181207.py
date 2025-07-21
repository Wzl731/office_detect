#!/usr/bin/env python3
"""
使用原始特征格式测试文件夹中的文件
支持命令行参数指定测试文件夹
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import shutil
from pathlib import Path
from original_feature_extractor import OriginalVBAFeatureExtractor
import warnings
warnings.filterwarnings('ignore')

class VBAMalwareDetectorOriginal:
    def __init__(self, models_dir='models'):
        """初始化检测器"""
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.extractor = OriginalVBAFeatureExtractor()
        
    def load_models(self):
        """加载训练好的模型"""
        print("📦 加载训练好的模型...")
        
        if not self.models_dir.exists():
            print(f"❌ 模型目录不存在: {self.models_dir}")
            return False
        
        # 加载特征列名
        feature_path = self.models_dir / 'feature_columns.pkl'
        if feature_path.exists():
            with open(feature_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            print(f"  ✅ 特征列名加载成功: {len(self.feature_columns)} 个特征")
        else:
            print("  ❌ 特征列名文件不存在")
            return False
        
        # 加载模型
        model_files = {
            'RandomForest': 'randomforest_model.pkl',
            'MLP': 'mlp_model.pkl', 
            'KNN': 'knn_model.pkl',
            'SVM': 'svm_model.pkl'
        }
        
        scaler_files = {
            'MLP': 'mlp_scaler.pkl',
            'KNN': 'knn_scaler.pkl', 
            'SVM': 'svm_scaler.pkl'
        }
        
        for model_name, model_file in model_files.items():
            model_path = self.models_dir / model_file
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"  ✅ {model_name} 模型加载成功")
                    
                    # 加载对应的标准化器
                    if model_name in scaler_files:
                        scaler_path = self.models_dir / scaler_files[model_name]
                        if scaler_path.exists():
                            with open(scaler_path, 'rb') as f:
                                self.scalers[model_name] = pickle.load(f)
                            print(f"  ✅ {model_name} 标准化器加载成功")
                        
                except Exception as e:
                    print(f"  ❌ {model_name} 模型加载失败: {e}")
            else:
                print(f"  ⚠️  {model_name} 模型文件不存在: {model_file}")
        
        if not self.models:
            print("❌ 没有成功加载任何模型")
            return False
            
        print(f"✅ 成功加载 {len(self.models)} 个模型")
        return True
    
    def extract_features_from_file(self, file_path):
        """从文件提取原始123维特征"""
        try:
            features_list = self.extractor.extract_features_from_file(file_path)
            if features_list is None:
                return None
            
            # 提取数值特征 (跳过文件名)
            features = np.array(features_list[1:]).reshape(1, -1)
            
            if features.shape[1] != 123:
                print(f"  ⚠️  {file_path.name}: 特征维度错误 - 期望123，实际{features.shape[1]}")
                return None
            
            return features
            
        except Exception as e:
            print(f"  ❌ {file_path.name}: 特征提取失败 - {e}")
            return None
    
    def predict_file(self, file_path):
        """预测单个文件"""
        features = self.extract_features_from_file(file_path)
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
        
        return results
    

    def is_office_file(self, file_path):
        """判断文件是否为Office文件（通过文件头判断）"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)

            # Office文件的魔数签名
            office_signatures = [
                b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',  # OLE2 (老版本Office)
                b'PK\x03\x04',  # ZIP格式 (新版本Office)
                b'PK\x05\x06',  # ZIP格式变体
                b'PK\x07\x08'   # ZIP格式变体
            ]

            for signature in office_signatures:
                if header.startswith(signature):
                    return True
            return False
        except:
            return False

    def test_good2bad_folder(self, folder_path='good2bad', save_files=True):
        """测试文件夹中的所有文件（包括无后缀名的Office文件）"""
        folder_path = Path(folder_path)

        if not folder_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return

        # 获取所有文件 (跳过文件头检测，直接处理所有文件)
        all_files = [f for f in folder_path.iterdir() if f.is_file()]
        office_files = all_files  # 直接处理所有文件，让特征提取器判断

        if not office_files:
            print(f"❌ 在 {folder_path} 中未找到任何文件")
            return

        # 输出文件夹中的所有文件名
        print(f"📋 文件夹 '{folder_path}' 中的所有文件列表:")
        print("=" * 80)
        for i, file_path in enumerate(office_files, 1):
            print(f"  {i:3d}. 📄 {file_path.name}")

        print(f"\n🔍 开始测试 {len(office_files)} 个文件...")
        print("=" * 80)

        # 统计结果
        total_files = len(office_files)
        successful_predictions = 0
        model_stats = {model_name: {'malicious': 0, 'benign': 0, 'errors': 0}
                      for model_name in self.models.keys()}
        
        # 新增统计
        ensemble_stats = {
            'rf_benign': 0,  # RF判断为良性
            'rf_malicious_others_benign': 0,  # RF判断为恶意，但其他模型多数投票为良性
            'rf_malicious_others_malicious': 0,  # RF判断为恶意，且其他模型多数投票为恶意
            'rf_error': 0  # RF预测失败
        }

        # 详细结果
        detailed_results = []

        # RandomForest检测出的恶意文件列表
        rf_malicious_files = []

        for i, file_path in enumerate(office_files, 1):
            print(f"\n[{i}/{total_files}] 🔍 分析: {file_path.name}")
            
            results = self.predict_file(file_path)
            
            if results:
                successful_predictions += 1
                
                # 显示预测结果
                print("  📊 预测结果:")
                
                # 首先检查RandomForest的结果
                if 'RandomForest' not in results or results['RandomForest']['prediction'] is None:
                    print(f"    RandomForest: 预测失败")
                    ensemble_stats['rf_error'] += 1
                    consensus = "❓ 不确定 (RF预测失败)"
                elif results['RandomForest']['prediction'] == 0:
                    # RandomForest认为是良性
                    print(f"    RandomForest: 良性 (置信度: {results['RandomForest']['confidence']:.3f})")
                    model_stats['RandomForest']['benign'] += 1
                    ensemble_stats['rf_benign'] += 1
                    consensus = "✅ 良性 (RF决定)"
                elif results['RandomForest']['confidence'] < 0.6:
                    # RandomForest判断为恶意但置信度较低，可能是复杂表格误报
                    print(f"    RandomForest: 恶意 (置信度较低: {results['RandomForest']['confidence']:.3f})")
                    model_stats['RandomForest']['benign'] += 1  # 归类为良性
                    ensemble_stats['rf_benign'] += 1
                    consensus = "✅ 良性 (RF置信度不足)"
                else:
                    # RandomForest认为是恶意，需要检查其他模型
                    print(f"    RandomForest: 恶意 (置信度: {results['RandomForest']['confidence']:.3f})")
                    model_stats['RandomForest']['malicious'] += 1
                    rf_malicious_files.append(file_path)
                    
                    # 统计其他模型的投票
                    other_votes = {'malicious': 0, 'benign': 0, 'error': 0}
                    
                    # 显示其他模型的结果并统计
                    for model_name, result in results.items():
                        if model_name != 'RandomForest':
                            if result['prediction'] is not None:
                                label = result['label']
                                confidence = f" (置信度: {result['confidence']:.3f})" if result['confidence'] else ""
                                print(f"    {model_name:12}: {label}{confidence}")
                                
                                # 统计
                                if result['prediction'] == 1:
                                    model_stats[model_name]['malicious'] += 1
                                    other_votes['malicious'] += 1
                                else:
                                    model_stats[model_name]['benign'] += 1
                                    other_votes['benign'] += 1
                            else:
                                model_stats[model_name]['errors'] += 1
                                other_votes['error'] += 1
                                print(f"    {model_name:12}: 预测失败")
                    
                    # 根据其他模型的多数投票决定最终结果
                    if other_votes['malicious'] >= 2:
                        consensus = "🚨 恶意 (RF+其他模型多数)"
                        ensemble_stats['rf_malicious_others_malicious'] += 1
                    else:
                        consensus = "✅ 良性 (其他模型多数否决)"
                        ensemble_stats['rf_malicious_others_benign'] += 1
                
                print(f"  🎯 新集成结果: {consensus}")
                
                # 保存详细结果
                detailed_results.append({
                    'filename': file_path.name,
                    'consensus': consensus,
                    'results': results
                })
            
            else:
                print("  ❌ 分析失败")
        
        # 显示总结
        self.print_summary(total_files, successful_predictions, model_stats, detailed_results, ensemble_stats)

        # 保存RandomForest检测出的恶意文件
        self.save_rf_malicious_files(rf_malicious_files, save_files)
    
    def print_summary(self, total_files, successful_predictions, model_stats, detailed_results, ensemble_stats):
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
        
        # 新增集成结果统计
        print("\n🔄 新集成策略统计:")
        total_ensemble = sum(ensemble_stats.values())
        if total_ensemble > 0:
            print(f"  RandomForest判断为良性: {ensemble_stats['rf_benign']} ({ensemble_stats['rf_benign']/total_ensemble*100:.1f}%)")
            print(f"  RandomForest判断为恶意，其他模型多数为良性: {ensemble_stats['rf_malicious_others_benign']} ({ensemble_stats['rf_malicious_others_benign']/total_ensemble*100:.1f}%)")
            print(f"  RandomForest判断为恶意，其他模型多数为恶意: {ensemble_stats['rf_malicious_others_malicious']} ({ensemble_stats['rf_malicious_others_malicious']/total_ensemble*100:.1f}%)")
            print(f"  RandomForest预测失败: {ensemble_stats['rf_error']} ({ensemble_stats['rf_error']/total_ensemble*100:.1f}%)")
        
        # 最终恶意文件统计
        final_malicious = ensemble_stats['rf_malicious_others_malicious']
        final_benign = ensemble_stats['rf_benign'] + ensemble_stats['rf_malicious_others_benign']
        final_uncertain = ensemble_stats['rf_error']
        
        print(f"\n🎯 最终判断结果:")
        print(f"  恶意文件: {final_malicious} ({final_malicious/total_ensemble*100:.1f}%)")
        print(f"  良性文件: {final_benign} ({final_benign/total_ensemble*100:.1f}%)")
        print(f"  无法判断: {final_uncertain} ({final_uncertain/total_ensemble*100:.1f}%)")
        
        # 保存详细结果到文件
        self.save_results_to_file(detailed_results)

    def save_results_to_file(self, detailed_results):
        """保存详细结果到文件"""
        results_file = Path('good2bad_original_test_results.txt')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("Good2Bad文件夹测试结果 (原始特征格式)\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("集成策略: RandomForest判断为良性则直接判为良性；RandomForest判断为恶意时，其他模型多数投票决定最终结果\n\n")
            
            for result in detailed_results:
                f.write(f"文件: {result['filename']}\n")
                f.write(f"集成结果: {result['consensus']}\n")
                f.write("各模型预测:\n")
                
                for model_name, model_result in result['results'].items():
                    if model_result['prediction'] is not None:
                        confidence = f" (置信度: {model_result['confidence']:.3f})" if model_result['confidence'] else ""
                        f.write(f"  {model_name}: {model_result['label']}{confidence}\n")
                    else:
                        f.write(f"  {model_name}: 预测失败\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
        
        print(f"📄 详细结果已保存到: {results_file}")

    def save_rf_malicious_files(self, rf_malicious_files, save_files=True):
        """保存RandomForest检测出的恶意文件到data/good2bad文件夹"""
        if not rf_malicious_files:
            print("\n📁 RandomForest未检测到恶意文件，无需保存")
            return

        if not save_files:
            print(f"\n📁 RandomForest检测出 {len(rf_malicious_files)} 个恶意文件 (已禁用保存功能)")
            return

        # 创建目标文件夹
        target_dir = Path('data/good2bad')
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 保存RandomForest检测出的 {len(rf_malicious_files)} 个恶意文件到: {target_dir}")
        print("=" * 80)

        saved_count = 0
        for i, source_file in enumerate(rf_malicious_files, 1):
            try:
                # 目标文件路径
                target_file = target_dir / source_file.name

                # 如果目标文件已存在，添加序号
                if target_file.exists():
                    stem = source_file.stem
                    suffix = source_file.suffix
                    counter = 1
                    while target_file.exists():
                        target_file = target_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                # 复制文件
                shutil.copy2(source_file, target_file)
                print(f"  {i:3d}. ✅ {source_file.name} -> {target_file.name}")
                saved_count += 1

            except Exception as e:
                print(f"  {i:3d}. ❌ {source_file.name} 复制失败: {e}")

        print(f"\n🎉 成功保存 {saved_count}/{len(rf_malicious_files)} 个恶意文件到 {target_dir}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='VBA恶意宏检测器 - 使用训练好的模型检测Office文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python test_with_original_features.py                    # 测试默认的good2bad文件夹
  python test_with_original_features.py --folder test_data # 测试test_data文件夹
  python test_with_original_features.py -f my_samples     # 测试my_samples文件夹
        """
    )

    parser.add_argument(
        '--folder', '-f',
        type=str,
        default='good2bad',
        help='要测试的文件夹路径 (默认: good2bad)'
    )

    parser.add_argument(
        '--models-dir', '-m',
        type=str,
        default='models',
        help='模型文件夹路径 (默认: models)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='禁用保存RandomForest检测出的恶意文件'
    )

    args = parser.parse_args()

    print("🎯 VBA恶意宏检测器 (原始特征格式)")
    print("=" * 50)
    print(f"📁 测试文件夹: {args.folder}")
    print(f"🤖 模型目录: {args.models_dir}")
    print()

    # 初始化检测器
    detector = VBAMalwareDetectorOriginal(models_dir=args.models_dir)

    # 加载模型
    if not detector.load_models():
        print("❌ 模型加载失败，请先运行 train_and_save_models.py 训练模型")
        return

    # 测试指定文件夹
    detector.test_good2bad_folder(args.folder, save_files=not args.no_save)

    print("\n🎉 测试完成！")

if __name__ == "__main__":
    main()
