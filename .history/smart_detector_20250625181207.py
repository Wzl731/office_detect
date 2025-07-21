#!/usr/bin/env python3
"""
智能VBA恶意宏检测器
集成复杂表格识别，降低误报率
"""

import argparse
from pathlib import Path
from detector import VBAMalwareDetectorOriginal
from analyze_false_positives import FalsePositiveAnalyzer

class SmartVBADetector(VBAMalwareDetectorOriginal):
    def __init__(self, models_dir='models'):
        super().__init__(models_dir)
        self.fp_analyzer = FalsePositiveAnalyzer()
    
    def is_complex_table(self, file_path):
        """判断文件是否为复杂表格"""
        analysis = self.fp_analyzer.analyze_vba_content(file_path)
        return analysis['is_complex_table']
    
    def predict_file_smart(self, file_path):
        """智能预测，考虑复杂表格因素"""
        # 首先进行常规预测
        results = self.predict_file(file_path)
        
        if not results:
            return None
        
        # 检查是否为复杂表格
        is_table = self.is_complex_table(file_path)
        
        # 如果是复杂表格，调整RandomForest的判断
        if is_table and 'RandomForest' in results:
            rf_result = results['RandomForest']
            
            if rf_result['prediction'] == 1:  # RF判断为恶意
                # 对复杂表格提高阈值要求
                if rf_result['confidence'] < 0.8:  # 提高到0.8
                    print(f"    🔄 复杂表格检测: 降低恶意判断 (置信度: {rf_result['confidence']:.3f} < 0.8)")
                    results['RandomForest']['prediction'] = 0
                    results['RandomForest']['label'] = '良性'
                    results['RandomForest']['confidence'] = 1 - rf_result['confidence']
        
        return results
    
    def test_folder_smart(self, folder_path='good2bad', save_files=True):
        """智能测试文件夹"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return
        
        # 获取所有文件
        all_files = [f for f in folder_path.iterdir() if f.is_file()]
        office_files = all_files
        
        if not office_files:
            print(f"❌ 在 {folder_path} 中未找到任何文件")
            return
        
        print(f"📋 文件夹 '{folder_path}' 中的所有文件列表:")
        print("=" * 80)
        for i, file_path in enumerate(office_files, 1):
            print(f"  {i:3d}. 📄 {file_path.name}")
        
        print(f"\n🔍 开始智能测试 {len(office_files)} 个文件...")
        print("=" * 80)
        
        # 统计结果
        total_files = len(office_files)
        successful_predictions = 0
        model_stats = {model_name: {'malicious': 0, 'benign': 0, 'errors': 0}
                      for model_name in self.models.keys()}
        
        # 新增统计
        ensemble_stats = {
            'rf_benign': 0,
            'rf_malicious_others_benign': 0,
            'rf_malicious_others_malicious': 0,
            'rf_error': 0,
            'complex_table_adjusted': 0  # 复杂表格调整次数
        }
        
        detailed_results = []
        rf_malicious_files = []
        
        for i, file_path in enumerate(office_files, 1):
            print(f"\n[{i}/{total_files}] 🔍 智能分析: {file_path.name}")
            
            # 检查是否为复杂表格
            is_table = self.is_complex_table(file_path)
            if is_table:
                print(f"  📊 识别为复杂表格")
            
            results = self.predict_file_smart(file_path)  # 使用智能预测
            
            if results:
                successful_predictions += 1

                # 显示所有模型的预测结果
                print("  📊 预测结果:")

                # 显示所有模型结果并统计
                for model_name, result in results.items():
                    if result['prediction'] is not None:
                        label = result['label']
                        confidence = f" (置信度: {result['confidence']:.3f})" if result['confidence'] else ""
                        print(f"    {model_name:12}: {label}{confidence}")

                        # 统计所有模型的结果
                        if result['prediction'] == 1:
                            model_stats[model_name]['malicious'] += 1
                        else:
                            model_stats[model_name]['benign'] += 1
                    else:
                        model_stats[model_name]['errors'] += 1
                        print(f"    {model_name:12}: 预测失败")

                # 集成决策：先看RandomForest，再看其他模型
                if 'RandomForest' not in results or results['RandomForest']['prediction'] is None:
                    ensemble_stats['rf_error'] += 1
                    consensus = "❓ 不确定 (RF预测失败)"
                elif results['RandomForest']['prediction'] == 0:
                    # RandomForest认为是良性，直接判定为良性
                    ensemble_stats['rf_benign'] += 1
                    consensus = "✅ 良性 (RF决定)"
                else:
                    # RandomForest认为是恶意，检查其他模型的多数投票
                    rf_malicious_files.append(file_path)

                    # 统计其他模型的投票
                    other_votes = {'malicious': 0, 'benign': 0, 'error': 0}
                    for model_name, result in results.items():
                        if model_name != 'RandomForest':
                            if result['prediction'] is not None:
                                if result['prediction'] == 1:
                                    other_votes['malicious'] += 1
                                else:
                                    other_votes['benign'] += 1
                            else:
                                other_votes['error'] += 1

                    # 根据其他模型的多数投票决定最终结果
                    if other_votes['malicious'] >= 2:
                        consensus = "🚨 恶意 (RF+其他模型多数)"
                        ensemble_stats['rf_malicious_others_malicious'] += 1
                    else:
                        consensus = "✅ 良性 (其他模型多数否决)"
                        ensemble_stats['rf_malicious_others_benign'] += 1
                
                print(f"  🎯 智能集成结果: {consensus}")
                
                detailed_results.append({
                    'filename': file_path.name,
                    'consensus': consensus,
                    'is_complex_table': is_table,
                    'results': results
                })
            
            else:
                print("  ❌ 分析失败")
        
        # 显示总结
        self.print_smart_summary(total_files, successful_predictions, model_stats, detailed_results, ensemble_stats)
        
        # 保存RandomForest检测出的恶意文件
        self.save_rf_malicious_files(rf_malicious_files, save_files)
    
    def print_smart_summary(self, total_files, successful_predictions, model_stats, detailed_results, ensemble_stats):
        """打印智能检测总结"""
        print("\n" + "=" * 80)
        print("📈 智能检测总结")
        print("=" * 80)
        
        print(f"📁 总文件数: {total_files}")
        print(f"✅ 成功分析: {successful_predictions}")
        print(f"❌ 分析失败: {total_files - successful_predictions}")
        
        # 复杂表格统计
        complex_tables = len([r for r in detailed_results if r.get('is_complex_table', False)])
        print(f"📊 复杂表格: {complex_tables} ({complex_tables/total_files*100:.1f}%)")
        
        print("\n📊 各模型检测统计 (所有模型参与所有文件检测):")
        for model_name, stats in model_stats.items():
            total = stats['malicious'] + stats['benign'] + stats['errors']
            if total > 0:
                malicious_rate = stats['malicious'] / total * 100
                benign_rate = stats['benign'] / total * 100
                error_rate = stats['errors'] / total * 100

                print(f"  {model_name:12}: 恶意 {stats['malicious']:3d} ({malicious_rate:5.1f}%) | "
                      f"良性 {stats['benign']:3d} ({benign_rate:5.1f}%) | "
                      f"错误 {stats['errors']:3d} ({error_rate:5.1f}%)")

        print("\n🎯 集成决策统计:")
        print(f"  RF判定良性 (直接通过): {ensemble_stats['rf_benign']}")
        print(f"  RF判定恶意 + 其他模型确认: {ensemble_stats['rf_malicious_others_malicious']}")
        print(f"  RF判定恶意 + 其他模型否决: {ensemble_stats['rf_malicious_others_benign']}")
        print(f"  RF预测失败: {ensemble_stats['rf_error']}")

        # 计算最终恶意检出率
        total_malicious_final = ensemble_stats['rf_malicious_others_malicious']
        total_processed = ensemble_stats['rf_benign'] + ensemble_stats['rf_malicious_others_malicious'] + ensemble_stats['rf_malicious_others_benign'] + ensemble_stats['rf_error']
        if total_processed > 0:
            final_malicious_rate = total_malicious_final / total_processed * 100
            print(f"\n🎯 最终恶意检出率: {total_malicious_final}/{total_processed} ({final_malicious_rate:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='智能VBA恶意宏检测器')
    parser.add_argument('--folder', '-f', type=str, default='data/good2bad', help='要测试的文件夹路径')
    parser.add_argument('--no-save', action='store_true', help='禁用保存恶意文件')
    
    args = parser.parse_args()
    
    print("🧠 智能VBA恶意宏检测器")
    print("=" * 50)
    print(f"📁 测试文件夹: {args.folder}")
    print()
    
    detector = SmartVBADetector()
    
    if not detector.load_models():
        print("❌ 模型加载失败")
        return
    
    detector.test_folder_smart(args.folder, save_files=not args.no_save)
    
    print("\n🎉 智能检测完成！")

if __name__ == "__main__":
    main()
