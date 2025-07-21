#!/usr/bin/env python3
"""
增强版VBA恶意宏检测器
实现RF后处理：恶意 → Excel文件? → 复杂表格? → 改判良性
"""

import re
import argparse
import pandas as pd
from pathlib import Path
from detector import VBAMalwareDetectorOriginal

class EnhancedVBADetector(VBAMalwareDetectorOriginal):
    def __init__(self, models_dir='models'):
        super().__init__(models_dir)
        
        # 统计记录
        self.decision_records = []
        self.override_records = []
        self.analysis_details = []
        self.excel_files_list = []  # 收集Excel文件列表

        # 统计计数器
        self.stats = {
            'total_files': 0,
            'rf_malicious': 0,
            'rf_benign': 0,
            'excel_files': 0,
            'complex_tables': 0,
            'overridden': 0
        }
    
    def _get_rf_prediction(self, results):
        """获取RandomForest的预测结果：0(良性), 1(恶意), None(失败)"""
        if not results or 'RandomForest' not in results:
            return None
        
        return results['RandomForest']['prediction']
    
    def post_process_rf_prediction(self, results, file_path):
        """
        简化的RF预测后处理决策流程

        流程：RF原始预测 → 恶意(1)? → Excel文件? → 复杂表格? → 改判良性
             ↓             ↓           ↓          ↓
           良性(0)      保持恶意     保持恶意    保持恶意
        """

        # 第1步：只处理RF恶意预测（按照detector.py的判断方式，不使用置信度阈值）
        rf_prediction = self._get_rf_prediction(results)

        if rf_prediction != 1:
            # RF判为良性(0)或预测失败(None) → 直接返回
            if rf_prediction == 0:
                print(f"    ✅ RF判为良性，无需后处理")
                self.stats['rf_benign'] += 1
            else:
                print(f"    ❌ RF预测失败，无法后处理")

            self._record_decision_path(file_path, rf_prediction, False, False, "keep_original")
            return results

        print(f"    🔍 RF判为恶意，进入后处理")
        self.stats['rf_malicious'] += 1

        # 第2步：Excel文件检测（基于文件扩展名，更直接准确）
        is_excel = self._is_excel_file_by_extension(file_path)

        if not is_excel:
            print(f"    📄 非Excel文件，保持恶意判断")
            self._record_decision_path(file_path, rf_prediction, False, False, "keep_original")
            return results

        print(f"    📊 识别为Excel文件，继续检测")
        self.stats['excel_files'] += 1

        # 收集Excel文件信息
        self._collect_excel_file_info(file_path, results)

        # 第3步：复杂表格检测（只对Excel文件进行）
        is_complex = self._is_complex_table_with_analysis(file_path)

        if not is_complex:
            print(f"    ⚠️  非复杂表格，保持恶意判断")
            self._record_decision_path(file_path, rf_prediction, True, False, "keep_original")
            return results

        # 第4步：改判为良性
        print(f"    🔄 识别为复杂表格，改判为良性")
        self.stats['complex_tables'] += 1
        self.stats['overridden'] += 1

        modified_results = self._override_rf_to_benign(results, file_path, "复杂表格")
        self._record_decision_path(file_path, rf_prediction, True, True, "override_to_benign")

        return modified_results

    def _is_excel_file_by_extension(self, file_path):
        """基于文件扩展名检测是否为Excel文件（更直接准确）"""
        excel_extensions = ['.xls', '.xlsx', '.xlsm', '.xlsb']
        file_extension = file_path.suffix.lower()

        is_excel = file_extension in excel_extensions

        print(f"    📊 文件扩展名: {file_extension}, 判断: {'Excel文件' if is_excel else '非Excel文件'}")

        return is_excel

    def _is_excel_file(self, file_path):
        """检测是否为Excel文件（基于VBA内容特征）"""
        vba_code = self.extractor.extract_vba_code(file_path)
        
        if not vba_code:
            print(f"    📄 无VBA代码，非Excel文件")
            return False
        
        # Excel独有特征模式
        excel_patterns = [
            # Excel对象
            r'\bWorksheet\b', r'\bWorkbook\b', r'\bWorksheets\b', r'\bWorkbooks\b',
            # Excel操作
            r'\.Cells\b', r'\.Range\b', r'\.UsedRange\b', r'\.CurrentRegion\b',
            # Excel功能
            r'\.Formula\b', r'\.Calculate\b', r'\.AutoFilter\b',
            # Excel特性
            r'\bPivotTable\b', r'\bChart\b', r'\bChartObject\b'
        ]
        
        # 统计Excel特征
        excel_count = sum(len(re.findall(pattern, vba_code, re.IGNORECASE)) 
                         for pattern in excel_patterns)
        
        # Excel文件判断：特征数 >= 5
        is_excel = excel_count >= 5
        
        print(f"    📊 Excel特征数: {excel_count}, 判断: {'Excel文件' if is_excel else '非Excel文件'}")
        
        return is_excel
    
    def _is_complex_table_with_analysis(self, file_path):
        """检测是否为复杂表格并记录分析详情"""
        vba_code = self.extractor.extract_vba_code(file_path)
        
        if not vba_code:
            return False
        
        # 良性表格特征
        benign_patterns = [
            r'\bWorksheet\b', r'\bRange\b', r'\bCells\b', r'\bSelection\b',
            r'\bSUM\b', r'\bAVERAGE\b', r'\bCOUNT\b', r'\bVLOOKUP\b', r'\bHLOOKUP\b',
            r'\bPivotTable\b', r'\bPivotCache\b', r'\bChart\b', r'\bAutoFilter\b',
            r'\bFormat\b', r'\bSort\b', r'\bAutoFit\b', r'\bCalculate\b',
            r'Application\.', r'ActiveSheet', r'ActiveCell', r'ActiveWorkbook'
        ]
        
        # 恶意行为特征
        malicious_patterns = [
            r'\bShell\b', r'\bCreateObject\b', r'\bWScript\b',
            r'\bDownload\b', r'\bHTTP\b', r'\bURL\b',
            r'\bExecute\b', r'\bCmd\b', r'powershell',
            r'base64', r'decode', r'encode'
        ]
        
        # 统计特征分数
        benign_score = sum(len(re.findall(pattern, vba_code, re.IGNORECASE)) 
                          for pattern in benign_patterns)
        malicious_score = sum(len(re.findall(pattern, vba_code, re.IGNORECASE)) 
                             for pattern in malicious_patterns)
        
        # 复杂表格判断逻辑
        is_complex = self._evaluate_complex_table(benign_score, malicious_score)
        
        print(f"    📊 良性分数: {benign_score}, 恶意分数: {malicious_score}")
        print(f"    📊 复杂表格判断: {'是' if is_complex else '否'}")
        
        # 记录分析详情
        self._record_analysis_details(file_path, benign_score, malicious_score, is_complex, len(vba_code))
        
        return is_complex
    
    def _evaluate_complex_table(self, benign_score, malicious_score):
        """评估是否为复杂表格"""
        
        # 判断条件1：高良性分数且无恶意行为
        condition1 = (benign_score >= 8 and malicious_score == 0)
        
        # 判断条件2：良性分数显著高于恶意分数
        condition2 = (benign_score >= 5 and benign_score >= malicious_score * 3)
        
        # 判断条件3：中等良性分数且恶意分数很低
        condition3 = (benign_score >= 6 and malicious_score <= 1)
        
        is_complex = condition1 or condition2 or condition3
        
        print(f"    📊 判断条件: C1={condition1}, C2={condition2}, C3={condition3} → {is_complex}")
        
        return is_complex
    
    def _override_rf_to_benign(self, results, file_path, reason):
        """将RandomForest结果改判为良性"""
        if 'RandomForest' not in results:
            return results
        
        # 保存原始信息
        original_confidence = results['RandomForest']['confidence']
        
        # 修改RF结果
        results['RandomForest']['prediction'] = 0
        results['RandomForest']['label'] = '良性'
        # 保持原始置信度不变
        
        # 记录改判信息
        self._record_override(file_path, reason, original_confidence)
        
        print(f"    🔄 改判完成: 恶意(置信度:{original_confidence:.3f}) → 良性({reason})")
        
        return results
    
    def _record_decision_path(self, file_path, rf_prediction, is_excel, is_complex, final_action):
        """记录决策路径"""
        self.decision_records.append({
            'filename': file_path.name,
            'rf_prediction': rf_prediction,
            'is_excel': is_excel,
            'is_complex_table': is_complex,
            'final_action': final_action
        })
    
    def _record_override(self, file_path, reason, original_confidence):
        """记录改判信息"""
        self.override_records.append({
            'filename': file_path.name,
            'reason': reason,
            'original_confidence': original_confidence
        })
    
    def _collect_excel_file_info(self, file_path, results):
        """收集Excel文件信息"""
        rf_confidence = results['RandomForest']['confidence']

        excel_info = {
            'filename': file_path.name,
            'rf_confidence': rf_confidence,
            'file_extension': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size
        }

        self.excel_files_list.append(excel_info)
        print(f"    📊 Excel文件收集: {file_path.name} (RF置信度: {rf_confidence:.3f})")

    def _record_analysis_details(self, file_path, benign_score, malicious_score, is_complex, vba_length):
        """记录分析详情"""
        self.analysis_details.append({
            'filename': file_path.name,
            'benign_score': benign_score,
            'malicious_score': malicious_score,
            'is_complex_table': is_complex,
            'vba_length': vba_length
        })
    
    def predict_file_enhanced(self, file_path):
        """增强预测：原始预测 + 后处理"""
        self.stats['total_files'] += 1
        
        # 第1步：获取原始预测结果
        results = self.predict_file(file_path)
        
        if not results:
            return None
        
        # 第2步：后处理
        enhanced_results = self.post_process_rf_prediction(results, file_path)
        
        return enhanced_results

    def test_folder_enhanced(self, folder_path='data/bad250623', save_files=True):
        """增强版文件夹测试"""
        folder_path = Path(folder_path)

        if not folder_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return

        # 获取所有文件
        all_files = [f for f in folder_path.iterdir() if f.is_file()]

        if not all_files:
            print(f"❌ 在 {folder_path} 中未找到任何文件")
            return

        print(f"📋 开始增强检测，共 {len(all_files)} 个文件")
        print("=" * 80)

        # 重置统计
        self.stats = {k: 0 for k in self.stats.keys()}
        self.decision_records = []
        self.override_records = []
        self.analysis_details = []
        self.excel_files_list = []

        # 模型统计
        model_stats = {model_name: {'malicious': 0, 'benign': 0, 'errors': 0}
                      for model_name in self.models.keys()}

        detailed_results = []
        rf_malicious_files = []

        for i, file_path in enumerate(all_files, 1):
            print(f"\n[{i}/{len(all_files)}] 🔍 增强分析: {file_path.name}")

            # 使用增强预测
            results = self.predict_file_enhanced(file_path)

            if results:
                # 显示所有模型结果
                print("  📊 预测结果:")
                consensus_votes = {'malicious': 0, 'benign': 0}

                for model_name, result in results.items():
                    if result['prediction'] is not None:
                        label = result['label']
                        confidence = f" (置信度: {result['confidence']:.3f})" if result['confidence'] else ""
                        print(f"    {model_name:12}: {label}{confidence}")

                        # 统计
                        if result['prediction'] == 1:
                            model_stats[model_name]['malicious'] += 1
                            consensus_votes['malicious'] += 1
                        else:
                            model_stats[model_name]['benign'] += 1
                            consensus_votes['benign'] += 1
                    else:
                        model_stats[model_name]['errors'] += 1
                        print(f"    {model_name:12}: 预测失败")

                # 集成预测结果
                if consensus_votes['malicious'] > consensus_votes['benign']:
                    consensus = "🚨 恶意"
                elif consensus_votes['benign'] > consensus_votes['malicious']:
                    consensus = "✅ 良性"
                else:
                    consensus = "❓ 不确定"

                print(f"  🎯 集成结果: {consensus}")

                # 检查RandomForest是否检测为恶意（用于保存文件）
                if 'RandomForest' in results and results['RandomForest']['prediction'] == 1:
                    rf_malicious_files.append(file_path)

                # 保存详细结果
                detailed_results.append({
                    'filename': file_path.name,
                    'consensus': consensus,
                    'results': results
                })

            else:
                print("  ❌ 分析失败")

        # 显示总结
        self.print_enhanced_summary(len(all_files), model_stats)

        # 保存RandomForest检测出的恶意文件
        self.save_rf_malicious_files(rf_malicious_files, save_files)

        # 生成详细报告
        self.generate_enhanced_reports()

    def print_enhanced_summary(self, total_files, model_stats):
        """打印增强检测总结"""
        print("\n" + "=" * 80)
        print("📈 增强检测总结")
        print("=" * 80)

        print(f"📁 总文件数: {total_files}")
        print(f"✅ 成功分析: {self.stats['total_files']}")

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

        print("\n🔄 后处理统计:")
        print(f"  RF恶意预测: {self.stats['rf_malicious']} 个")
        print(f"  RF良性预测: {self.stats['rf_benign']} 个")
        print(f"  Excel文件: {self.stats['excel_files']} 个")
        print(f"  复杂表格: {self.stats['complex_tables']} 个")
        print(f"  改判文件: {self.stats['overridden']} 个")

        if self.stats['rf_malicious'] > 0:
            excel_rate = self.stats['excel_files'] / self.stats['rf_malicious'] * 100
            complex_rate = self.stats['complex_tables'] / self.stats['rf_malicious'] * 100
            override_rate = self.stats['overridden'] / self.stats['rf_malicious'] * 100

            print(f"\n📊 后处理效果:")
            print(f"  恶意文件中Excel文件比例: {excel_rate:.1f}%")
            print(f"  恶意文件中复杂表格比例: {complex_rate:.1f}%")
            print(f"  恶意文件改判比例: {override_rate:.1f}%")

        # 显示Excel文件详情
        if self.excel_files_list:
            print(f"\n📋 检测到的Excel文件列表 ({len(self.excel_files_list)}个):")
            for i, excel_info in enumerate(self.excel_files_list[:10], 1):
                print(f"  {i:2d}. {excel_info['filename'][:40]:<40} "
                      f"RF置信度: {excel_info['rf_confidence']:.3f} "
                      f"扩展名: {excel_info['file_extension']}")

            if len(self.excel_files_list) > 10:
                print(f"  ... 还有 {len(self.excel_files_list)-10} 个Excel文件")

            # Excel文件置信度分析
            confidences = [f['rf_confidence'] for f in self.excel_files_list]
            print(f"\n📊 Excel文件RF置信度分析:")
            print(f"  最小值: {min(confidences):.3f}")
            print(f"  最大值: {max(confidences):.3f}")
            print(f"  平均值: {sum(confidences)/len(confidences):.3f}")

            # 按扩展名统计
            extensions = {}
            for excel_info in self.excel_files_list:
                ext = excel_info['file_extension']
                extensions[ext] = extensions.get(ext, 0) + 1

            print(f"\n📊 Excel文件扩展名分布:")
            for ext, count in sorted(extensions.items()):
                print(f"  {ext}: {count} 个")

    def generate_enhanced_reports(self):
        """生成增强检测报告"""
        print("\n📄 生成详细报告...")

        # 保存决策路径
        if self.decision_records:
            df_decisions = pd.DataFrame(self.decision_records)
            df_decisions.to_csv('enhanced_decision_paths.csv', index=False)
            print(f"  ✅ 决策路径已保存: enhanced_decision_paths.csv")

        # 保存改判记录
        if self.override_records:
            df_overrides = pd.DataFrame(self.override_records)
            df_overrides.to_csv('enhanced_overrides.csv', index=False)
            print(f"  ✅ 改判记录已保存: enhanced_overrides.csv")

        # 保存分析详情
        if self.analysis_details:
            df_analysis = pd.DataFrame(self.analysis_details)
            df_analysis.to_csv('enhanced_analysis_details.csv', index=False)
            print(f"  ✅ 分析详情已保存: enhanced_analysis_details.csv")

            # 分析复杂表格特征
            self._analyze_complex_table_patterns(df_analysis)

        # 保存Excel文件列表
        if self.excel_files_list:
            df_excel = pd.DataFrame(self.excel_files_list)
            df_excel.to_csv('enhanced_excel_files.csv', index=False)
            print(f"  ✅ Excel文件列表已保存: enhanced_excel_files.csv")

    def _analyze_complex_table_patterns(self, df_analysis):
        """分析复杂表格的模式"""
        complex_files = df_analysis[df_analysis['is_complex_table'] == True]

        if len(complex_files) == 0:
            return

        print(f"\n📊 复杂表格特征分析 ({len(complex_files)}个文件):")

        # 分数分布
        print(f"  良性分数: 最小={complex_files['benign_score'].min()}, "
              f"最大={complex_files['benign_score'].max()}, "
              f"平均={complex_files['benign_score'].mean():.1f}")
        print(f"  恶意分数: 最小={complex_files['malicious_score'].min()}, "
              f"最大={complex_files['malicious_score'].max()}, "
              f"平均={complex_files['malicious_score'].mean():.1f}")

        # 高恶意分数的复杂表格（可能误判）
        high_malicious = complex_files[complex_files['malicious_score'] >= 3]
        if len(high_malicious) > 0:
            print(f"\n⚠️  高恶意分数的复杂表格 ({len(high_malicious)}个，需要审查):")
            for _, row in high_malicious.head(5).iterrows():
                print(f"    {row['filename']}: 良性={row['benign_score']}, 恶意={row['malicious_score']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版VBA恶意宏检测器')
    parser.add_argument('--folder', type=str, default='data/bad250623',
                       help='要测试的文件夹路径')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存检测出的恶意文件')

    args = parser.parse_args()

    # 创建检测器
    detector = EnhancedVBADetector()

    # 加载模型
    if not detector.load_models():
        print("❌ 模型加载失败")
        return

    # 运行增强检测
    detector.test_folder_enhanced(args.folder, save_files=not args.no_save)


if __name__ == "__main__":
    main()
