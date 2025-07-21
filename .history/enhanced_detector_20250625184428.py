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
        RF预测后处理决策流程
        
        流程：RF原始预测 → 恶意(1)? → Excel文件? → 复杂表格? → 改判良性
             ↓             ↓           ↓          ↓
           良性(0)      保持恶意     保持恶意    保持恶意
        """
        
        # 第1步：只处理RF恶意预测
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
        
        print(f"    🔍 RF判为恶意 (置信度: {results['RandomForest']['confidence']:.3f})，进入后处理")
        self.stats['rf_malicious'] += 1
        
        # 第2步：Excel文件检测
        is_excel = self._is_excel_file(file_path)
        
        if not is_excel:
            print(f"    📄 非Excel文件，保持恶意判断")
            self._record_decision_path(file_path, rf_prediction, False, False, "keep_original")
            return results
        
        print(f"    📊 识别为Excel文件，继续检测")
        self.stats['excel_files'] += 1
        
        # 第3步：复杂表格检测
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
