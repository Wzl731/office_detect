#!/usr/bin/env python3
"""
分析good2bad文件夹中的误报文件
识别复杂表格的特征，帮助降低误报率
"""

import os
import pandas as pd
from pathlib import Path
from original_feature_extractor import OriginalVBAFeatureExtractor
import re

class FalsePositiveAnalyzer:
    def __init__(self):
        self.extractor = OriginalVBAFeatureExtractor()
        
    def analyze_vba_content(self, file_path):
        """分析VBA内容，识别是否为复杂表格"""
        vba_code = self.extractor.extract_vba_code(file_path)
        
        if not vba_code:
            return {
                'has_vba': False,
                'is_complex_table': False,
                'benign_score': 0,
                'malicious_score': 0
            }
        
        # 良性表格指标
        benign_patterns = [
            r'\bWorksheet\b', r'\bRange\b', r'\bCells\b', r'\bSelection\b',
            r'\bSUM\b', r'\bAVERAGE\b', r'\bCOUNT\b', r'\bVLOOKUP\b',
            r'\bPIVOT\b', r'\bCHART\b', r'\bFORMAT\b', r'\bSort\b',
            r'\bFilter\b', r'\bAutoFit\b', r'\bCalculate\b',
            r'Application\.', r'ActiveSheet', r'ActiveCell'
        ]
        
        # 可疑恶意指标
        malicious_patterns = [
            r'\bShell\b', r'\bCreateObject\b', r'\bWScript\b',
            r'\bDownload\b', r'\bExecute\b', r'\bCmd\b',
            r'powershell', r'base64', r'decode'
        ]
        
        benign_score = 0
        malicious_score = 0
        
        for pattern in benign_patterns:
            benign_score += len(re.findall(pattern, vba_code, re.IGNORECASE))
            
        for pattern in malicious_patterns:
            malicious_score += len(re.findall(pattern, vba_code, re.IGNORECASE))
        
        # 判断是否为复杂表格
        is_complex_table = (benign_score > 5 and malicious_score == 0) or \
                          (benign_score > malicious_score * 3)
        
        return {
            'has_vba': True,
            'is_complex_table': is_complex_table,
            'benign_score': benign_score,
            'malicious_score': malicious_score,
            'vba_length': len(vba_code),
            'vba_lines': len(vba_code.split('\n'))
        }
    
    def analyze_folder(self, folder_path):
        """分析文件夹中的所有文件"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return
        
        files = list(folder_path.iterdir())
        print(f"🔍 分析 {len(files)} 个文件...")
        
        results = []
        complex_tables = []
        
        for i, file_path in enumerate(files, 1):
            if file_path.is_file():
                print(f"[{i}/{len(files)}] 分析: {file_path.name}")
                
                analysis = self.analyze_vba_content(file_path)
                analysis['filename'] = file_path.name
                analysis['file_size'] = file_path.stat().st_size
                
                results.append(analysis)
                
                if analysis['is_complex_table']:
                    complex_tables.append(file_path.name)
                    print(f"  📊 识别为复杂表格 (良性分数: {analysis['benign_score']}, 恶意分数: {analysis['malicious_score']})")
        
        # 生成报告
        self.generate_report(results, complex_tables)
        
        return results, complex_tables
    
    def generate_report(self, results, complex_tables):
        """生成分析报告"""
        print("\n" + "=" * 80)
        print("📈 误报分析报告")
        print("=" * 80)
        
        total_files = len(results)
        has_vba = len([r for r in results if r['has_vba']])
        complex_table_count = len(complex_tables)
        
        print(f"📁 总文件数: {total_files}")
        print(f"🔍 包含VBA的文件: {has_vba}")
        print(f"📊 识别为复杂表格: {complex_table_count}")
        print(f"📊 复杂表格比例: {complex_table_count/total_files*100:.1f}%")
        
        if complex_tables:
            print(f"\n📋 复杂表格文件列表:")
            for i, filename in enumerate(complex_tables[:10], 1):  # 只显示前10个
                print(f"  {i:2d}. {filename}")
            
            if len(complex_tables) > 10:
                print(f"  ... 还有 {len(complex_tables)-10} 个文件")
        
        # 保存详细结果
        df = pd.DataFrame(results)
        report_file = "false_positive_analysis.csv"
        df.to_csv(report_file, index=False)
        print(f"\n📄 详细结果已保存到: {report_file}")

def main():
    analyzer = FalsePositiveAnalyzer()
    
    # 分析good2bad文件夹
    folder_path = "data/good2bad"
    
    if not Path(folder_path).exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        print("请先运行检测器生成good2bad文件夹")
        return
    
    results, complex_tables = analyzer.analyze_folder(folder_path)
    
    print(f"\n🎯 建议:")
    print(f"1. 对于识别出的 {len(complex_tables)} 个复杂表格文件，可以考虑降低检测敏感度")
    print(f"2. 可以将这些文件作为良性样本加入训练集")
    print(f"3. 调整RandomForest的置信度阈值")

if __name__ == "__main__":
    main()
