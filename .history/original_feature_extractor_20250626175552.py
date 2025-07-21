#!/usr/bin/env python3
"""
原始VBA特征提取器
完全按照原项目的123维特征格式进行特征提取
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from oletools import olevba
import warnings
warnings.filterwarnings('ignore')

class OriginalVBAFeatureExtractor:
    def __init__(self):
        """初始化特征提取器"""
        # VBA函数列表 (65个) - 对应列13-77
        self.vba_functions = [
            'Asc', 'AscB', 'AscW', 'Chr', 'ChrB', 'ChrW', 'Mid', 'Join', 'InStr', 'Replace',
            'Right', 'StrConv', 'Abs', 'Atn', 'Cos', 'Exp', 'Log', 'Hex', 'Oct', 'Str',
            'Val', 'Int', 'Fix', 'Sgn', 'Rnd', 'Sin', 'Sqr', 'Tan', 'CBool', 'CByte',
            'CCur', 'CDate', 'CDbl', 'CDec', 'CInt', 'CLng', 'CLngLng', 'CLngPtr', 'CSng', 'CStr',
            'CVar', 'DDB', 'FV', 'IPmt', 'PV', 'Pmt', 'Rate', 'SLN', 'SYD', 'Array',
            'Strreverse', 'Xor', 'LBound', 'LCase', 'Left', 'LTrim', 'RTrim', 'Trim', 'Space', 'Split',
            'InStrRev', 'UBound', 'UCase', 'Round', 'CallByName'
        ]
        
        # 可疑关键词列表 (46个) - 对应列78-123
        self.suspicious_keywords = [
            'Shell', 'CreateObject', 'GetObject', '.Run', '.Exec', '.Create', 'Kill', '.StartupPath',
            'ShellExecute', 'Shell.Application', 'Binary', 'Lib', 'System', 'Wscript.Shell', 'Document_Open', 'Auto_Open',
            'ShowWindow', 'Workbook_Open', 'Print', 'FileCopy', 'Virtual', 'AutoOpen', 'Open', 'Windows',
            'Write', 'Document_Close', 'Output', 'vbhide', 'ExecuteExcel4Macro', 'SaveToFile', 'Environ', 'CreateTextFile',
            'dde', 'CreateProcessA', 'CreateThread', 'CreateUserThread', 'VirtualAlloc', 'VirtualAllocEx', 'RtlMoveMemory', 'WriteProcessMemory',
            'SetContextThread', 'QueueApcThread', 'WriteVirtualMemory', 'VirtualProtect', 'cmd.exe', 'powershell.exe'
        ]
    
    def extract_vba_code(self, file_path):
        """从Office文件中提取VBA代码"""
        try:
            vba_parser = olevba.VBA_Parser(str(file_path))
            if vba_parser.detect_vba_macros():
                vba_code = ""
                for (filename, stream_path, vba_filename, vba_code_part) in vba_parser.extract_macros():
                    if vba_code_part:
                        vba_code += vba_code_part + "\n"
                vba_parser.close()
                return vba_code.strip()
            else:
                vba_parser.close()
                return ""
        except Exception as e:
            print(f"  ⚠️  VBA提取失败: {e}")
            return ""
    
    def calculate_obfuscation_features(self, vba_code):
        """计算混淆特征 (77个特征，对应列2-78)"""
        if not vba_code:
            return [0] * 77
        
        features = []
        
        # 基本混淆特征 (12个特征)
        lines = vba_code.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # 1. NUM_PROC - 过程数量
        proc_count = len(re.findall(r'\b(Sub|Function)\s+\w+', vba_code, re.IGNORECASE))
        features.append(proc_count)
        
        # 2. LINE_CNT_SUM_PROC - 代码行数
        features.append(len(non_empty_lines))
        
        # 3. LINE_LEN - 平均行长度
        avg_line_len = np.mean([len(line) for line in non_empty_lines]) if non_empty_lines else 0
        features.append(avg_line_len)
        
        # 4. LINE_CONCAT_CNT - 字符串拼接次数
        concat_count = len(re.findall(r'&', vba_code))
        features.append(concat_count)
        
        # 5. LINE_OPS_CNT - 操作符数量
        ops_count = len(re.findall(r'[+\-*/=<>]', vba_code))
        features.append(ops_count)
        
        # 6. LINE_PARENTHESE_CNT - 括号数量
        paren_count = vba_code.count('(') + vba_code.count(')')
        features.append(paren_count)
        
        # 7. LINE_STR_CNT - 字符串数量
        str_count = len(re.findall(r'"[^"]*"', vba_code))
        features.append(str_count)
        
        # 8-12. PROCEDURE_* 特征 (重复一些基本特征)
        features.extend([concat_count, paren_count, ops_count, vba_code.count('='), str_count])
        
        # VBA函数特征 (65个特征，对应列13-77)
        for func in self.vba_functions:
            pattern = r'\b' + re.escape(func) + r'\s*\('
            count = len(re.findall(pattern, vba_code, re.IGNORECASE))
            features.append(count)
        
        return features
    
    def calculate_suspicious_features(self, vba_code):
        """计算可疑关键词特征 (46个特征，对应列78-123)"""
        if not vba_code:
            return [0] * 46

        features = []
        for keyword in self.suspicious_keywords:
            if keyword.startswith('.'):
                # 对于以.开头的关键词，使用特殊匹配
                pattern = re.escape(keyword)
            else:
                # 对于普通关键词，使用单词边界匹配
                pattern = r'\b' + re.escape(keyword) + r'\b'

            count = len(re.findall(pattern, vba_code, re.IGNORECASE))
            features.append(count)

        return features
    
    def extract_features_from_file(self, file_path):
        """从单个文件提取完整的123维特征"""
        file_path = Path(file_path)

        # 提取VBA代码
        vba_code = self.extract_vba_code(file_path)

        # 计算混淆特征 (77个)
        obfuscation_features = self.calculate_obfuscation_features(vba_code)

        # 计算可疑关键词特征 (46个)
        suspicious_features = self.calculate_suspicious_features(vba_code)

        # 组合所有特征: 文件名 + 77 + 46 = 124列
        all_features = [file_path.name] + obfuscation_features + suspicious_features

        if len(all_features) != 124:
            print(f"  ⚠️  特征维度错误: 期望124，实际{len(all_features)}")
            return None

        return all_features
    
    def extract_features_from_folder(self, folder_path, output_file=None):
        """从文件夹中提取所有Office文件的特征"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return None
        
        # 获取所有Office文件
        office_files = []
        for ext in ['*.xls', '*.xlsx', '*.doc', '*.docx']:
            office_files.extend(folder_path.glob(ext))
        
        if not office_files:
            print(f"❌ 在 {folder_path} 中未找到Office文件")
            return None
        
        print(f"🔍 开始提取 {len(office_files)} 个文件的特征...")
        
        # 创建列名 (增强版)
        columns = (['FILENAME'] +
                  [f'FEATURE_{i+1}' for i in range(77)] +
                  [f'SUSPICIOUS_{i+1}' for i in range(46)] +
                  [f'BENIGN_{i+1}' for i in range(15)])
        
        # 提取特征
        all_features = []
        successful_count = 0
        
        for i, file_path in enumerate(office_files, 1):
            print(f"[{i}/{len(office_files)}] 处理: {file_path.name}")
            
            features = self.extract_features_from_file(file_path)
            if features:
                all_features.append(features)
                successful_count += 1
            else:
                print(f"  ❌ 特征提取失败")
        
        if not all_features:
            print("❌ 没有成功提取任何特征")
            return None
        
        # 创建DataFrame
        df = pd.DataFrame(all_features, columns=columns)
        
        # 保存到文件
        if output_file:
            output_path = Path(output_file)
            if output_path.suffix.lower() == '.csv':
                df.to_csv(output_path, index=False)
            else:
                df.to_excel(output_path, index=False)
            print(f"📄 特征已保存到: {output_path}")
        
        print(f"✅ 成功提取 {successful_count}/{len(office_files)} 个文件的特征")
        return df

def main():
    """主函数"""
    print("🎯 原始VBA特征提取器")
    print("=" * 50)
    
    extractor = OriginalVBAFeatureExtractor()
    
    # 提取good2bad文件夹的特征
    df = extractor.extract_features_from_folder('good2bad', 'good2bad_features.xlsx')
    
    if df is not None:
        print(f"\n📊 特征提取完成:")
        print(f"  - 文件数量: {len(df)}")
        print(f"  - 特征维度: {len(df.columns)}")
        print(f"  - 输出文件: good2bad_features.xlsx")

if __name__ == "__main__":
    main()
