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
        """
        VBA 内建函数使用频率
        
        
        """
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
            #使用oletools库解析Office文件
            vba_parser = olevba.VBA_Parser(str(file_path))
            # 检测是否包含VBA宏
            if vba_parser.detect_vba_macros():
                # 提取VBA宏代码
                vba_code = ""
                for (filename, stream_path, vba_filename, vba_code_part) in vba_parser.extract_macros():
                    if vba_code_part:
                        vba_code += vba_code_part + "\n"
                vba_parser.close()
                return vba_code.strip()
            else:
                # 如果没有VBA宏，则返回空字符串
                vba_parser.close()
                return ""
        except Exception as e:
            # 如果出现异常，则打印错误信息并返回空字符串
            print(f"  ⚠️  VBA提取失败: {e}")
            return ""
    
    def calculate_obfuscation_features(self, vba_code):
        """计算混淆特征 (77个特征，对应列2-78)
        
        混淆特征用于检测VBA宏是否被故意混淆以隐藏其真实意图。
        这些特征反映了代码的结构特性，是检测代码是否被故意混淆的基础指标。
        混淆是恶意宏的常见特征，但混淆本身不一定表示恶意。
        """
        if not vba_code:
            return [0] * 77
        
        features = []
        
        # 1️⃣ 基本混淆特征 (12个特征)
        # 这些特征关注代码的基本结构，能有效区分正常编写的VBA代码和经过混淆的代码
        lines = vba_code.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # 1. NUM_PROC - 过程数量
        # 恶意宏通常会分散功能到多个小过程中，以隐藏整体逻辑
        # 异常多的过程可能表明代码被故意分割以逃避检测
        proc_count = len(re.findall(r'\b(Sub|Function)\s+\w+', vba_code, re.IGNORECASE))
        features.append(proc_count)
        
        # 2. LINE_CNT_SUM_PROC - 代码行数
        # 恶意宏往往比良性宏更长或更短
        # 极短的宏可能只是触发器，极长的可能包含大量混淆代码
        features.append(len(non_empty_lines))
        
        # 3. LINE_LEN - 最大行长度（修改为符合论文描述）
        # 混淆代码通常有非常长的行（如编码的payload）
        max_line_len = max([len(line) for line in non_empty_lines]) if non_empty_lines else 0
        features.append(max_line_len)
        
        # 4. LINE_CONCAT_CNT - 单行最大字符串拼接次数（修改为符合论文描述）
        # 恶意宏常用字符串拼接来隐藏可疑字符串
        # 例如："po"&"wer"&"sh"&"ell"来隐藏"powershell"或"po"+"wer"+"sh"+"ell"
        max_concat_count = max([(line.count('&') + line.count('+')) for line in non_empty_lines]) if non_empty_lines else 0
        features.append(max_concat_count)
        
        # 5. LINE_OPS_CNT - 单行最大算术运算符数量（修改为符合论文描述）
        # 混淆代码通常包含大量运算符
        # 用于动态构建命令或解码payload
        max_ops_count = max([len(re.findall(r'[+\-*/]', line)) for line in non_empty_lines]) if non_empty_lines else 0
        features.append(max_ops_count)
        
        # 6. LINE_PARENTHESE_CNT - 单行最大函数调用括号数量（修改为符合论文描述）
        # 计算与单词相邻的左括号数量，用于检测函数调用
        # 例如：Chr(65)中的(会被计数，但(1+2)中的(不会
        max_func_call_count = max([len(re.findall(r'\b\w+\s*\(', line)) for line in non_empty_lines]) if non_empty_lines else 0
        features.append(max_func_call_count)
        
        # 7. LINE_STR_CNT - 单行最大字符串数量（修改为符合论文描述）
        # 恶意宏通常包含大量字符串
        # 用于构建命令或存储编码的payload
        max_str_count = max([len(re.findall(r'"[^"]*"', line)) for line in non_empty_lines]) if non_empty_lines else 0
        features.append(max_str_count)
        
        # 8-12. PROCEDURE_* 特征 统计每个过程内部的拼接符、运算符、括号、字符串、赋值符数量的最大值。
        # 单个过程内的“高密度混淆”行为（如大量拼接、调用） 
        # 提取所有过程（Sub/Function）
        procedures = re.findall(r'(Sub|Function)\s+\w+.*?End (Sub|Function)', vba_code, flags=re.IGNORECASE | re.DOTALL)

        # 如果没有明确过程结构，就以整个代码为一个过程
        if not procedures:
            procedures = [(None, None, vba_code)]

        # 重构每个过程的代码内容
        procedure_blocks = []
        for match in re.finditer(r'(Sub|Function)\s+\w+.*?End (Sub|Function)', vba_code, flags=re.IGNORECASE | re.DOTALL):
            proc_code = match.group(0)
            procedure_blocks.append(proc_code)

        # 如果没有识别出分块（fallback）
        if not procedure_blocks:
            procedure_blocks = [vba_code]

        # 对每个过程分别计算混淆统计值
        concat_list = []
        ops_list = []
        paren_list = []
        assign_list = []
        str_list = []

        for proc in procedure_blocks:
            concat_list.append(proc.count('&'))
            ops_list.append(len(re.findall(r'[+\-*/=<>]', proc)))
            paren_list.append(proc.count('(') + proc.count(')'))
            assign_list.append(proc.count('='))
            str_list.append(len(re.findall(r'"[^"]*"', proc)))

        # F6–F10: 每类在所有过程中的最大值
        features.extend([
            max(concat_list),
            max(ops_list),
            max(paren_list),
            max(str_list),
            max(assign_list),
        ])


        # features.extend([concat_count, paren_count, ops_count, vba_code.count('='), str_count])
        
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
        """从单个文件提取完整的特征"""
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
            print(f" 特征维度错误: {len(all_features)}")
            return None

        return all_features
    
    def extract_features_from_folder(self, folder_path, output_file=None, save_csv=False, csv_path=None):
        """从文件夹中提取所有文件的特征，不预先筛选文件类型"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return None
        
        # 获取文件夹中的所有文件
        all_files = [f for f in folder_path.iterdir() if f.is_file()]
        
        if not all_files:
            print(f"❌ 在 {folder_path} 中未找到任何文件")
            return None
        
        print(f"🔍 开始提取 {len(all_files)} 个文件的特征...")
        
        # 创建列名
        columns = self.get_feature_names()
        
        # 提取特征
        all_features = []
        successful_count = 0
        
        for i, file_path in enumerate(all_files, 1):
            print(f"[{i}/{len(all_files)}] 处理: {file_path.name}")
            
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
        
        # 保存到CSV文件
        if save_csv:
            if csv_path is None and output_file:
                # 如果没有指定CSV路径但有Excel路径，则使用相同的文件名但扩展名改为.csv
                csv_path = str(Path(output_file).with_suffix('.csv'))
            
            if csv_path:
                self.save_features_to_csv(df, csv_path)
        
        print(f"✅ 成功提取 {successful_count}/{len(all_files)} 个文件的特征")
        return df

    def save_features_to_csv(self, features_df, output_file, encoding='utf-8'):
        """将提取的特征保存为CSV文件
        
        参数:
            features_df (pandas.DataFrame): 包含特征的DataFrame
            output_file (str): 输出CSV文件的路径
            encoding (str, optional): 文件编码. 默认为'utf-8'
        
        返回:
            bool: 保存成功返回True，否则返回False
        """
        try:
            # 确保输出目录存在
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为CSV文件
            features_df.to_csv(output_path, index=False, encoding=encoding)
            print(f"📄 特征已保存到CSV: {output_path}")
            return True
        except Exception as e:
            print(f"❌ 保存CSV文件失败: {e}")
            return False

def main():
    """
# 基本用法，只指定输入文件夹
python original_feature_extractor.py samples

# 指定输出文件
python original_feature_extractor.py samples -o results.xlsx

# 同时保存为CSV格式
python original_feature_extractor.py samples --csv

# 指定CSV文件路径
python original_feature_extractor.py samples --csv --csv-path results.csv

# 完整示例
python original_feature_extractor.py samples -o results/samples_features.xlsx --csv
    
    """
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="VBA特征提取工具")
    parser.add_argument("input_folder", help="包含待分析文件的文件夹路径")
    parser.add_argument("-o", "--output", help="输出Excel文件路径 (默认为input_folder_features.xlsx)", default=None)
    parser.add_argument("--csv", action="store_true", help="同时保存为CSV格式")
    parser.add_argument("--csv-path", help="CSV文件保存路径 (默认与Excel路径相同但扩展名为.csv)", default=None)
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果未指定输出文件，则使用默认名称
    if args.output is None:
        folder_name = Path(args.input_folder).name
        args.output = f"{folder_name}_features.xlsx"
    
    print("🎯 原始VBA特征提取器")
    print("=" * 50)
    print(f"📁 输入文件夹: {args.input_folder}")
    print(f"📄 输出文件: {args.output}")
    if args.csv:
        csv_path = args.csv_path or Path(args.output).with_suffix('.csv')
        print(f"📄 CSV输出: {csv_path}")
    print("=" * 50)
    
    # 创建特征提取器并处理文件
    extractor = OriginalVBAFeatureExtractor()
    df = extractor.extract_features_from_folder(
        args.input_folder, 
        args.output, 
        save_csv=args.csv, 
        csv_path=args.csv_path
    )
    
    if df is not None:
        print(f"\n📊 特征提取完成:")
        print(f"  - 文件数量: {len(df)}")
        print(f"  - 特征维度: {len(df.columns)}")
        output_files = [args.output]
        if args.csv:
            csv_path = args.csv_path or str(Path(args.output).with_suffix('.csv'))
            output_files.append(csv_path)
        print(f"  - 输出文件: {' 和 '.join(output_files)}")

if __name__ == "__main__":
    main()
