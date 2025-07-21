#!/usr/bin/env python3
"""
从ds1.xls中提取特征名称，并将这些名称应用到original_feature_extractor.py中
"""

import pandas as pd
import sys
from pathlib import Path

def extract_feature_names_from_ds1():
    """从ds1.xls中提取特征名称"""

    # 读取ds1.xls文件
    try:
        print("📊 正在读取 ds1.xls 文件...")
        df = pd.read_excel('ds1.xls')

        # 获取列名（特征名称）
        feature_names = df.columns.tolist()

        print(f"✅ 成功读取，共发现 {len(feature_names)} 个特征")
        print("\n📋 特征名称列表:")
        for i, name in enumerate(feature_names, 1):
            print(f"  {i:3d}. {name}")

        return feature_names

    except Exception as e:
        print(f"❌ 读取ds1.xls失败: {e}")
        return None

def create_feature_names_method(feature_names):
    """创建get_feature_names方法的代码"""

    if not feature_names:
        return None

    # 生成方法代码
    method_code = '''    def get_feature_names(self):
        """获取所有特征的名称列表"""
        feature_names = [
'''

    # 添加每个特征名称
    for name in feature_names:
        # 转义特殊字符
        escaped_name = name.replace("'", "\\'").replace('"', '\\"')
        method_code += f"            '{escaped_name}',\n"

    method_code += '''        ]
        return feature_names
'''

    return method_code