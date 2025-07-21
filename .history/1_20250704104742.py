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

def update_original_feature_extractor(feature_names):
    """更新original_feature_extractor.py文件"""

    extractor_file = Path('original_feature_extractor.py')

    if not extractor_file.exists():
        print(f"❌ 文件不存在: {extractor_file}")
        return False

    try:
        # 读取原文件内容
        print("📖 正在读取 original_feature_extractor.py...")
        with open(extractor_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 创建get_feature_names方法
        method_code = create_feature_names_method(feature_names)
        if not method_code:
            print("❌ 无法创建特征名称方法")
            return False

        # 找到类定义的位置，在__init__方法后添加新方法
        init_end = content.find('        ]')  # 找到suspicious_keywords列表的结束
        if init_end == -1:
            print("❌ 无法找到__init__方法的结束位置")
            return False

        # 找到下一个方法的开始位置
        next_method_start = content.find('\n    def ', init_end)
        if next_method_start == -1:
            print("❌ 无法找到下一个方法的位置")
            return False

        # 在__init__方法后插入新方法
        new_content = (
            content[:next_method_start] +
            '\n' + method_code +
            content[next_method_start:]
        )

        # 修改extract_features_from_folder方法中的列名创建部分
        old_columns_line = "        columns = ['FILENAME'] + [f'FEATURE_{i+1}' for i in range(77)] + [f'SUSPICIOUS_{i+1}' for i in range(46)]"
        new_columns_line = "        columns = self.get_feature_names()"

        new_content = new_content.replace(old_columns_line, new_columns_line)

        # 写入更新后的内容
        print("💾 正在更新 original_feature_extractor.py...")
        with open(extractor_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print("✅ 成功更新 original_feature_extractor.py")
        return True

    except Exception as e:
        print(f"❌ 更新文件失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 特征名称提取和应用工具")
    print("=" * 60)
    print("📋 功能: 从ds1.xls提取特征名称并应用到original_feature_extractor.py")
    print()

    try:
        # 1. 从ds1.xls提取特征名称
        feature_names = extract_feature_names_from_ds1()
        if not feature_names:
            print("❌ 无法提取特征名称，程序退出")
            sys.exit(1)

        print(f"\n📊 提取到的特征数量: {len(feature_names)}")

        # 2. 更新original_feature_extractor.py
        print("\n🔧 开始更新 original_feature_extractor.py...")
        success = update_original_feature_extractor(feature_names)

        if success:
            print("\n🎉 任务完成!")
            print("✅ 已成功将ds1.xls中的特征名称应用到original_feature_extractor.py")
            print("✅ 现在特征提取器将使用有意义的特征名称而不是通用的FEATURE_1, FEATURE_2等")
        else:
            print("\n❌ 任务失败!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()