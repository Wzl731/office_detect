#!/usr/bin/env python3
"""
为数据集添加标签列
处理原始的无标签数据集，根据样本顺序添加标签：前面是良性(0)，后面是恶意(1)
"""

import pandas as pd
import argparse
from pathlib import Path


def add_labels_to_dataset(input_file, output_file, benign_count, malicious_count=None):
    """
    为数据集添加标签列
    
    Args:
        input_file: 输入数据集文件路径
        output_file: 输出数据集文件路径  
        benign_count: 良性样本数量
        malicious_count: 恶意样本数量（如果为None，则为剩余所有样本）
    """
    print(f"📊 处理数据集: {input_file}")
    
    try:
        # 读取数据集
        df = pd.read_excel(input_file)
        print(f"  ✅ 数据集加载成功: {len(df)} 条记录, {len(df.columns)} 列")
        
        # 检查数据集大小
        total_samples = len(df)
        if malicious_count is None:
            malicious_count = total_samples - benign_count
        
        expected_total = benign_count + malicious_count
        if total_samples != expected_total:
            print(f"  ⚠️  警告: 数据集总数({total_samples}) != 良性({benign_count}) + 恶意({malicious_count})")
            print(f"      将按实际数据集大小处理")
            if total_samples < benign_count:
                print(f"  ❌ 错误: 数据集总数小于良性样本数")
                return False
            malicious_count = total_samples - benign_count
        
        # 创建标签列
        labels = [0] * benign_count + [1] * malicious_count
        
        # 确保标签数量与数据集大小匹配
        if len(labels) > total_samples:
            labels = labels[:total_samples]
        elif len(labels) < total_samples:
            # 如果标签不够，剩余的都标记为恶意
            labels.extend([1] * (total_samples - len(labels)))
        
        # 添加标签列到第一列
        df.insert(0, 'label', labels)
        
        print(f"  📋 标签统计:")
        print(f"    良性样本: {sum(1 for x in labels if x == 0)}")
        print(f"    恶意样本: {sum(1 for x in labels if x == 1)}")
        print(f"    总样本数: {len(labels)}")
        
        # 保存处理后的数据集
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)
        
        print(f"  💾 已保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False


def process_ds1_and_ds2():
    """处理 ds1.xls 和 ds2.xls"""
    print("🎯 批量处理 ds1.xls 和 ds2.xls")
    print("=" * 50)
    
    # 处理 ds1.xls (包含黑白样本)
    print("\n📊 处理 ds1.xls (黑白样本)")
    success1 = add_labels_to_dataset(
        input_file='ds_date/ds1.xls',
        output_file='ds_date/ds1_labeled.xls', 
        benign_count=2939,  # 根据原始代码中的配置
        malicious_count=13734
    )
    
    # 处理 ds2.xls (只有黑样本)
    print("\n📊 处理 ds2.xls (只有恶意样本)")
    
    # 先检查 ds2.xls 的大小
    try:
        df2 = pd.read_excel('ds_date/ds2.xls')
        ds2_count = len(df2)
        print(f"  ds2.xls 样本数量: {ds2_count}")
        
        success2 = add_labels_to_dataset(
            input_file='ds_date/ds2.xls',
            output_file='ds_date/ds2_labeled.xls',
            benign_count=0,  # ds2 只有恶意样本
            malicious_count=ds2_count
        )
    except Exception as e:
        print(f"  ❌ 无法读取 ds2.xls: {e}")
        success2 = False
    
    if success1 and success2:
        print("\n✅ 所有数据集处理完成！")
        print("📁 输出文件:")
        print("  - ds_date/ds1_labeled.xls (黑白样本)")
        print("  - ds_date/ds2_labeled.xls (恶意样本)")
        
        # 合并数据集
        print("\n🔄 合并数据集...")
        try:
            df1 = pd.read_excel('ds_date/ds1_labeled.xls')
            df2 = pd.read_excel('ds_date/ds2_labeled.xls')
            
            # 确保列名一致
            if list(df1.columns) != list(df2.columns):
                print("  ⚠️  警告: 两个数据集的列名不完全一致")
                # 取交集
                common_cols = list(set(df1.columns) & set(df2.columns))
                df1 = df1[common_cols]
                df2 = df2[common_cols]
                print(f"  使用公共列: {len(common_cols)} 列")
            
            # 合并
            combined_df = pd.concat([df1, df2], ignore_index=True)
            combined_df.to_excel('ds_date/combined_dataset.xls', index=False)
            
            print(f"  ✅ 合并完成: {len(combined_df)} 条记录")
            print(f"    良性样本: {sum(combined_df['label'] == 0)}")
            print(f"    恶意样本: {sum(combined_df['label'] == 1)}")
            print(f"  💾 已保存到: ds_date/combined_dataset.xls")
            
        except Exception as e:
            print(f"  ❌ 合并失败: {e}")
    else:
        print("\n❌ 部分数据集处理失败")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='为数据集添加标签列')
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 单个文件处理命令
    single_parser = subparsers.add_parser('single', help='处理单个数据集文件')
    single_parser.add_argument('--input', '-i', required=True, help='输入数据集文件路径')
    single_parser.add_argument('--output', '-o', required=True, help='输出数据集文件路径')
    single_parser.add_argument('--benign-count', '-b', type=int, required=True, help='良性样本数量')
    single_parser.add_argument('--malicious-count', '-m', type=int, help='恶意样本数量（可选，默认为剩余所有样本）')
    
    # 批量处理命令
    batch_parser = subparsers.add_parser('batch', help='批量处理 ds1.xls 和 ds2.xls')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        success = add_labels_to_dataset(
            args.input, 
            args.output, 
            args.benign_count, 
            args.malicious_count
        )
        if success:
            print("✅ 处理完成！")
        else:
            print("❌ 处理失败！")
            
    elif args.command == 'batch':
        process_ds1_and_ds2()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
