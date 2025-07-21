#!/usr/bin/env python3
"""
批量对data文件夹中的所有子文件夹运行特征提取
使用original_feature_extractor.py提取特征并保存为Excel文件
"""

import os
import subprocess
import sys
from pathlib import Path

def run_feature_extraction():
    """对data文件夹中的所有子文件夹运行特征提取"""

    # 定义路径
    data_dir = Path("data")
    extractor_script = "original_feature_extractor.py"

    # 检查data文件夹是否存在
    if not data_dir.exists():
        print(f"❌ 错误: {data_dir} 文件夹不存在")
        return

    # 检查特征提取脚本是否存在
    if not Path(extractor_script).exists():
        print(f"❌ 错误: {extractor_script} 文件不存在")
        return

    # 获取data文件夹中的所有子文件夹
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"❌ 错误: {data_dir} 中没有找到子文件夹")
        return

    print(f"🔍 发现 {len(subdirs)} 个子文件夹:")
    for subdir in subdirs:
        print(f"  📁 {subdir.name}")

    print("\n🚀 开始批量特征提取...")
    print("=" * 60)

    # 对每个子文件夹运行特征提取
    for i, subdir in enumerate(subdirs, 1):
        folder_name = subdir.name
        output_file = data_dir / f"{folder_name}_features.xlsx"

        print(f"\n[{i}/{len(subdirs)}] 📊 处理文件夹: {folder_name}")
        print(f"  📂 输入路径: {subdir}")
        print(f"  📄 输出文件: {output_file}")

        # 构建命令
        cmd = [
            "python",
            extractor_script,
            str(subdir),
            "-o",
            str(output_file)
        ]

        print(f"  🔧 执行命令: {' '.join(cmd)}")

        try:
            # 运行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )

            if result.returncode == 0:
                print(f"  ✅ 成功完成: {folder_name}")
                if output_file.exists():
                    file_size = output_file.stat().st_size / 1024 / 1024  # MB
                    print(f"  📊 输出文件大小: {file_size:.2f} MB")
                else:
                    print(f"  ⚠️  警告: 输出文件未生成")
            else:
                print(f"  ❌ 失败: {folder_name}")
                print(f"  错误信息: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"  ⏰ 超时: {folder_name} (超过1小时)")
        except Exception as e:
            print(f"  ❌ 异常: {folder_name} - {e}")

    print("\n" + "=" * 60)
    print("📈 批量处理完成!")

    # 显示生成的文件
    print("\n📄 生成的特征文件:")
    feature_files = list(data_dir.glob("*_features.xlsx"))
    if feature_files:
        for feature_file in feature_files:
            file_size = feature_file.stat().st_size / 1024 / 1024  # MB
            print(f"  ✅ {feature_file.name} ({file_size:.2f} MB)")
    else:
        print("  ❌ 没有生成任何特征文件")

def main():
    """主函数"""
    print("🎯 批量特征提取工具")
    print("=" * 60)
    print("📋 功能: 对data文件夹中的所有子文件夹运行特征提取")
    print("🔧 使用: python original_feature_extractor.py samples -o results.xlsx")
    print("💾 输出: 保存到data文件夹中，文件名格式为 {文件夹名}_features.xlsx")
    print()

    try:
        run_feature_extraction()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()