#!/usr/bin/env python3
"""
调试文件类型检测问题
分析为什么某些文件被识别为RTF
"""

import os
from pathlib import Path
from oletools import olevba

def check_file_headers(file_path):
    """检查文件头信息"""
    file_path = Path(file_path)
    
    print(f"🔍 分析文件: {file_path.name}")
    print("-" * 50)
    
    # 1. 读取文件头
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)  # 读取前32字节
        
        print(f"📄 文件大小: {file_path.stat().st_size} 字节")
        print(f"🔢 文件头 (hex): {header.hex()}")
        print(f"📝 文件头 (ascii): {header[:16]}")
        
        # 2. 检查常见文件格式标识
        if header.startswith(b'{\\rtf1'):
            print("✅ 检测到RTF文件头: {\\rtf1")
        elif header.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
            print("✅ 检测到OLE2文件头 (老版Office)")
        elif header.startswith(b'PK\x03\x04'):
            print("✅ 检测到ZIP文件头 (新版Office)")
        elif header.startswith(b'\x00\x00'):
            print("⚠️  文件头为空字节，可能是损坏文件")
        else:
            print("❓ 未知文件格式")
            
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 3. 使用python-magic检测
    try:
        import magic
        file_type = magic.from_file(str(file_path))
        mime_type = magic.from_file(str(file_path), mime=True)
        print(f"🔮 python-magic检测: {file_type}")
        print(f"📋 MIME类型: {mime_type}")
    except ImportError:
        print("⚠️  python-magic未安装，跳过检测")
    except Exception as e:
        print(f"⚠️  magic检测失败: {e}")
    
    # 4. 使用oletools检测
    try:
        print(f"\n🛠️  oletools检测结果:")
        vba_parser = olevba.VBA_Parser(str(file_path))
        print(f"  文件类型: {vba_parser.type}")
        print(f"  包含VBA: {vba_parser.detect_vba_macros()}")
        vba_parser.close()
    except Exception as e:
        print(f"  ❌ oletools检测失败: {e}")

def analyze_rtf_files_in_folder(folder_path):
    """分析文件夹中被误判为RTF的文件"""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return
    
    print(f"🔍 分析文件夹: {folder_path}")
    print("=" * 60)
    
    # 获取所有文件
    all_files = [f for f in folder_path.iterdir() if f.is_file()]
    rtf_files = []
    
    for file_path in all_files[:20]:  # 限制检查20个文件
        try:
            vba_parser = olevba.VBA_Parser(str(file_path))
            vba_parser.close()
        except Exception as e:
            error_msg = str(e)
            if "is RTF" in error_msg:
                rtf_files.append(file_path)
                print(f"📄 RTF文件: {file_path.name}")
    
    print(f"\n📊 统计结果:")
    print(f"  总文件数: {len(all_files)}")
    print(f"  检查文件数: {min(20, len(all_files))}")
    print(f"  RTF文件数: {len(rtf_files)}")
    
    # 详细分析前几个RTF文件
    if rtf_files:
        print(f"\n🔍 详细分析RTF文件:")
        for i, rtf_file in enumerate(rtf_files[:3], 1):
            print(f"\n--- RTF文件 {i} ---")
            check_file_headers(rtf_file)

def check_specific_files():
    """检查特定的问题文件"""
    # 从终端输出中看到的RTF文件名
    problem_files = [
        "23835e0a5eac9d4c76bd142ed94580afa45d0aefe52dc503d863a3430ad2d159",
        "6b41545c33f90f7123064e7517b402bd05b8fbf8f68ba91749eb09049a528f27",
        "91ecf8c00227e2ccbb1c70d30cfc3aa126e4713c40dbc9662ccf81535fef3a05"
    ]
    
    data_folder = Path("data/sample")
    
    print("🎯 检查特定问题文件:")
    print("=" * 60)
    
    for filename in problem_files:
        file_path = data_folder / filename
        if file_path.exists():
            print(f"\n{'='*20}")
            check_file_headers(file_path)
        else:
            print(f"❌ 文件不存在: {filename}")

def suggest_rtf_handling():
    """建议RTF文件处理方案"""
    print(f"\n💡 RTF文件处理建议:")
    print("=" * 60)
    
    print("🎯 方案1: 跳过RTF文件")
    print("  - 在特征提取前检测文件类型")
    print("  - RTF文件直接返回全零特征向量")
    print("  - 优点: 简单快速")
    print("  - 缺点: 可能遗漏恶意RTF")
    
    print("\n🎯 方案2: 使用rtfobj分析")
    print("  - 使用oletools.rtfobj提取嵌入对象")
    print("  - 对提取的对象进行VBA分析")
    print("  - 优点: 完整分析")
    print("  - 缺点: 复杂度高")
    
    print("\n🎯 方案3: 文件类型预过滤")
    print("  - 在文件夹扫描时过滤RTF文件")
    print("  - 只处理真正的Office文件")
    print("  - 优点: 避免处理问题")
    print("  - 缺点: 需要准确的文件类型检测")

def main():
    """主函数"""
    print("🔍 RTF文件类型检测问题分析")
    print("=" * 60)
    
    # 检查特定问题文件
    check_specific_files()
    
    # 分析sample文件夹中的RTF文件
    analyze_rtf_files_in_folder("data/sample")
    
    # 提供处理建议
    suggest_rtf_handling()

if __name__ == "__main__":
    main()
