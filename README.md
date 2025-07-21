# Office 恶意宏检测系统

基于机器学习的 Microsoft Office 文档恶意宏检测系统。

## 功能特性

- **多模型集成**: 使用随机森林、多层感知机、K近邻和支持向量机的集成分类器
- **VBA特征提取**: 从VBA宏代码中提取全面的特征
- **集成投票**: 结合多个模型的预测结果以提高准确性
- **批量处理**: 处理整个文件夹的Office文档
- **命令行界面**: 易于使用的CLI批量分析工具

## 系统要求

- Python 3.7+
- 依赖包:
  ```
  numpy
  pandas
  scikit-learn
  pickle
  ```

## 安装

1. 克隆仓库:
   ```bash
   git clone https://github.com/Wzl731/office_detect.git
   cd office_detect
   ```

2. 安装依赖:
   ```bash
   pip install numpy pandas scikit-learn
   ```

## 使用方法

### 基本用法

分析Office文档文件夹:

```bash
python detector.py --folder /path/to/documents
```

### 命令行选项

- `--folder, -f`: 包含要分析的Office文档的文件夹路径
- `--models-dir, -m`: 训练模型目录路径 (默认: models_0711)
- `--no-save`: 禁用分类文件保存功能
- `--save-type`: 选择保存类型: `all`(全部), `malicious`(恶意), 或 `benign`(良性) (默认: all)

### 使用示例

```bash
# 分析 'samples' 文件夹中的文档
python detector.py --folder samples

# 使用自定义模型目录
python detector.py --folder samples --models-dir my_models

# 仅保存恶意文件
python detector.py --folder samples --save-type malicious

# 分析但不保存任何文件
python detector.py --folder samples --no-save
```

## 工作原理

1. **特征提取**: 系统从VBA宏代码中提取各种特征，包括:
   - API调用和函数使用情况
   - 字符串模式和关键词
   - 代码结构指标
   - 可疑行为指示器

2. **模型集成**: 四种不同的机器学习模型分析特征:
   - 随机森林 (Random Forest)
   - 多层感知机 (MLP)
   - K近邻算法 (KNN)
   - 支持向量机 (SVM)

3. **投票决策**: 最终分类基于所有模型的多数投票

## 输出结果

系统提供:
- 各个模型的预测结果和置信度分数
- 集成投票结果
- 分类统计信息
- 自动文件组织 (恶意文件保存到 `data/good2bad2`, 良性文件保存到 `data/bad2good2`)

## 模型训练

系统使用位于 `models_0711` 目录中的预训练模型。这些模型在包含良性和恶意Office文档的综合数据集上进行了训练。

## 文件结构

```
office_detect/
├── detector.py              # 主检测脚本
├── original_feature_extractor.py  # 特征提取模块
├── feature222.py           # VBA特征提取器
├── models_0711/            # 预训练模型目录
├── data/                   # 分类文件输出目录
└── README.md              # 本文件
```

## 贡献

欢迎贡献代码！请随时提交问题和功能改进请求。

## 许可证

本项目仅用于教育和研究目的。

## 免责声明

此工具专为合法的安全研究和恶意软件分析目的而设计。用户有责任确保遵守适用的法律法规。
