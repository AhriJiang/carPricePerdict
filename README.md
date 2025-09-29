# 二手车价格预测项目

## 项目简介

本项目旨在使用机器学习技术（特别是CatBoost算法）对二手车价格进行预测。项目包含完整的数据处理流程、特征工程方法、模型训练和评估功能，以及独立的预测脚本。

## 主要功能

- **数据预处理**：处理缺失值、异常值和日期格式转换
- **特征工程**：
  - 创建时间相关特征（车龄、注册季节等）
  - 生成车辆相关特征
  - 构建统计特征（品牌级别价格统计等）
  - 实现交叉特征（品牌+车型组合等）
- **模型训练**：使用CatBoost算法训练回归模型
- **模型评估**：计算RMSE、MAE、R2等评估指标
- **特征重要性分析**：可视化展示关键预测特征
- **独立预测**：支持使用保存的模型对新数据进行预测

## 项目结构

```
e:\TianChi\carPrice\
├── feature_engineering_and_catboost.py    # 主脚本：特征工程和模型训练
├── feature_engineering_and_catboost.ipynb # Jupyter Notebook版本
├── predict_with_saved_model.py           # 独立预测脚本
├── processed_data/                       # 处理后的数据和保存的模型
│   ├── fe_X_train.joblib                 # 训练特征数据
│   ├── fe_X_val.joblib                   # 验证特征数据
│   ├── fe_y_train.joblib                 # 训练目标数据
│   ├── fe_y_val.joblib                   # 验证目标数据
│   ├── fe_test_data.joblib               # 测试特征数据
│   ├── fe_sale_ids.joblib                # 测试ID数据
│   ├── fe_cat_features.joblib            # 分类特征列表
│   └── fe_catboost_model.cbm             # 保存的CatBoost模型
├── used_car_train_20200313.csv           # 训练数据集
├── used_car_testB_20200421.csv           # 测试数据集
└── predict_test_result.csv               # 预测结果输出
```

## 技术栈

- **Python 3.7+**
- **Pandas**: 数据处理和分析
- **NumPy**: 科学计算
- **Matplotlib/Seaborn**: 数据可视化
- **CatBoost**: 梯度提升决策树算法
- **Scikit-learn**: 模型评估和数据分割
- **Joblib**: 模型和数据持久化

## 安装说明

### 1. 克隆或下载项目

```bash
git clone <repository_url>
cd e:\TianChi\carPrice
```

### 2. 安装依赖

使用pip安装所需的Python包：

```bash
pip install pandas numpy matplotlib seaborn catboost scikit-learn joblib
```

### 3. 准备数据

确保以下数据文件存在于项目根目录：
- `used_car_train_20200313.csv`: 训练数据集
- `used_car_testB_20200421.csv`: 测试数据集

## 使用方法

### 方法一：运行完整的特征工程和模型训练

```bash
python feature_engineering_and_catboost.py
```

这个脚本会执行：
- 数据加载和预处理
- 特征工程
- 模型训练和评估
- 生成预测结果
- 保存模型和处理后的数据

### 方法二：使用保存的模型进行预测

如果已经训练过模型，可以直接使用独立的预测脚本：

```bash
python predict_with_saved_model.py
```

这个脚本会：
- 加载保存的模型
- 加载处理过的测试数据
- 进行预测并保存结果

## 模型性能

训练好的CatBoost模型在验证集上的性能：

- **均方根误差 (RMSE)**: 1299.96
- **平均绝对误差 (MAE)**: 535.64
- **R2分数**: 0.9688

## 重要特征

模型的前10个重要特征（按重要性排序）：

1. v_3 (25.88%)
2. v_0 (20.22%)
3. v_12 (15.68%)
4. v_8 (3.88%)
5. v_9 (3.20%)
6. v_6 (3.05%)
7. v_10 (2.28%)
8. kilometer_brand_ratio (2.05%)
9. power (1.97%)
10. v_14 (1.84%)

## 贡献指南

欢迎贡献代码和提出问题！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 许可证

本项目采用MIT许可证 - 详情请查看LICENSE文件

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目仓库: <repository_url>

---

*最后更新：2024年*