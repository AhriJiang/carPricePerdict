#!/usr/bin/env python
# coding: utf-8

"""
独立测试脚本：使用保存的CatBoost模型预测测试数据
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """
    加载处理过的测试数据和必要的特征信息
    """
    print("正在加载处理过的数据...")
    
    # 定义数据文件路径
    data_dir = 'processed_data'
    
    try:
        # 加载处理过的测试数据
        X_test = joblib.load(os.path.join(data_dir, 'fe_test_data.joblib'))
        test_ids = joblib.load(os.path.join(data_dir, 'fe_sale_ids.joblib'))
        cat_features = joblib.load(os.path.join(data_dir, 'fe_cat_features.joblib'))
        
        print(f"测试数据加载完成，形状: {X_test.shape}")
        print(f"测试ID数量: {len(test_ids)}")
        print(f"分类特征数量: {len(cat_features)}")
        
        return X_test, test_ids, cat_features
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请确保processed_data目录下存在必要的文件")
        raise

def load_model():
    """
    加载保存的CatBoost模型
    """
    print("正在加载模型...")
    
    model_path = 'processed_data/fe_catboost_model.cbm'
    
    try:
        model = CatBoostRegressor()
        model.load_model(model_path)
        print(f"模型加载成功: {model_path}")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保模型文件存在")
        raise

def predict_test_data(model, X_test, test_ids, cat_features):
    """
    预测测试集数据
    
    参数:
    - model: 训练好的CatBoost模型
    - X_test: 测试数据特征
    - test_ids: 测试数据ID
    - cat_features: 分类特征列表
    
    返回:
    - 预测结果数据框
    """
    print("正在预测测试集...")
    
    # 创建测试数据池
    test_pool = Pool(X_test, cat_features=cat_features)
    
    # 预测
    predictions = model.predict(test_pool)
    
    # 创建提交文件
    submit_data = pd.DataFrame({
        'SaleID': test_ids,
        'price': predictions
    })
    
    # 保存预测结果
    output_file = 'predict_test_result.csv'
    submit_data.to_csv(output_file, index=False)
    print(f"预测结果已保存到 {output_file}")
    
    # 打印预测结果的基本统计信息
    print("\n预测结果统计信息:")
    print(f"预测价格均值: {predictions.mean():.2f}")
    print(f"预测价格中位数: {np.median(predictions):.2f}")
    print(f"预测价格最小值: {predictions.min():.2f}")
    print(f"预测价格最大值: {predictions.max():.2f}")
    
    return submit_data

def main():
    """
    主函数
    """
    print("=== CatBoost模型测试预测脚本 ===")
    
    try:
        # 加载处理过的数据
        X_test, test_ids, cat_features = load_processed_data()
        
        # 加载模型
        model = load_model()
        
        # 执行预测
        submit_data = predict_test_data(model, X_test, test_ids, cat_features)
        
        # 显示前几行预测结果
        print("\n预测结果前5行:")
        print(submit_data.head())
        
        print("\n预测完成！")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()